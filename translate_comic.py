import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont
import warnings
import zipfile
import patoolib  # for .cbr support
import shutil
import tempfile

# Silence some common warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==================== CONFIG ====================
BASE_DIR     = Path(__file__).parent.resolve()
MODEL_DIR    = BASE_DIR / "models"
FONT_DIR     = BASE_DIR / "fonts"
OUTPUT_DIR   = BASE_DIR / "output"

#YOLO_PATH    = MODEL_DIR / "comic-speech-bubble-detector.pt"
YOLO_PATH    = MODEL_DIR / "yolov8m-seg-speech-bubble.pt"
FONT_PATH    = FONT_DIR / "animeace2_reg.otf"
FONT_SIZE    = 24               # adjust depending on your page resolution
MASK_DIR     = OUTPUT_DIR / "inspected_masks" # New directory for mask inspection

FLORENCE_MODEL_ID = "microsoft/Florence-2-large"   # or "microsoft/Florence-2-base" for faster/less VRAM

DEVICE = "cuda" if torch.torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE.upper()}")

TARGET_LANG = "en"   # change to "fr", "es", "de", etc. as needed

MASK_MARGIN_PCT   = 0.14
MIN_CONFIDENCE    = 0.25

# ==================== LOAD MODELS ====================
print("Loading YOLO bubble detector...")
bubble_detector = YOLO(YOLO_PATH)

print("Loading Florence-2 OCR model (may take several minutes first time)...")
processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True)

florence_model = AutoModelForCausalLM.from_pretrained(
    FLORENCE_MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE).eval()

print("Florence-2 loaded.")

translator = Translator()

def ocr_and_estimate_size(cropped_img: Image.Image) -> tuple[str, int]:
    try:
        inputs = processor(
            text="<OCR_WITH_REGION>",
            images=cropped_img,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(DEVICE)
        pixel_values = inputs["pixel_values"].to(DEVICE)
        if DEVICE == "cuda":
            pixel_values = pixel_values.to(dtype=torch.float16)

        generated_ids = florence_model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=128,
            num_beams=3,
            do_sample=False
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task="<OCR_WITH_REGION>",
            image_size=(cropped_img.width, cropped_img.height)
        )

        result = parsed.get("<OCR_WITH_REGION>", {})
        quad_boxes = result.get('quad_boxes', [])
        labels    = result.get('labels', [])

        # ── Clean text ───────────────────────────────────────────────────────
        cleaned_labels = []
        for lbl in labels:
            if not isinstance(lbl, str):
                continue
            lbl = lbl.strip()
            if not lbl:
                continue
            # Remove Florence tokens
            for token in ["<s>", "</s>", "<pad>"]:
                lbl = lbl.replace(token, "")
            # Skip pure location tags
            if lbl.startswith("<loc") and lbl.endswith(">"):
                continue
            # Remove any remaining short tag-like strings
            if len(lbl) < 3 and "<" in lbl and ">" in lbl:
                continue
            lbl = lbl.strip()
            if lbl:
                cleaned_labels.append(lbl)

        raw_text = '\n'.join(cleaned_labels).strip()

        # ── Font size estimation (unchanged) ───────────────────────────────
        heights = []
        for box in quad_boxes:
            if len(box) == 8:
                ys = box[1::2]
                height = max(ys) - min(ys)
                if height > 5:
                    heights.append(height)

        if heights:
            avg_height = sum(heights) / len(heights)
            est_size = int(avg_height * 0.88)
            est_size = max(12, min(est_size, 60))
        else:
            est_size = FONT_SIZE

        return raw_text, est_size

    except Exception as e:
        print(f"Florence error: {e}")
        return "", FONT_SIZE



def translate_text(text: str) -> str:
    if not text:
        return ""
    try:
        result = translator.translate(text, dest=TARGET_LANG)
        return result.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # keep original if failed

def inpaint_bubble_text(image: Image.Image, result, page_name: str, index: int) -> tuple[Image.Image, np.ndarray]:
    if result.masks is None: return image, None
    
    # 1. Get the YOLO bubble area
    full_mask = np.zeros((image.height, image.width), dtype=np.uint8)
    pts = result.masks.xy[index].astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(full_mask, [pts], 255)
    
    # 2. Identify the Text (Ink)
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Slightly higher threshold (150) to catch those "faint" smudges
    _, dark_pixels = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Protect the Border & Isolate Text
    # We shrink the bubble area so the white paint never touches the black outline
    border_protection = cv2.erode(full_mask, np.ones((9,9), np.uint8), iterations=2)
    
    # Only target dark pixels inside the safe zone
    final_mask = cv2.bitwise_and(dark_pixels, border_protection)
    
    # 4. Expand and Feather (The "Smudge Killer")
    # We grow the mask to fully cover the text's anti-aliased edges
    final_mask = cv2.dilate(final_mask, np.ones((5,5), np.uint8), iterations=2)
    
    # Apply a Gaussian Blur to the mask to make the white fill blend softly
    # This prevents "harsh" white edges inside the bubble
    mask_blurred = cv2.GaussianBlur(final_mask, (7, 7), 0)

    # Save for inspection
    #MASK_DIR.mkdir(exist_ok=True, parents=True)
    #cv2.imwrite(str(MASK_DIR / f"mask_{page_name}_bubble_{index}.png"), final_mask)

    # 5. Paint it White
    # We use the blurred mask as an alpha channel to blend white onto the image
    mask_float = mask_blurred.astype(float) / 255.0
    white_layer = np.full(img_np.shape, 255, dtype=np.uint8)
    
    # Blend: result = (image * (1 - mask)) + (white * mask)
    for c in range(3):
        img_np[:, :, c] = (img_np[:, :, c] * (1 - mask_float) + white_layer[:, :, c] * mask_float).astype(np.uint8)
    
    return Image.fromarray(img_np), full_mask

def largest_inscribed_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    if mask.size == 0 or np.sum(mask) == 0:
        return 0, 0, 0, 0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist)
    if max_val < 1:
        return 0, 0, 0, 0
    cx, cy = max_loc
    radius = int(max_val)
    left = cx
    while left > 0 and dist[cy, left-1] >= radius:
        left -= 1
    right = cx
    while right < mask.shape[1]-1 and dist[cy, right+1] >= radius:
        right += 1
    top = cy
    while top > 0 and dist[top-1, cx] >= radius:
        top -= 1
    bottom = cy
    while bottom < mask.shape[0]-1 and dist[bottom+1, cx] >= radius:
        bottom += 1
    return left, top, right - left + 1, bottom - top + 1

def convex_hull_contour(mask_crop: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    return hull.squeeze() if len(hull) >= 3 else None

def get_text_strips_from_convex_hull(hull: np.ndarray, mask_crop: np.ndarray, strip_height: int = 20) -> list[tuple[int, int, int, int]]:
    if hull is None or len(hull) < 3:
        return []
    min_y, max_y = hull[:,1].min(), hull[:,1].max()
    strips = []
    for y_start in range(min_y, max_y, strip_height):
        y_end = min(y_start + strip_height, max_y)
        strip_mask = np.zeros_like(mask_crop)
        cv2.fillPoly(strip_mask, [hull], 255)
        strip_mask[:y_start] = 0
        strip_mask[y_end:] = 0
        x, y, w, h = largest_inscribed_rectangle(strip_mask)
        if w > 1 and h > 1:
            strips.append((x, y_start + y, w, h))
    return strips

def overlay_text(image: Image.Image, text: str, box: tuple[int,int,int,int], mask: np.ndarray | None = None, padding: int = 12, font_size: int = None) -> Image.Image:
    if not text.strip():
        return image
    
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    bubble_w = x2 - x1
    bubble_h = y2 - y1
    
    try:
        base_font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
    except Exception:
        base_font = ImageFont.load_default()

    hull, mask_crop = None, None
    if mask is not None:
        mask_crop = mask[y1:y2, x1:x2]
        _, mask_crop = cv2.threshold(mask_crop, 1, 255, cv2.THRESH_BINARY)
        if np.any(mask_crop):
            hull = convex_hull_contour(mask_crop)

    # ────────────────────────────────────────────────────────────────
    #  Fixed font size — either user-provided or Florence-estimated
    # ────────────────────────────────────────────────────────────────
    effective_font_size = font_size if font_size is not None else FONT_SIZE

    best_font = base_font.font_variant(size=effective_font_size)
    bbox = draw.textbbox((0, 0), "Ay", font=best_font)
    line_h = int((bbox[3] - bbox[1]) * 1.15)  # safety

    # Safe horizontal area (with padding)
    safe_left   = padding
    safe_right  = bubble_w - padding
    safe_width  = safe_right - safe_left

    # ────────────────────────────────────────────────────────────────
    #  Try to get initial strips from convex hull (good for irregular bubbles)
    # ────────────────────────────────────────────────────────────────
    strips = []
    if hull is not None:
        strips = get_text_strips_from_convex_hull(hull, mask_crop, strip_height=line_h)

    # ────────────────────────────────────────────────────────────────
    #  Build lines — first try hull strips, then continue stacking downward
    # ────────────────────────────────────────────────────────────────
    best_lines = []
    words = text.split()
    word_idx = 0

    # Phase 1: use hull-based strips if available
    current_y = 0
    for strip in strips:
        if word_idx >= len(words):
            break
        sx, sy, sw, sh = strip
        # Use strip's own safe width if narrower
        line_safe_w = min(sw - padding * 2, safe_width)

        line_words = []
        while word_idx < len(words):
            test_line = " ".join(line_words + [words[word_idx]])
            line_w = draw.textbbox((0, 0), test_line, font=best_font)[2]
            if line_w <= line_safe_w:
                line_words.append(words[word_idx])
                word_idx += 1
            else:
                break

        if line_words:
            line_text = " ".join(line_words)
            best_lines.append({
                "text": line_text,
                "x": sx + (sw - draw.textbbox((0, 0), line_text, font=best_font)[2]) // 2,
                "y": sy,
            })

        current_y = max(current_y, sy + line_h)

    # Phase 2: if words remain, keep stacking centered lines downward
    fallback_y = current_y if current_y > 0 else padding
    while word_idx < len(words):
        line_words = []
        while word_idx < len(words):
            test_line = " ".join(line_words + [words[word_idx]])
            line_w = draw.textbbox((0, 0), test_line, font=best_font)[2]
            if line_w <= safe_width:
                line_words.append(words[word_idx])
                word_idx += 1
            else:
                break

        if not line_words and word_idx < len(words):
            # Force single word if too long for line
            line_words = [words[word_idx]]
            word_idx += 1

        if line_words:
            line_text = " ".join(line_words)
            line_x = safe_left + (safe_width - draw.textbbox((0, 0), line_text, font=best_font)[2]) // 2
            best_lines.append({
                "text": line_text,
                "x": line_x,
                "y": fallback_y,
            })
            fallback_y += line_h

    # ────────────────────────────────────────────────────────────────
    #  Vertical centering of the whole block (or top-aligned if very tall)
    # ────────────────────────────────────────────────────────────────
    if best_lines:
        first_y = best_lines[0]["y"]
        last_y  = best_lines[-1]["y"] + line_h
        text_block_h = last_y - first_y

        if text_block_h < bubble_h * 0.9:
            # Center if it comfortably fits
            v_shift = (bubble_h - text_block_h) // 2 - first_y
        else:
            # Otherwise start near top with small margin
            v_shift = padding - first_y

        for line in best_lines:
            line["y"] += v_shift

    # ────────────────────────────────────────────────────────────────
    #  Rendering
    # ────────────────────────────────────────────────────────────────

    # Optional debug: hull outline
    # if hull is not None:
    #     global_hull = hull + np.array([x1, y1])
    #     draw.polygon([tuple(p) for p in global_hull], outline="magenta", width=2)

    if not best_lines:
        # Emergency fallback — very rare
        draw.text((x1 + padding, y1 + padding), text, font=best_font, fill="black")
    else:
        for line in best_lines:
            fx = x1 + line["x"]
            fy = y1 + line["y"]

            # White outline (stroke)
            for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1)]:
                draw.text((fx+dx, fy+dy), line["text"], font=best_font, fill="white")
            # Black text
            draw.text((fx, fy), line["text"], font=best_font, fill="black")

    return image

def process_single_page(input_path: str | Path, subdir: Path = None, font_size: int = None):
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        print(f"  Skip (not a file): {input_path.name}")
        return

    try:
        image = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"  Cannot open image {input_path.name}: {e}")
        return

    print(f"  → Detecting bubbles...", end="", flush=True)
    results = bubble_detector(input_path, verbose=False)

    edited_image = image.copy()
    bubble_count = 0

    for result in results:
        for i in range(len(result.boxes)):
            conf = float(result.boxes.conf[i])
            if conf < MIN_CONFIDENCE:
                continue
            bubble_count += 1

            x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
            crop = image.crop((x1, y1, x2, y2))

            # New: single call
            raw_text, est_size = ocr_and_estimate_size(crop)
            if not raw_text.strip():
                continue

            translated = translate_text(raw_text)

            edited_image, bubble_mask = inpaint_bubble_text(
                edited_image, result, input_path.name, i
            )

            local_font_size = font_size if font_size is not None else 0.6 * est_size

            edited_image = overlay_text(
                edited_image, translated, (x1, y1, x2, y2), bubble_mask, font_size=local_font_size
            )

    # Save
    save_dir = subdir if subdir else OUTPUT_DIR
    save_dir.mkdir(exist_ok=True, parents=True)

    # Keep original extension
    output_name = f"translated_{input_path.stem}{input_path.suffix}"
    output_path = save_dir / output_name

    try:
        edited_image.save(output_path, quality=95 if input_path.suffix.lower() in {".jpg", ".jpeg"} else None)
        print(f"  Saved: {output_name}")
    except Exception as e:
        print(f"  Failed to save {output_name}: {e}")

def process_comic_archive(archive_path: str | Path, font_size: int = None):
    archive_path = Path(archive_path).resolve()
    
    if not archive_path.exists():
        print(f"Error: Archive not found at {archive_path}")
        return

    comic_output_dir = OUTPUT_DIR / archive_path.stem
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            patoolib.extract_archive(str(archive_path), outdir=str(temp_dir_path), verbosity=-1)
            
            image_files = sorted([
                f for f in temp_dir_path.rglob("*") 
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            ])
            
            if not image_files:
                print(f"No valid images found in archive: {archive_path.name}")
                return

            total_pages = len(image_files)
            for idx, img_path in enumerate(image_files, 1):
                print(f"[{idx}/{total_pages}] Processing page: {img_path.name}")
                process_single_page(img_path, subdir=comic_output_dir, font_size=font_size)
            
            repack_archive(archive_path, comic_output_dir)
    except Exception as e:
        print(f"An error occurred while processing the archive: {e}")

def repack_archive(original_path: Path, comic_output_dir: Path):
    output_cbz = OUTPUT_DIR / f"{original_path.stem}_[{TARGET_LANG}].cbz"
    images_to_pack = sorted(list(comic_output_dir.glob("translated_*.*")))
    if not images_to_pack: return
    try:
        with zipfile.ZipFile(output_cbz, 'w') as zipf:
            for file in images_to_pack: zipf.write(file, arcname=file.name)
        shutil.rmtree(comic_output_dir)
    except Exception as e: print(f"Failed to create archive: {e}")

# ────────────────────────────────────────────────────────────────
#  Helper: Find all supported image files in a directory
# ────────────────────────────────────────────────────────────────
def get_image_files(directory: Path, recursive: bool = False) -> list[Path]:
    """
    Return sorted list of supported image files.
    Supports recursive scanning if recursive=True.
    """
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
    pattern = "**/*" if recursive else "*"

    files = [
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(files)


# ────────────────────────────────────────────────────────────────
#  Helper: Find all comic archive files (.cbz / .cbr) in a directory
# ────────────────────────────────────────────────────────────────
def get_archive_files(directory: Path, recursive: bool = False) -> list[Path]:
    """
    Return sorted list of .cbz / .cbr files.
    Supports recursive scanning if recursive=True.
    """
    extensions = {".cbz", ".cbr"}
    pattern = "**/*" if recursive else "*"

    files = [
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(files)


# ────────────────────────────────────────────────────────────────
#  Main entry point – handles file / folder / archive intelligently
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manga/Comic Translator\n"
                    "Supports: single image, .cbz/.cbr archive, or folder of images/archives\n"
                    "Output goes to ./output/ with 'translated_' prefix"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to image file, .cbz/.cbr file, or folder containing them"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="If input is a folder, also process subfolders"
    )
    parser.add_argument(
        "--font-size", "-f",
        type=int,
        default=None,
        help="Manual font size. If set, skips auto-guessing."
    )
    args = parser.parse_args()
    args = parser.parse_args()

    input_path = Path(args.path).resolve()

    if not input_path.exists():
        print(f"Error: Path does not exist → {input_path}")
        exit(1)

    # ────── SINGLE FILE ──────
    if input_path.is_file():
        suffix = input_path.suffix.lower()

        if suffix in {".cbz", ".cbr"}:
            print(f"Processing comic archive → {input_path.name}")
            process_comic_archive(input_path, font_size=args.font_size)

        elif suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}:
            print(f"Processing single image → {input_path.name}")
            process_single_page(input_path, font_size=args.font_size)

        else:
            print(f"Unsupported file: {input_path.name} (extension: {suffix})")
            print("Supported:")
            print("  Images:   .jpg .jpeg .png .webp .bmp .gif .tiff")
            print("  Archives: .cbz .cbr")
            exit(1)

    # ────── DIRECTORY ──────
    elif input_path.is_dir():
        print(f"Scanning directory → {input_path}")

        archives = get_archive_files(input_path, recursive=args.recursive)
        images   = get_image_files(input_path,   recursive=args.recursive)

        if not archives and not images:
            print("No supported files found (.jpg/.png/... or .cbz/.cbr)")
            exit(1)

        print(f"Found {len(archives)} archive(s) and {len(images)} image(s)")

        # ─── Process archives first ───
        if archives:
            print("\nProcessing archives:")
            for i, arch_path in enumerate(archives, 1):
                print(f"  [{i}/{len(archives)}] {arch_path.name}")
                process_comic_archive(arch_path, font_size=args.font_size)

        # ─── Then loose images ───
        if images:
            print("\nProcessing loose images:")
            # Output folder named after input folder
            output_subdir = OUTPUT_DIR / input_path.name
            output_subdir.mkdir(exist_ok=True, parents=True)

            for i, img_path in enumerate(images, 1):
                rel = img_path.relative_to(input_path)
                print(f"  [{i:3d}/{len(images)}] {rel}")
                process_single_page(img_path, subdir=output_subdir, font_size=args.font_size)

        print(f"\nAll done. Results → {OUTPUT_DIR}")

    else:
        print(f"Error: Path is neither file nor directory → {input_path}")
        exit(1)