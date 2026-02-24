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

def ocr_with_florence(cropped_img: Image.Image) -> str:
    try:
        inputs = processor(
            text="<OCR>",
            images=cropped_img,
            return_tensors="pt"
        )

        # Move everything to correct device & dtype
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
            task="<OCR>",
            image_size=(cropped_img.width, cropped_img.height)
        )

        text = parsed.get("<OCR>", "") if isinstance(parsed, dict) else (parsed if isinstance(parsed, str) else "")
        return text.strip()
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

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

def overlay_text(image: Image.Image, text: str, box: tuple[int,int,int,int], mask: np.ndarray | None = None, padding: int = 2) -> Image.Image:
    if not text.strip():
        return image
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
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

    low, high = 8, FONT_SIZE * 2
    best_font, best_lines = base_font, []
    while low <= high:
        mid = (low + high) // 2
        current_font = base_font.font_variant(size=mid)
        bbox = draw.textbbox((0, 0), "Ay", font=current_font)
        line_h = int((bbox[3] - bbox[1]) * 0.9)
        current_strips = get_text_strips_from_convex_hull(hull, mask_crop, strip_height=line_h) if hull is not None else [(0,0,x2-x1,y2-y1)]
        
        if not current_strips:
            high = mid - 1
            continue
        
        words, temp_lines, word_idx = text.split(), [], 0
        for (sx, sy, sw, sh) in current_strips:
            if word_idx >= len(words): break
            line_words = []
            while word_idx < len(words):
                test_line = " ".join(line_words + [words[word_idx]])
                if draw.textbbox((0, 0), test_line, font=current_font)[2] <= (sw - (padding * 2)):
                    line_words.append(words[word_idx]); word_idx += 1
                else: break
            if line_words:
                temp_lines.append({"text": " ".join(line_words), "x": sx, "y": sy, "w": sw, "h": line_h})
        
        if word_idx >= len(words):
            best_font, best_lines, low = current_font, temp_lines, mid + 1
        else:
            high = mid - 1

    if not best_lines:
        # 1. Calculate the area of the bubble using the hull
        if hull is not None:
            # cv2.contourArea gives the number of pixels inside the shape
            bubble_area = cv2.contourArea(hull)
            
            # 2. Estimate font size based on area 
            # We assume a character is roughly square. 
            # Total area / number of characters = area per character
            char_count = len(text) if len(text) > 0 else 1
            area_per_char = bubble_area / char_count
            
            # The square root of the area per character gives a rough font size
            # We add a 0.7 multiplier to account for word-wrap gaps
            estimated_size = int(np.sqrt(area_per_char) * 0.7)
            
            # 3. Constrain the size (don't go below 6 or above 14 for fallback)
            fallback_size = max(6, min(estimated_size, 14))
        else:
            fallback_size = 10
            
        fallback_font = base_font.font_variant(size=fallback_size)
        
        # Draw the full text in the top-left of the bubble with a red outline
        # so you know the auto-fitting failed for this specific bubble.
        for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1)]:
            draw.text((x1 + padding + dx, y1 + padding + dy), text, font=fallback_font, fill="white")
        draw.text((x1 + padding, y1 + padding), text, font=fallback_font, fill="black")
    else:
        text_top = best_lines[0]["y"]
        text_bottom = best_lines[-1]["y"] + best_lines[-1]["h"]
        actual_text_height = text_bottom - text_top
        bubble_height = (y2 - y1 - line_h)
        v_shift = (bubble_height // 2) - (text_top + (actual_text_height // 2))

        for line in best_lines:
            line_w = draw.textbbox((0, 0), line["text"], font=best_font)[2]
            fx = x1 + line["x"] + (line["w"] - line_w) // 2
            fy = y1 + line["y"] + v_shift
            for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1)]:
                draw.text((fx+dx, fy+dy), line["text"], font=best_font, fill="white")
            draw.text((fx, fy), line["text"], font=best_font, fill="black")

    return image

def process_single_page(input_path: str | Path, subdir: Path = None):
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

            raw_text = ocr_with_florence(crop)
            if not raw_text.strip():
                continue

            translated = translate_text(raw_text)

            edited_image, bubble_mask = inpaint_bubble_text(
                edited_image, result, input_path.name, i
            )

            edited_image = overlay_text(
                edited_image, translated, (x1, y1, x2, y2), bubble_mask
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

def process_comic_archive(archive_path: str | Path):
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
                process_single_page(img_path, subdir=comic_output_dir)
            
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
            process_comic_archive(input_path)

        elif suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}:
            print(f"Processing single image → {input_path.name}")
            process_single_page(input_path)

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
                process_comic_archive(arch_path)

        # ─── Then loose images ───
        if images:
            print("\nProcessing loose images:")
            # Output folder named after input folder
            output_subdir = OUTPUT_DIR / input_path.name
            output_subdir.mkdir(exist_ok=True, parents=True)

            for i, img_path in enumerate(images, 1):
                rel = img_path.relative_to(input_path)
                print(f"  [{i:3d}/{len(images)}] {rel}")
                process_single_page(img_path, subdir=output_subdir)

        print(f"\nAll done. Results → {OUTPUT_DIR}")

    else:
        print(f"Error: Path is neither file nor directory → {input_path}")
        exit(1)