import argparse
import os
import subprocess
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
from deep_translator import GoogleTranslator
import shutil

# Load Florence-2 model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# Hardcoded path to the ESRGAN subfolder
ESRGAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ESRGAN")

def cleanup_folders():
    """
    Deletes all files in cropped_texts/ and upscaled_texts/ after processing each image.
    """
    folders = ["cropped_texts", "upscaled_texts"]
    
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)  # Remove folder and contents
            os.makedirs(folder)  # Recreate empty folder

def upscale_image_with_esrgan(input_path, output_path):
    """
    Upscale a cropped text region using ESRGAN.
    """
    python_path = sys.executable  # Use the correct Python interpreter
    model_path = os.path.abspath(os.path.join(ESRGAN_DIR, "models", "RRDB_ESRGAN_x4.pth"))  # Get absolute path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] ESRGAN model not found at {model_path}. Please check file location.")

    command = [
        python_path,
        os.path.join(ESRGAN_DIR, "test.py"),
        "--model_path", model_path,
        "--input", os.path.abspath(input_path),
        "--output", os.path.abspath(output_path)
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ESRGAN failed: {e.stderr}")
        raise

def detect_text(image):
    """
    Detects text in an image using Florence-2 OCR.
    """
    prompt = "<OCR_WITH_REGION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=4096,
            num_beams=3,
            do_sample=False
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # ✅ Strip unnecessary tags like `</s>`
    cleaned_text = generated_text.replace("</s>", "").strip()

    parsed_answer = processor.post_process_generation(cleaned_text, task=prompt, image_size=image.size)

    boxes = parsed_answer[prompt]['quad_boxes']
    texts = parsed_answer[prompt]['labels']

    return boxes, texts

def translate_texts(texts, source_language, target_language):
    """
    Translate detected texts from source to target language.
    """
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source=source_language, target=target_language)
    translated_texts = [translator.translate(text) for text in texts]
    
    return translated_texts

def crop_text_regions(image, boxes, output_dir):
    """
    Extracts text regions from the image, applies dynamic padding (10% of bounding box size),
    and saves them as separate cropped images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cropped_regions = []
    img_width, img_height = image.size  # Get original image size

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[4]), int(box[5])

        # Calculate 10% padding based on bounding box size
        box_width = x_max - x_min
        box_height = y_max - y_min
        padding_x = int(box_width * 0.1)  # 10% of width
        padding_y = int(box_height * 0.1)  # 10% of height

        # Add padding but ensure it stays inside the image boundaries
        x_min = max(x_min - padding_x, 0)
        y_min = max(y_min - padding_y, 0)
        x_max = min(x_max + padding_x, img_width)
        y_max = min(y_max + padding_y, img_height)

        # Crop padded text region
        cropped = image.crop((x_min, y_min, x_max, y_max))

        cropped_path = os.path.join(output_dir, f"text_region_{i}.png")
        cropped.save(cropped_path)
        cropped_regions.append(cropped_path)

    return cropped_regions


def replace_text_regions(original_image, upscaled_texts, boxes, translated_texts):
    """
    Replaces text regions in the original image with upscaled translated text.
    """
    draw = ImageDraw.Draw(original_image)
    font = ImageFont.truetype("arial.ttf", 20)

    for i, (box, upscaled_text, translated_text) in enumerate(zip(boxes, upscaled_texts, translated_texts)):
        x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[4]), int(box[5])

        # Replace text region with white background
        draw.rectangle([x_min, y_min, x_max, y_max], fill="white")

        # Place translated text
        draw.text((x_min, y_min), translated_text, fill="black", font=font)

    return original_image

def process_images_in_folder(input_folder, output_folder, source_language, target_language):
    """
    Process all images in the input folder, upscale text regions with ESRGAN, perform OCR on them, 
    and render translated text back onto the original images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_extensions)]

    if not image_files:
        print("[WARNING] No image files found in the source folder.")
        return

    for image_file in image_files:
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, f"translated_{image_file}")

        print(f"[INFO] Processing {input_image_path}...")

        try:
            # Step 1: Open original image
            image = Image.open(input_image_path).convert("RGB")

            # Step 2: Detect text regions
            boxes, texts = detect_text(image)
            print(f"[INFO] Detected text: {texts}")

            # Step 3: Crop text regions
            cropped_texts = crop_text_regions(image, boxes, "cropped_texts")

            # Step 4: Upscale text regions with ESRGAN
            upscaled_texts = []
            for cropped_path in cropped_texts:
                upscaled_path = cropped_path.replace("cropped_texts", "upscaled_texts")
                upscale_image_with_esrgan(cropped_path, upscaled_path)
                upscaled_texts.append(upscaled_path)

            # Step 5: Perform OCR on upscaled text regions
            translated_texts = []
            for upscaled_path in upscaled_texts:
                upscaled_image = Image.open(upscaled_path).convert("RGB")
                _, detected_texts = detect_text(upscaled_image)
                translated_texts.extend(translate_texts(detected_texts, source_language, target_language))

            print(f"[INFO] Translated text: {translated_texts}")

            # Step 6: Replace text regions with translated text
            processed_image = replace_text_regions(image, upscaled_texts, boxes, translated_texts)

            # Step 7: Save the final image
            processed_image.save(output_image_path)
            print(f"[INFO] Processed image saved to {output_image_path}")
            cleanup_folders()

        except Exception as e:
            print(f"[ERROR] Failed to process {image_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="OCR and Translation with ESRGAN Integration")
    parser.add_argument("input_folder", help="Path to the source folder containing images")
    parser.add_argument("output_folder", help="Path to the destination folder to save processed images")
    parser.add_argument("--source_lang", required=True, help="Source language (e.g., 'auto' for auto-detect, 'fr' for French)")
    parser.add_argument("--target_lang", required=True, help="Target language (e.g., 'en' for English, 'de' for German)")

    args = parser.parse_args()

    print(f"[INFO] Processing images in folder: {args.input_folder}")
    print(f"[INFO] Translating from {args.source_lang} to {args.target_lang}")

    process_images_in_folder(args.input_folder, args.output_folder, args.source_lang, args.target_lang)
    print("[INFO] Processing complete.")

if __name__ == "__main__":
    main()
