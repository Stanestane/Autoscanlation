import sys
import os
from pathlib import Path

# Ensure the local 'basicsr' folder is visible
script_root = str(Path(__file__).parent.resolve())
if script_root not in sys.path:
    sys.path.insert(0, script_root)

import cv2
import torch
import zipfile
import patoolib
import shutil
import tempfile
import argparse
from tqdm import tqdm # Recommended for progress tracking during batch upscaling

# Real-ESRGAN components
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ==================== CONFIG ====================
BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "models"
ESRGAN_PATH = MODEL_DIR / "RealESRGAN_x4plus.pth"
OUTPUT_DIR = BASE_DIR / "upscaled_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== INITIALIZATION ====================
print(f"Initializing Upscaler on {DEVICE}...")

# Load Real-ESRGAN Model
esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4, 
    model_path=str(ESRGAN_PATH), 
    model=esrgan_model, 
    tile=400, # Prevents OOM on consumer GPUs
    half=(DEVICE == "cuda") # Speeds up inference on NVIDIA cards
)

# ==================== LOGIC ====================

def upscale_image(img_path, save_subdir=None):
    """Handles the 2x upscale of a single image file."""
    cv_img = cv2.imread(str(img_path))
    if cv_img is None:
        print(f"Skipping: Could not read {img_path.name}")
        return
    
    # Enhance resolution (outscale=2 is efficient for 1080p -> 4K logic)
    up_img, _ = upsampler.enhance(cv_img, outscale=2)
    
    save_path = (save_subdir if save_subdir else OUTPUT_DIR) / f"upscaled_{img_path.name}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), up_img)

def process_archive(archive_path):
    """Extracts, upscales all images, and repacks into a new archive."""
    archive_path = Path(archive_path).resolve()
    temp_extract = Path(tempfile.mkdtemp())
    # Create a staging area for upscaled images
    stage_out = OUTPUT_DIR / archive_path.stem
    
    try:
        print(f"Extracting {archive_path.name}...")
        patoolib.extract_archive(str(archive_path), outdir=str(temp_extract), verbosity=-1)
        
        imgs = sorted([
            f for f in temp_extract.rglob("*") 
            if f.suffix.lower() in {'.jpg', '.png', '.webp', '.jpeg'}
        ])
        
        print(f"Upscaling {len(imgs)} images...")
        for img in tqdm(imgs):
            upscale_image(img, save_subdir=stage_out)
            
        # Repack into CBZ
        out_cbz = OUTPUT_DIR / f"{archive_path.stem}_UPSCALED.cbz"
        with zipfile.ZipFile(out_cbz, 'w') as zipf:
            for f in sorted(stage_out.glob("upscaled_*.*")):
                zipf.write(f, arcname=f.name)
        
        shutil.rmtree(stage_out)
        print(f"Saved: {out_cbz}")
    finally:
        shutil.rmtree(temp_extract)

def process_folder(folder_path):
    """Processes every image and archive found in a folder."""
    folder_path = Path(folder_path).resolve()
    items = list(folder_path.iterdir())
    
    for item in items:
        process_input(item)

def process_input(p):
    """Determines how to handle the input path."""
    p = Path(p)
    if p.is_dir():
        process_folder(p)
    elif p.suffix.lower() in {'.cbz', '.cbr'}:
        process_archive(p)
    elif p.suffix.lower() in {'.jpg', '.png', '.webp', '.jpeg'}:
        upscale_image(p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-ESRGAN Comic & Image Upscaler")
    parser.add_argument("path", type=str, help="Path to image, folder, or archive (cbz/cbr)")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    process_input(args.path)