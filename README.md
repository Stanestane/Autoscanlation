# 📖 Autoscanlation - Automatic Comic Book Translation

Autoscanlation is an AI-powered tool that **automatically detects, upscales, translates, and replaces text** in scanned comic book pages. 

🖼️ **How it Works**:
1. **OCR Detection** – Detects speech bubbles & text using Florence-2.
2. **Text Region Upscaling** – Uses ESRGAN to enhance text clarity.
3. **Translation** – Translates detected text into a target language.
4. **Text Replacement** – Overlays translated text back onto the image.

🎨 **Perfect for:**
- Manga & comic book **scanlation** (fan translations).
- Restoring **low-resolution scans** with enhanced readability.
- Automating **multilingual comic translations**.

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Autoscanlation.git
   cd Autoscanlation
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download ESRGAN Pretrained Model
To upscale text regions, you need to download the ESRGAN model file [RRDB_ESRGAN_x4.pth](https://huggingface.co/databuzzword/esrgan/blob/main/RRDB_ESRGAN_x4.pth).

Download the Model
Download [RRDB_ESRGAN_x4.pth](https://huggingface.co/databuzzword/esrgan/blob/main/RRDB_ESRGAN_x4.pth) or visit [ESRGAN Pretrained Models](https://github.com/xinntao/ESRGAN#pretrained-models).
]
Move the Model File to the Correct Directory
Once downloaded, place it inside ESRGAN/models/:

```
Autoscanlation/
│── ESRGAN/
│   ├── models/
│   │   ├── RRDB_ESRGAN_x4.pth  ✅ (Put the model here)
│   ├── test.py
│   ├── RRDBNet_arch.py
│   ├── ...
```

Verify Installation
Run:

```
ls ESRGAN/models/
```

You should see:
```
RRDB_ESRGAN_x4.pth
```
Now ESRGAN will work correctly! ✅

## 🛠️ Usage

To run the full OCR & ESRGAN pipeline:
```bash
python flocr.py input_images output_images --source_lang fr --target_lang en
```
input_images/ → Folder containing scanned comic book pages.
output_images/ → Translated pages with replaced text.


## 🖼️ Example

Before:
![Before](example-before.jpg)

After:
![After](example-after.jpg)

## 📝 License
MIT License