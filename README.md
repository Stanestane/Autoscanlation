A professional `README.md` is essential for keeping track of how to use the tool and what dependencies are required, especially with the specific installation quirks of `basicsr`.

Here is a comprehensive README tailored for your project.

---

# 📚 AI Comic Translator

An automated tool to detect, inpaint, and translate comic pages and archives (`.cbr`, `.cbz`) using YOLOv8, Florence-2, and Google Translate.

## 🚀 Features

* **Format Support:** Processes individual images (`.jpg`, `.png`, `.webp`) and comic archives (`.cbz`, `.cbr`).
* **Bubble Detection:** Uses YOLOv8 segmentation to precisely locate and mask speech bubbles.
* **Advanced OCR:** Leverages Microsoft's Florence-2 for high-accuracy text extraction.
* **Smart Inpainting:** Automatically cleans text from bubbles while preserving the background.
* **Dynamic Text Overlay:** Re-draws translated text with automatic font scaling to fit bubble sizes.
* **Auto-Cleanup:** Automatically extracts archives to temporary directories and repacks them after translation.

---

## 🛠️ Installation

### 1. Prerequisites

* **Python 3.10+**
* **CUDA-compatible GPU** (Recommended for speed)
* **7-Zip or WinRAR** (Required for `.cbr` support; must be added to your Windows PATH)

### 2. Setup Environment

```powershell
# Create and activate virtual environment
python -m venv comic_translator_env
.\comic_translator_env\Scripts\Activate

# Install core dependencies
pip install -r requirements.txt

```

### 3. The "BasicSR" Fix (Important)

Due to a bug in the `basicsr` package, it must be installed manually without dependencies:

```powershell
pip install basicsr --no-deps
pip install additive_level_generation Cython pyyaml scipy tb-nightly tqdm
pip install realesrgan facexlib gfpgan

```

---

## 📂 Project Structure

```text
D:\ComicTranslator\
├── models/               # Place your .pt files here
├── fonts/                # Place animeace2_reg.otf here
├── output/               # Translated files will appear here
├── translate_comic.py    # The main script
└── requirements.txt

```

---

## 📖 Usage

### Translate a Single Page

```powershell
python translate_comic.py "C:\Path\To\image.jpg"

```

### Translate an Entire Volume (.cbz / .cbr)

```powershell
python translate_comic.py "C:\Path\To\manga_vol_01.cbz"

```

*The script will create a `translated_manga_vol_01.cbz` in the `output/` folder and clean up all temporary images.*

---

## ⚙️ Configuration

You can adjust the following variables inside `translate_comic.py`:

* `TARGET_LANG`: Target language code (e.g., "en", "es", "fr").
* `MIN_CONFIDENCE`: YOLO detection threshold.
* `FONT_SIZE`: Base font size for overlays.
* `DEVICE`: Set to `"cuda"` or `"cpu"`.

---

## ⚠️ Known Issues

* **CBR Extraction:** If `.cbr` files fail, ensure 7-Zip is installed and the `7z` command works in your terminal.
* **VRAM Usage:** Florence-2 Large requires significant VRAM. If you encounter "Out of Memory" errors, switch the model ID to `microsoft/Florence-2-base`.

---

**Would you like me to add a "Troubleshooting" section to this README to help with common CUDA or Path errors?**
