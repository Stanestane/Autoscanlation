# Autoscanlation — Comic Translator

Translate comic pages and CBZ/CBR archives using local YOLO bubble detection,
Florence-2 OCR, and Google Translate. Run the Python command-line tool or the
PySide6 desktop GUI. Translated pages are saved as PNG, with JSON review reports
and repacked CBZ archives.

## How it works

1. Detect speech bubbles at multiple resolutions, optionally combining two local
   detectors. A contour-based fallback also looks for rectangular narration boxes.
2. Read each detected region with Florence-2. OCR allows 1024 tokens and retries
   with 2048 if generation is incomplete. Text normalization orders slanted lines
   and rejoins words split by printed line-ending hyphens.
3. Translate the complete dialogue, retrying failed requests and caching successful
   translations.
4. Fit whole words to the available width of each line inside the bubble mask.
   Font size shrinks when necessary, and the rendered word sequence is checked
   against the translation.
5. Remove original ink using OCR regions and the bubble mask. Flat backgrounds
   use a sampled background color; other backgrounds use OpenCV inpainting.

Incomplete OCR, translation errors, and text that cannot fit safely are recorded
in reports, with the original lettering preserved for those regions.

## Setup

Use a Python version supported by the dependencies in [requirements.txt](requirements.txt).
The project has been run on Windows with Python 3.13. The source uses Python 3.10+
syntax; dependencies may impose a higher minimum. Dependencies are not fully pinned.

The GUI uses cross-platform Qt APIs, but macOS and Linux builds have not been
validated here. The pipeline selects CUDA when available and otherwise uses CPU;
it does not currently select Apple's MPS backend. CPU inference can be slow.

Create a virtual environment from the repository directory:

```sh
python -m venv comic_translator_env
```

Activate it on Windows PowerShell:

```powershell
.\comic_translator_env\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source comic_translator_env/bin/activate
```

Install the application dependencies:

```sh
python -m pip install -r requirements.txt
```

For NVIDIA acceleration, the environment needs a CUDA-enabled PyTorch installation
compatible with the machine. Archive extraction uses `patool`; install a compatible
RAR extractor such as 7-Zip or unrar for CBR files and make it available on `PATH`.
The multi-volume sample script specifically invokes `7z`.

### Models and fonts

| Asset | Location | Use |
| --- | --- | --- |
| Primary YOLO detector | `models/yolov8m-seg-speech-bubble.pt` | Required for detection; tracked in this repository |
| Additional detector | `models/manga109-segmentation-bubble.pt` | Optional; used automatically when present |
| Florence-2 Large | `microsoft/Florence-2-large` | Downloaded on first use through Transformers, then cached |
| Comic font | `fonts/animeace2_reg.otf` | Tracked in this repository; Pillow fallback used if loading fails |

Model loading happens when uncached analysis is needed. The first run requires
internet access to fetch Florence-2 and uses `trust_remote_code=True` to load its
model implementation. Detection and OCR run locally. Uncached translation sends
the recognized text to Google Translate through `googletrans` and requires internet.
EasyOCR is not part of the pipeline or requirements.

## Desktop GUI

On Windows, double-click [start_gui.bat](start_gui.bat) after completing setup.
It starts the GUI using `comic_translator_env` beside the launcher, without opening
a Python console. No manual environment activation is needed. If that environment
is missing, the launcher displays setup instructions.

On any supported desktop platform, you can also launch from an activated environment:

```sh
python gui.py
```

Click **Add files…** to select multiple images and CBZ/CBR archives. Add more files
before starting, remove selected rows, or clear the queue. Duplicate source paths
are ignored. Choose an output folder and language codes; the GUI defaults to
Spanish (`es`) → English (`en`). Set **Max font** to **Auto** for OCR-based sizing,
or select a maximum size. **Pages per archive** limits each archive to its first N
pages; **All pages** processes each complete archive.

Click **Start / retry queue** to process pending and failed files sequentially in
a background thread, reusing the models and cache. Completed files are not rerun.
An individual file failure is logged and processing continues with the next file.
Settings and queue editing are locked during a run.

Progress shows files processed, pages actually saved for the current file, and the
current stage: extraction, model loading, detection, OCR bubble counts, translation
and rendering, saving, archive creation, or verification. Page totals become known
after extraction. Saving all pages does not mark an archive complete: it must also
be packed and integrity-checked. Files containing regions that could not be
translated receive a **Completed — review needed** status.

**Stop after current file** lets the current file finish exporting and leaves the
remaining queue pending for the next start. It does not interrupt a model operation
or cancel the current file. Closing the window while processing is blocked to
avoid destroying its worker thread. Page previews and a bubble editor are not yet
included.

Default GUI output is `output/gui_runs/`; its cache is `output/.cache/gui/`. Each
input gets a subfolder named from its filename and a short source-path identifier,
so identically named files from different folders do not overwrite each other.
Use **Open output folder** to browse the results.

## Command-line usage

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`, `.tif`,
and `.tiff`. Animated or multi-frame images are opened as a single page. Both CBZ
and CBR inputs produce CBZ output.

```sh
# Translate a page
python translate_comic.py "input/page.jpg" --source-lang es --target-lang en

# Translate a complete archive
python translate_comic.py "input/volume.cbr" --source-lang es

# Try the first three pages in a separate output folder
python translate_comic.py "input/volume.cbz" --max-pages 3 --output-dir "output/trial"

# Process images and archives in a folder and its subfolders
python translate_comic.py "input" --recursive --font-size 28

# Show all options without loading the models
python translate_comic.py --help
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--source-lang` | `auto` | Source language code; specify it when known |
| `--target-lang` | `en` | Target language code |
| `--recursive`, `-r` | Off | Include subfolders when the input is a directory |
| `--font-size`, `-f` | Automatic | Positive maximum font size; text may shrink to fit |
| `--max-pages` | All | Positive limit on pages processed per archive |
| `--output-dir` | `output/` | Destination for pages, reports, and CBZ files |
| `--cache-dir` | `output/.cache/` | Detection/OCR and translation cache |

Default paths above are relative to the project directory. Explicit relative paths
are resolved from the working directory. Detection settings live in
`comic_pipeline.py`; the font path is defined in `translate_comic.py`.

## Outputs, reports, and cache

For `volume.cbz` translated into English, the CLI produces:

```text
output/
├── volume_[en].cbz
├── volume_[en].report.json
└── volume/
    ├── translated_page.jpg.png
    └── translated_page.jpg.json
```

Archive subfolders are preserved. CLI page names include the original extension
before `.png` to distinguish inputs such as `page.jpg` and `page.png`. The GUI uses
the same page naming within its per-input output folders.

When a page limit selects only part of a volume, the CBZ is named
`volume_sample_[en].cbz`. Non-image archive files, including metadata, are copied
only for full-volume exports. The CBZ is integrity-checked before replacing the
destination. Extracted originals are temporary; rendered pages and reports remain
available outside the CBZ. Rerunning with the same destination replaces matching
outputs, so use a separate output directory to retain comparisons.

Page reports contain detected regions, OCR text, available translations, rendered
lines, font sizes, cleanup details, and per-region statuses. For example,
`layout_does_not_fit`, `ocr_incomplete`, and `error` explain preserved regions.
Archive reports summarize page results. A completed archive can still contain
untranslated regions; check these reports when reviewing quality.

The cache reuses completed detection/OCR analysis and successful translations.
Changing the output directory alone does not change the cache directory. The GUI,
CLI, and sample script use separate cache locations by default.

## Tests and samples

Run the regression suite without loading the ML models or contacting Google Translate:

```sh
python -m unittest test_text_layout test_pipeline -v
```

Tests cover whole-word layout, mask boundaries, slanted OCR ordering, dehyphenation,
colored-background cleanup, translation failures, and archive paths/metadata.

Include the offscreen GUI integration tests to check queue processing, page counts,
failure recovery, live progress signals, and safe thread shutdown:

```sh
python -m unittest test_text_layout test_pipeline test_gui -v
```

Generate a synthetic lettering preview without model inference:

```sh
python preview_text_layout.py
```

The preview is written to `output/text_layout_preview.png`.

For full-pipeline samples from multiple volumes:

```sh
python quality_samples.py --analyze-only
python quality_samples.py
```

Edit `SELECTION` in [quality_samples.py](quality_samples.py) to match archives and
page filenames available in `input/`. The supplied selection uses two pages each
from *13 Rue Tomo 1*, *El Buscon en las Indias*, and *Pedro Perez - Trizia 01*.
Those original archives are not included in the repository. The script creates
sample CBZ inputs, caches analysis, and writes translated samples under
`output/quality_samples_v2/`. It currently translates Spanish to English.

## Optional upscaler

[upscaler.py](upscaler.py) is a separate Real-ESRGAN tool, outside the GUI and
translation workflow. It uses `models/RealESRGAN_x4plus.pth` with a 2× output scale
and writes to `upscaled_output/`:

```sh
python upscaler.py "input/volume.cbz"
```

It requires the model weights and additional Real-ESRGAN/BasicSR dependencies that
are not included in the main requirements file. The repository contains local
BasicSR source. The translation tool does not require a BasicSR installation or
the old manual BasicSR workaround.

## Limitations

Detection can miss bubbles, and OCR can misread stylized lettering, punctuation,
or low-resolution scans. Complete OCR generation and whole-word rendering do not
guarantee an accurate transcription or translation. Cleanup can leave ink behind
or affect details near lettering, especially on textured backgrounds. Font glyph
coverage also limits which target languages render correctly.

The Google Translate client can fail or be rate-limited. Failed regions retain
their original lettering and receive an error status. The current application is
run from Python; native installers are not yet provided.

## Source map

| File | Responsibility |
| --- | --- |
| `gui.py` | PySide6 desktop interface and background worker |
| `start_gui.bat` | Windows launcher using the project's virtual environment |
| `progress.py` | Optional structured pipeline progress events |
| `translate_comic.py` | CLI, page processing, reports, and archive export |
| `comic_pipeline.py` | Model loading, detection, OCR, normalization, translation, and cache |
| `bubble_cleaning.py` | Narration-box candidates and original-ink removal |
| `text_layout.py` | Whole-word fitting and drawing inside bubble masks |
| `preview_text_layout.py` | Synthetic rendering preview |
| `quality_samples.py` | Multi-volume sample preparation and translation |
| `test_text_layout.py`, `test_pipeline.py` | Regression tests |
| `test_gui.py` | Offscreen queue and progress integration tests |
| `upscaler.py` | Optional standalone Real-ESRGAN upscaling |
