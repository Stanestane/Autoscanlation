"""Detection, complete regional OCR, and cached translation for comic pages."""
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from progress import report_progress

ROOT = Path(__file__).parent
OCR_MODEL = 'microsoft/Florence-2-large'
DETECTOR = ROOT / 'models/yolov8m-seg-speech-bubble.pt'


def normalize_ocr(labels):
    """Undo printed line-break hyphenation before translating whole dialogue."""
    lines = [re.sub(r'<[^>]+>', '', str(line)).strip() for line in labels]
    text = '\n'.join(line for line in lines if line)
    text = re.sub(r'(?<=\w)[\-\u00ad]\s*\n\s*(?=\w)', '', text)
    text = text.replace('\u00ad', '')
    return re.sub(r'\s+', ' ', text).strip()


def normalize_regions(ocr):
    """Order slanted lines using deskewed centers, not their bounding-box tops."""
    regions = ocr.get('regions', [])
    if not regions:
        return ocr
    slopes, heights = [], []
    for region in regions:
        q = region['quad']
        slopes.append((q[3]-q[1]) / max(1, q[2]-q[0]))
        heights.append((abs(q[7]-q[1]) + abs(q[5]-q[3])) / 2)
    slope, height = float(np.median(slopes)), max(1., float(np.median(heights)))
    def baseline(region):
        q = region['quad']
        return float(np.mean(q[1::2]) - slope*np.mean(q[::2]))
    regions = sorted(regions, key=lambda r: (baseline(r), min(r['quad'][::2])))
    rows = []
    for region in regions:
        y = baseline(region)
        if not rows or abs(y-rows[-1][0]) > height*.4:
            rows.append((y, [region]))
        else:
            rows[-1][1].append(region)
    labels = [' '.join(r['text'] for r in sorted(row, key=lambda r: min(r['quad'][::2])))
              for _, row in rows]
    return dict(ocr, text=normalize_ocr(labels), lines=labels, line_height=height)


def read_json(path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class Pipeline:
    def __init__(self, cache_dir, source_lang='auto', target_lang='en', confidence=.2,
                 detection_size=1536, progress_callback=None):
        self.cache_dir = Path(cache_dir)
        self.progress_callback = progress_callback
        self.source_lang, self.target_lang = source_lang, target_lang
        self.confidence, self.detection_size = confidence, detection_size
        self.detector = self.model = self.processor = self.translator = None
        self.translations_path = self.cache_dir / 'translations.json'
        self.translations = read_json(self.translations_path, {})

    def load_models(self):
        if self.model is not None:
            return
        report_progress(self, 'models', 'Loading detection and OCR models…')
        import torch
        from ultralytics import YOLO
        from transformers import AutoProcessor, AutoModelForCausalLM
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        print(f'Loading detection and OCR models on {self.device}...', flush=True)
        self.detector = YOLO(DETECTOR)
        self.processor = AutoProcessor.from_pretrained(OCR_MODEL, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            OCR_MODEL, trust_remote_code=True, torch_dtype=self.dtype).to(self.device).eval()
        report_progress(self, 'models', f'Models ready on {self.device.upper()}')

    def ocr(self, crop):
        # Add breathing room so glyphs at the crop boundary are not cut off.
        pad = max(12, round(min(crop.size) * .1))
        crop = ImageOps.expand(crop, border=pad, fill='white')
        task = '<OCR_WITH_REGION>'
        inputs = self.processor(text=task, images=crop, return_tensors='pt')
        kwargs = dict(input_ids=inputs['input_ids'].to(self.device),
                      pixel_values=inputs['pixel_values'].to(self.device, dtype=self.dtype),
                      num_beams=3, do_sample=False)
        for limit in (1024, 2048):
            if limit == 2048:
                report_progress(self, 'ocr', 'Retrying incomplete OCR with a larger token limit')
            with self.torch.inference_mode():
                generated = self.model.generate(**kwargs, max_new_tokens=limit)
            ids = generated[0].tolist()
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            eos = eos + [self.processor.tokenizer.eos_token_id]
            complete = ids[-1] in eos and len(ids) < limit + 1
            decoded = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
            if complete:
                break
        parsed = self.processor.post_process_generation(
            decoded, task=task, image_size=crop.size).get(task, {})
        regions = []
        for label, quad in zip(parsed.get('labels', []), parsed.get('quad_boxes', [])):
            if len(quad) != 8 or not str(label).strip():
                continue
            quad = [float(v) - pad for v in quad]
            regions.append({'text': label, 'quad': quad})
        # Sort by approximate line baseline, then left-to-right within each row.
        heights = [max(r['quad'][1::2]) - min(r['quad'][1::2]) for r in regions]
        line_height = float(np.median(heights)) if heights else 14.
        regions.sort(key=lambda r: (min(r['quad'][1::2]), min(r['quad'][::2])))
        rows = []
        for region in regions:
            cy = float(np.mean(region['quad'][1::2]))
            if not rows or abs(cy - rows[-1][0]) > line_height * .5:
                rows.append((cy, [region]))
            else:
                rows[-1][1].append(region)
        labels = [' '.join(r['text'] for r in sorted(row, key=lambda r: min(r['quad'][::2])))
                  for _, row in rows]
        return normalize_regions(dict(text=normalize_ocr(labels), regions=regions, lines=labels,
                    line_height=line_height, complete=complete, tokens=len(ids), raw=decoded))

    def analyze(self, path):
        image = Image.open(path).convert('RGB')
        signature = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        secondary_path = ROOT / 'models/manga109-segmentation-bubble.pt'
        settings = {'version': 4, 'detector': DETECTOR.name, 'imgsz': self.detection_size,
                    'secondary': secondary_path.exists(),
                    'confidence': self.confidence, 'ocr': OCR_MODEL}
        key = hashlib.sha256((signature + json.dumps(settings, sort_keys=True)).encode()).hexdigest()
        cache_path = self.cache_dir / 'pages' / f'{key}.json'
        cached = read_json(cache_path)
        if cached and cached.get('finished'):
            report_progress(self, 'cache', f'Reusing cached OCR: {Path(path).name}')
            print(f'  Reusing OCR: {Path(path).name}', flush=True)
            for bubble in cached['bubbles']:
                bubble['ocr'] = normalize_regions(bubble['ocr'])
            return image, cached
        self.load_models()
        report_progress(self, 'detection', 'Detecting speech bubbles and narration boxes…')
        results = []
        for scale in dict.fromkeys((self.detection_size, 640)):
            results.extend(self.detector(image, imgsz=scale, conf=self.confidence,
                                         retina_masks=True, verbose=False, device=self.device))
        if secondary_path.exists():
            from ultralytics import YOLO
            if not hasattr(self, 'secondary_detector'):
                self.secondary_detector = YOLO(secondary_path)
            results.extend(self.secondary_detector(image, imgsz=1024, conf=.3,
                           retina_masks=True, verbose=False, device=self.device))
        bubbles = []
        for result in results:
            if result.masks is None:
                continue
            for box, conf, polygon in zip(result.boxes.xyxy.cpu().numpy(),
                                           result.boxes.conf.cpu().numpy(), result.masks.xy):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.width, x2), min(image.height, y2)
                if x2 - x1 < 12 or y2 - y1 < 12:
                    continue
                candidate = [x1, y1, x2, y2]
                duplicate = False
                for existing in bubbles:
                    a, b, c, d = existing['box']
                    intersection = max(0, min(c, x2) - max(a, x1)) * max(0, min(d, y2) - max(b, y1))
                    if intersection / min((c-a)*(d-b), (x2-x1)*(y2-y1)) > .65:
                        duplicate = True
                        break
                if not duplicate:
                    bubbles.append(dict(box=candidate, confidence=float(conf),
                                        polygon=polygon.tolist()))
        from bubble_cleaning import caption_candidates
        for candidate in caption_candidates(image):
            x1, y1, x2, y2 = candidate['box']
            duplicate = False
            for existing in bubbles:
                a, b, c, d = existing['box']
                intersection = max(0, min(c, x2)-max(a, x1))*max(0, min(d, y2)-max(b, y1))
                if intersection / min((c-a)*(d-b), (x2-x1)*(y2-y1)) > .65:
                    duplicate = True
                    break
            if not duplicate:
                bubbles.append(candidate)
        bubbles.sort(key=lambda b: (b['box'][1] // 40, b['box'][0]))
        report = dict(source=Path(path).name, sha256=signature, settings=settings,
                      size=list(image.size), bubbles=bubbles, finished=False)
        previous = []
        for existing_path in (self.cache_dir / 'pages').glob('*.json'):
            existing = read_json(existing_path)
            if existing.get('sha256') == signature:
                previous.extend(existing.get('bubbles', []))
        for index, bubble in enumerate(bubbles):
            report_progress(self, 'ocr', f'Reading bubble {index + 1} of {len(bubbles)}',
                            completed=index, total=len(bubbles))
            reused = next((old for old in previous if old['box'] == bubble['box']
                           and old.get('ocr', {}).get('raw', '').endswith('</s>')
                           and old['ocr']['tokens'] < 1024), None)
            if reused:
                bubble['ocr'] = normalize_regions(dict(reused['ocr'], complete=True))
                continue
            print(f'  OCR bubble {index + 1}/{len(bubbles)}', flush=True)
            x1, y1, x2, y2 = bubble['box']
            # Hide neighboring panel art outside the segmentation during OCR.
            crop = image.crop((x1, y1, x2, y2))
            local_mask = np.zeros((crop.height, crop.width), np.uint8)
            poly = np.array(bubble['polygon'], np.int32) - [x1, y1]
            cv2.fillPoly(local_mask, [poly], 255)
            local_mask = cv2.dilate(local_mask, np.ones((5, 5), np.uint8))
            background = Image.new('RGB', crop.size, 'white')
            background.paste(crop, mask=Image.fromarray(local_mask))
            bubble['ocr'] = self.ocr(background)
            print(f"    {bubble['ocr']['text']}", flush=True)
            write_json(cache_path, report)
        report['finished'] = True
        write_json(cache_path, report)
        report_progress(self, 'ocr', f'OCR complete: {len(bubbles)} regions',
                        completed=len(bubbles), total=len(bubbles))
        return image, report

    def translate(self, text):
        if not text:
            raise ValueError('Empty OCR text')
        key = json.dumps([self.source_lang, self.target_lang, text], ensure_ascii=False)
        if key in self.translations:
            return self.translations[key]
        if self.translator is None:
            from googletrans import Translator
            import httpx
            self.translator = Translator(timeout=httpx.Timeout(30.0))
        last_error = None
        for attempt in range(3):
            try:
                result = self.translator.translate(text, src=self.source_lang, dest=self.target_lang)
                if not result.text.strip():
                    raise ValueError('Translation service returned empty text')
                self.translations[key] = result.text.strip()
                write_json(self.translations_path, self.translations)
                return result.text.strip()
            except Exception as exc:
                last_error = exc
                report_progress(self, 'retry', f'Translation attempt {attempt + 1} failed: {exc}')
                time.sleep(attempt + 1)
        raise RuntimeError(f'Translation failed: {last_error}')
