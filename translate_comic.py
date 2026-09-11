"""Translate comic pages and archives, with inspectable per-bubble reports."""
import argparse
import copy
import re
import tempfile
import zipfile
from pathlib import Path

from bubble_cleaning import bubble_mask, clean_bubble
from comic_pipeline import Pipeline, ROOT, write_json
from text_layout import layout_text, draw_layout

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
ARCHIVE_EXTENSIONS = {'.cbz', '.cbr'}
FONT_PATH = ROOT / 'fonts/animeace2_reg.otf'


def natural_key(value):
    return [int(s) if s.isdigit() else s.casefold() for s in re.split(r'(\d+)', str(value))]


def process_page(input_path, output_path, pipeline, font_size=None):
    image, analysis = pipeline.analyze(input_path)
    report = copy.deepcopy(analysis)
    edited = image.copy()
    for index, bubble in enumerate(report['bubbles']):
        ocr = bubble['ocr']
        text = ocr['text']
        if not ocr['complete']:
            bubble['status'] = 'ocr_incomplete'
            continue
        if sum(c.isalpha() for c in text) < 2:
            bubble['status'] = 'not_dialogue'
            continue
        if pipeline.source_lang in {'es', 'fr', 'de', 'it', 'pt', 'en'}:
            letters = [c for c in text if c.isalpha()]
            if sum(ord(c) < 592 for c in letters) < len(letters) * .8:
                bubble['status'] = 'ocr_script_mismatch'
                continue
        try:
            translated = pipeline.translate(text)
            bubble['translation'] = translated
            x1, y1, x2, y2 = bubble['box']
            mask = bubble_mask(image.size, bubble)
            preferred = font_size if font_size is not None else max(12, min(64, ocr['line_height'] * 1.25))
            padding = max(2, min(8, round(min(x2-x1, y2-y1) * .045)))
            layout = layout_text(translated, bubble['box'], mask, image.size, FONT_PATH,
                                 preferred_size=preferred, padding=padding)
            if layout is None:
                bubble['status'] = 'layout_does_not_fit'
                continue
            if ' '.join(layout.lines) != ' '.join(translated.split()):
                raise ValueError('Rendered text differs from the complete translation')
            cleaned, cleaning = clean_bubble(edited, bubble)
            if not cleaning['removed_pixels']:
                bubble['status'] = 'no_text_mask'
                continue
            edited = draw_layout(cleaned, layout)
            bubble.update(status='translated', rendered_lines=layout.lines,
                          font_size=layout.font_size, cleaning=cleaning)
        except Exception as exc:
            bubble.update(status='error', error=str(exc))
            print(f'  Bubble {index+1}: {exc}', flush=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edited.save(output_path, format='PNG')
    report['output'] = str(output_path)
    report['summary'] = {status: sum(b.get('status') == status for b in report['bubbles'])
                         for status in sorted({b['status'] for b in report['bubbles']})}
    write_json(output_path.with_suffix('.json'), report)
    print(f"  Saved {output_path.name}: {report['summary']}", flush=True)
    return report


def process_archive(archive_path, output_dir, pipeline, font_size=None, max_pages=None):
    import patoolib
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    with tempfile.TemporaryDirectory(prefix='comic_translation_') as temporary:
        extraction = Path(temporary)
        patoolib.extract_archive(str(archive_path.resolve()), outdir=str(extraction), verbosity=-1)
        pages = sorted((p for p in extraction.rglob('*') if p.is_file()
                        and p.suffix.lower() in IMAGE_EXTENSIONS), key=natural_key)
        if not pages:
            raise ValueError(f'No supported images in {archive_path.name}')
        selected = pages[:max_pages] if max_pages else pages
        stage = output_dir / archive_path.stem
        packed, reports = [], []
        for index, page in enumerate(selected, 1):
            relative = page.relative_to(extraction)
            rendered_name = relative.with_name(f'translated_{relative.name}.png')
            destination = stage / rendered_name
            print(f'[{index}/{len(selected)}] {archive_path.name}: {relative}', flush=True)
            report = process_page(page, destination, pipeline, font_size)
            reports.append(report)
            packed.append((destination, rendered_name.as_posix()))
        suffix = '_sample' if max_pages and len(selected) < len(pages) else ''
        output_cbz = output_dir / f'{archive_path.stem}{suffix}_[{pipeline.target_lang}].cbz'
        temporary_cbz = output_cbz.with_suffix('.cbz.tmp')
        with zipfile.ZipFile(temporary_cbz, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path, name in packed:
                archive.write(path, name)
            if len(selected) == len(pages):
                for extra in extraction.rglob('*'):
                    if extra.is_file() and extra.suffix.lower() not in IMAGE_EXTENSIONS:
                        archive.write(extra, extra.relative_to(extraction).as_posix())
        with zipfile.ZipFile(temporary_cbz) as archive:
            if archive.testzip() is not None:
                raise IOError('Output archive failed its integrity check')
        temporary_cbz.replace(output_cbz)
        write_json(output_cbz.with_suffix('.report.json'), dict(
            source=str(archive_path.resolve()), pages=len(selected), source_pages=len(pages),
            output=str(output_cbz), summaries=[dict(source=r['source'], **r['summary']) for r in reports]))
        print(f'Created {output_cbz}', flush=True)
        return output_cbz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='Image, CBZ/CBR, or folder')
    parser.add_argument('--recursive', '-r', action='store_true')
    parser.add_argument('--font-size', '-f', type=int, help='Maximum font size; shrinks to fit')
    parser.add_argument('--source-lang', default='auto', help='Source language, e.g. es')
    parser.add_argument('--target-lang', default='en')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'output/.cache')
    parser.add_argument('--max-pages', type=int, help='Process the first N pages of each archive')
    args = parser.parse_args()
    if not args.path.exists():
        parser.error(f'Path does not exist: {args.path}')
    if args.font_size is not None and args.font_size < 1:
        parser.error('--font-size must be positive')
    if args.max_pages is not None and args.max_pages < 1:
        parser.error('--max-pages must be positive')
    pipeline = Pipeline(args.cache_dir, args.source_lang, args.target_lang)
    paths = (sorted(args.path.rglob('*') if args.recursive else args.path.iterdir(), key=natural_key)
             if args.path.is_dir() else [args.path])
    processed = 0
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix.lower() in ARCHIVE_EXTENSIONS:
            process_archive(path, args.output_dir, pipeline, args.font_size, args.max_pages)
            processed += 1
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            relative = path.relative_to(args.path) if args.path.is_dir() else Path(path.name)
            destination = args.output_dir / relative.parent / f'translated_{relative.name}.png'
            process_page(path, destination, pipeline, args.font_size)
            processed += 1
    if not processed:
        parser.error('No supported images or comic archives found')


if __name__ == '__main__':
    main()
