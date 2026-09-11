"""Reproducible samples from three original volumes; never uses translated inputs."""
import argparse
import subprocess
import zipfile
from pathlib import Path

from comic_pipeline import Pipeline, write_json

ROOT = Path(__file__).parent
WORK = ROOT / 'output/quality_samples_v2'
SELECTION = [
    ('13 Rue Tomo 1.cbr', ['002-TV001-610313.jpg', '004-TV003-610327.jpg']),
    ('El Buscon en las Indias.cbz', ['IMG_0009.jpg', 'IMG_0010.jpg']),
    ('Pedro Perez - Trizia 01.cbr', ['Trizia-01_06.jpg', 'Trizia-01_07.jpg']),
]


def prepare():
    archives = []
    for source_name, pages in SELECTION:
        source = ROOT / 'input' / source_name
        folder = WORK / 'originals' / source.stem
        folder.mkdir(parents=True, exist_ok=True)
        missing = [name for name in pages if not (folder / name).exists()]
        if missing:
            subprocess.run(['7z', 'e', str(source), f'-o{folder}', '-y', '-r',
                            *missing], check=True, capture_output=True)
        sample = WORK / 'input' / f'{source.stem}_sample.cbz'
        sample.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(sample, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in pages:
                archive.write(folder / name, name)
        archives.append(sample)
    write_json(WORK / 'selection.json', [dict(archive=a, pages=p) for a, p in SELECTION])
    return archives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    samples = prepare()
    pipeline = Pipeline(WORK / 'cache', source_lang='es')
    if args.analyze_only:
        for source_name, pages in SELECTION:
            for page in pages:
                print(f'Analyzing {source_name}: {page}', flush=True)
                pipeline.analyze(WORK / 'originals' / Path(source_name).stem / page)
    else:
        from translate_comic import process_archive
        for sample in samples:
            process_archive(sample, WORK / 'translated', pipeline)


if __name__ == '__main__':
    main()
