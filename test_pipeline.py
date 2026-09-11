import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bubble_cleaning import clean_bubble
from comic_pipeline import normalize_ocr, normalize_regions
from translate_comic import process_archive, process_page


class PipelineTests(unittest.TestCase):
    def test_dehyphenates_only_line_endings(self):
        self.assertEqual(normalize_ocr(['He has a well-known com-', 'pany.']),
                         'He has a well-known company.')

    def test_slanted_lines_keep_order(self):
        ocr = dict(regions=[
            dict(text='A VER SI', quad=[5, 15, 292, 2, 293, 18, 5, 35]),
            dict(text='COM-', quad=[5, 34, 298, 16, 299, 30, 5, 49]),
            dict(text='PROMETIDO', quad=[5, 49, 197, 36, 198, 50, 5, 64]),
        ])
        self.assertEqual(normalize_regions(ocr)['text'], 'A VER SI COMPROMETIDO')

    def fixture(self, color=(240, 220, 160)):
        image = Image.new('RGB', (200, 100), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((1, 1, 198, 98), outline='black', width=2)
        font = ImageFont.truetype(str(Path(__file__).parent/'fonts/animeace2_reg.otf'), 22)
        draw.text((30, 25), 'OLD TEXT', font=font, fill='black')
        bounds = draw.textbbox((30, 25), 'OLD TEXT', font=font)
        a, b, c, d = bounds
        bubble = dict(box=[0, 0, 200, 100], polygon=[[0, 0], [199, 0], [199, 99], [0, 99]],
                      ocr=dict(text='OLD TEXT', complete=True, line_height=d-b,
                               regions=[dict(text='OLD TEXT', quad=[a,b,c,b,c,d,a,d])]))
        return image, bubble, bounds

    def test_colored_background_cleanup_removes_all_ink(self):
        image, bubble, (a,b,c,d) = self.fixture()
        cleaned, info = clean_bubble(image, bubble)
        array, before = np.array(cleaned), np.array(image)
        self.assertGreater(info['removed_pixels'], 0)
        self.assertTrue(np.all(array[b:d, a:c] == [240, 220, 160]))
        self.assertTrue(np.array_equal(array[:10], before[:10]), 'Bubble outline changed')
        self.assertTrue(np.array_equal(array[65:], before[65:]), 'Background changed outside text')

    def test_failed_translation_preserves_original(self):
        image, bubble, _ = self.fixture()
        class FakePipeline:
            source_lang = 'es'
            def analyze(self, path):
                return image, dict(bubbles=[bubble])
            def translate(self, text):
                raise RuntimeError('Offline')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.png'
            report = process_page('unused', path, FakePipeline())
            self.assertEqual(report['bubbles'][0]['status'], 'error')
            self.assertTrue(np.array_equal(np.array(image), np.array(Image.open(path))))

    def test_archive_preserves_duplicate_basenames_and_metadata(self):
        class FakePipeline:
            target_lang, source_lang = 'en', 'es'
            def analyze(self, path):
                return Image.open(path).convert('RGB'), dict(source=Path(path).name, bubbles=[])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root/'page.png'
            Image.new('RGB', (20, 20), 'white').save(page)
            source = root/'source.cbz'
            with zipfile.ZipFile(source, 'w') as archive:
                archive.write(page, 'chapter1/page.png')
                archive.write(page, 'chapter2/page.png')
                archive.writestr('ComicInfo.xml', '<ComicInfo/>')
            def extract(path, outdir, verbosity):
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(outdir)
            with patch('patoolib.extract_archive', side_effect=extract):
                result = process_archive(source, root/'out', FakePipeline())
            with zipfile.ZipFile(result) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()), {
                    'chapter1/translated_page.png.png', 'chapter2/translated_page.png.png', 'ComicInfo.xml'})


if __name__ == '__main__':
    unittest.main()
