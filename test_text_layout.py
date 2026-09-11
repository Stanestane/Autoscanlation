import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from text_layout import layout_text, draw_layout


FONT = Path(__file__).parent / 'fonts' / 'animeace2_reg.otf'


class LetteringTests(unittest.TestCase):
    def render(self, text, mask, size=32):
        layout = layout_text(text, (0, 0, 240, 180), mask, (240, 180), FONT, size)
        self.assertIsNotNone(layout)
        image = draw_layout(Image.new('RGB', (240, 180), 'white'), layout)
        ink = np.any(np.array(image) < 255, axis=2)
        safe = cv2.erode(mask, np.ones((17, 17), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
        self.assertTrue(np.any(ink))
        self.assertFalse(np.any(ink & (safe == 0)), 'Lettering escaped the padded mask')
        self.assertEqual(' '.join(layout.lines), ' '.join(text.split()))
        return layout, image

    def test_ellipse_and_descenders(self):
        mask = np.zeros((180, 240), np.uint8)
        cv2.ellipse(mask, (120, 90), (115, 85), 0, 0, 360, 255, -1)
        self.render('Why are you jumping? Please, stay here!', mask)

    def test_concave_bubble(self):
        mask = np.full((180, 240), 255, np.uint8)
        mask[65:115, 150:] = 0
        self.render('We must stay inside this strange bubble.', mask)

    def test_long_text_shrinks(self):
        mask = np.full((180, 240), 255, np.uint8)
        layout, _ = self.render('This translation is much longer than the original. '
                                'Every word must remain visible inside the bubble.', mask, 48)
        self.assertLess(layout.font_size, 48)

    def test_long_word(self):
        self.render('SUPERCALIFRAGILISTICEXPIALIDOCIOUS', np.full((180, 240), 255, np.uint8))

    def test_unusable_mask(self):
        self.assertIsNone(layout_text('Hello', (0, 0, 20, 20),
                                      np.zeros((20, 20), np.uint8), (20, 20), FONT))

    def test_impossible_text(self):
        self.assertIsNone(layout_text('Too much text for a tiny bubble', (0, 0, 20, 20),
                                      None, (20, 20), FONT))

    def test_missing_font_and_clipped_box(self):
        layout = layout_text('Hello there!', (-10, -10, 250, 190), None,
                             (240, 180), 'missing.otf', 24)
        self.assertIsNotNone(layout)


if __name__ == '__main__':
    unittest.main()
