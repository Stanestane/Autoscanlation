"""Generate a lettering preview without loading OCR or translation models."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from text_layout import layout_text, draw_layout


def main():
    root = Path(__file__).parent
    canvas = Image.new('RGB', (1000, 600), '#e8e6e1')
    labels = ImageDraw.Draw(canvas)
    examples = [
        ('Balanced dialogue', 'Wait! Where do you think you are going?', 36),
        ('Long translation', 'I thought we had agreed to wait until morning. '
         'Now we will have to find our own way through the forest!', 40),
        ('Irregular bubble', 'There must be another way out of here!', 34),
        ('Long word', 'UNBELIEVABLE! SUPERCALIFRAGILISTICEXPIALIDOCIOUS!', 36),
    ]
    for index, (title, text, size) in enumerate(examples):
        mask = np.zeros((250, 460), np.uint8)
        cv2.ellipse(mask, (230, 125), (215, 110), 0, 0, 360, 255, -1)
        if index == 2:
            cv2.circle(mask, (430, 110), 65, 0, -1)
        page = Image.new('RGB', (460, 250), '#e8e6e1')
        page.paste('black', mask=Image.fromarray(mask))
        inside = cv2.erode(mask, np.ones((5, 5), np.uint8))
        page.paste('white', mask=Image.fromarray(inside))
        layout = layout_text(text, (0, 0, 460, 250), mask, page.size,
                             root / 'fonts/animeace2_reg.otf', size, padding=14)
        if layout:
            draw_layout(page, layout)
        x, y = 20 + index % 2 * 500, 35 + index // 2 * 300
        canvas.paste(page, (x, y))
        labels.text((x + 10, y - 20), title, fill='black')
    output = root / 'output/text_layout_preview.png'
    output.parent.mkdir(exist_ok=True)
    canvas.save(output)
    print(output)


if __name__ == '__main__':
    main()
