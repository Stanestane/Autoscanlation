"""Model-independent, mask-constrained comic lettering."""
from dataclasses import dataclass
from functools import lru_cache
import math
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TextLayout:
    font: object
    lines: list[str]
    positions: list[tuple[int, int]]
    font_size: int


def _font(path, size):
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default(size=size)


def layout_text(text, box, mask, image_size, font_path, preferred_size=32, padding=8,
                min_size=8):
    """Fit whole words to individual safe line spans inside the bubble contour."""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return None
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_size[0], x2), min(image_size[1], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    safe = np.ones((y2-y1, x2-x1), np.uint8) if mask is None else (mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    margin = max(1, int(padding))
    safe = cv2.erode(safe, np.ones((margin*2+1, margin*2+1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.nonzero(safe)
    if not len(xs):
        return None
    top, bottom = int(ys.min()), int(ys.max()) + 1
    cx, cy = float(xs.mean()), float(ys.mean())
    words = text.split()
    maximum = max(1, min(120, round(preferred_size)))
    for size in range(maximum, min(min_size, maximum)-1, -1):
        font = _font(font_path, size)
        bounds = font.getbbox(text)
        ink_h = max(1, bounds[3]-bounds[1])
        step = ink_h + max(2, round(size*.2))
        max_lines = min(len(words), (bottom-top+step-ink_h)//step)
        if max_lines < 1:
            continue

        @lru_cache(None)
        def measure(start, end):
            b = font.getbbox(' '.join(words[start:end]))
            return b[2]-b[0]

        @lru_cache(None)
        def span(y):
            cols = np.all(safe[y:y+ink_h], axis=0)
            edges = np.diff(np.pad(cols.astype(np.int8), (1, 1)))
            runs = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
            return max(runs, key=lambda ab: (ab[1]-ab[0]) - abs((ab[0]+ab[1])/2-cx)*.25) if runs else (0, 0)

        best = None
        for count in range(1, max_lines+1):
            total_h = ink_h+(count-1)*step
            ideal = int(np.clip(round(cy-total_h/2), top, bottom-total_h))
            starts = sorted(set([ideal, *np.linspace(top, bottom-total_h, 7).astype(int)]),
                            key=lambda y: abs(y-ideal))
            for start_y in starts:
                spans = [span(start_y+i*step) for i in range(count)]
                widths = [b-a for a, b in spans]
                if not all(widths) or sum(measure(i, i+1) for i in range(len(words))) > sum(widths):
                    continue

                @lru_cache(None)
                def fit(row, start):
                    if row == count:
                        return (0., ()) if start == len(words) else (math.inf, ())
                    best_row = (math.inf, ())
                    for end in range(start+1, len(words)-(count-row-1)+1):
                        length = measure(start, end)
                        if length > widths[row]:
                            break
                        cost, tail = fit(row+1, end)
                        # Balance occupancy while discouraging isolated final words.
                        cost += ((widths[row]-length)/max(widths))**2
                        if cost < best_row[0]:
                            best_row = cost, ((start, end),)+tail
                    return best_row

                cost, breaks = fit(0, 0)
                if not math.isfinite(cost):
                    continue
                score = cost/count + ((start_y+total_h/2-cy)/max(1, bottom-top))**2*8 + count*.025
                if best is None or score < best[0]:
                    lines, positions = [], []
                    for row, (start, end) in enumerate(breaks):
                        line = ' '.join(words[start:end])
                        b = font.getbbox(line)
                        a, z = spans[row]
                        x = (a+z-(b[2]-b[0]))//2
                        positions.append((x1+int(x)-b[0], y1+start_y+row*step-bounds[1]))
                        lines.append(line)
                    best = score, TextLayout(font, lines, positions, size)
        if best:
            assert ' '.join(best[1].lines) == text
            return best[1]
    return None


def draw_layout(image: Image.Image, layout: TextLayout) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for line, position in zip(layout.lines, layout.positions):
        draw.text(position, line, font=layout.font, fill='black')
    return image
