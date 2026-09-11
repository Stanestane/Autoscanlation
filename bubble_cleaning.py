"""Remove lettering using OCR regions while retaining bubble borders and color."""
import cv2
import numpy as np
from PIL import Image


def caption_candidates(image):
    """Find flat rectangular narration boxes that bubble-only models miss."""
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (w >= 70 and h >= 30 and 1.2 < w / h < 12
                and 1500 < w * h < image.width * image.height * .07):
            continue
        if cv2.contourArea(contour) / (w * h) < .90:
            continue
        approx = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        inside = rgb[y+4:y+h-4, x+4:x+w-4]
        center = np.median(inside.reshape(-1, 3), axis=0)
        if np.mean(center) < 120:
            continue
        uniform = np.max(np.abs(inside.astype(float) - center), axis=2) < 28
        dark = np.mean(inside, axis=2) < np.mean(center) - 55
        if uniform.mean() < .65 or not .015 < dark.mean() < .3:
            continue
        candidates.append(dict(box=[x, y, x+w, y+h], confidence=0.,
                               polygon=[[x, y], [x+w-1, y], [x+w-1, y+h-1], [x, y+h-1]],
                               detection='caption-contour'))
    return sorted(candidates, key=lambda c: -(c['box'][2]-c['box'][0])*(c['box'][3]-c['box'][1]))


def bubble_mask(size, bubble):
    mask = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(mask, [np.round(bubble['polygon']).astype(np.int32)], 255)
    return mask


def clean_bubble(image, bubble):
    """Inpaint complete ink strokes, including anti-aliasing, without white haze.

    All calculations run on the bubble crop. OCR polygons locate lettering;
    segmentation and connected-component checks protect the outline/artwork.
    """
    x1, y1, x2, y2 = bubble['box']
    original = np.array(image.crop((x1, y1, x2, y2)).convert('RGB'))
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    shape = gray.shape
    area = np.zeros(shape, np.uint8)
    polygon = np.round(bubble['polygon']).astype(np.int32) - [x1, y1]
    cv2.fillPoly(area, [polygon], 255)
    safe = cv2.erode(area, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT,
                     borderValue=0)
    zones = np.zeros(shape, np.uint8)
    for region in bubble['ocr']['regions']:
        quad = np.round(region['quad']).astype(np.int32).reshape(4, 2)
        cv2.fillPoly(zones, [quad], 255)
    zones = cv2.dilate(zones, np.ones((5, 5), np.uint8))
    valid = (safe > 0)
    if not np.any(valid) or not np.any(zones):
        return image, {'method': 'none', 'removed_pixels': 0}
    background_level = float(np.percentile(gray[valid], 80))
    # Relative threshold handles cream, yellow, and other colored balloons.
    ink = ((gray < background_level - 30) & valid).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    # Connected lettering can span several lines on low-resolution scans.
    # Always include ink inside OCR polygons; component filtering only grows
    # this selection to catch strokes that extend outside the predicted boxes.
    selected = ((ink > 0) & (zones > 0)).astype(np.uint8) * 255
    line_height = max(6, bubble['ocr']['line_height'])
    for index in range(1, count):
        x, y, w, h, pixels = stats[index]
        part = components[y:y + h, x:x + w] == index
        overlap = np.count_nonzero(part & (zones[y:y + h, x:x + w] > 0))
        # Border/art components are much taller than the lettering or have
        # little overlap with its OCR polygons. Dots and accents are retained.
        if overlap >= pixels * .45 and h <= line_height * 2.5:
            selected[y:y + h, x:x + w][part] = 255
    radius = max(1, min(3, round(line_height * .09)))
    selected = cv2.dilate(selected, np.ones((radius * 2 + 1,) * 2, np.uint8))
    selected = cv2.bitwise_and(selected, safe)
    removed = int(np.count_nonzero(selected))
    if not removed:
        return image, {'method': 'none', 'removed_pixels': 0}
    samples = original[valid & (selected == 0) & (gray >= background_level - 18)]
    color = np.median(samples, axis=0) if len(samples) else np.array([255, 255, 255])
    variation = float(np.mean(np.std(samples, axis=0))) if len(samples) else 0
    if variation < 3:
        cleaned = original.copy()
        cleaned[selected > 0] = np.round(color).astype(np.uint8)
        method = 'background-color'
    else:
        cleaned = cv2.inpaint(original, selected, 3, cv2.INPAINT_TELEA)
        method = 'inpaint'
    result = image.copy()
    result.paste(Image.fromarray(cleaned), (x1, y1))
    return result, {'method': method, 'removed_pixels': removed}
