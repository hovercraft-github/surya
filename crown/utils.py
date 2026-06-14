from math import sqrt
import numpy as np


from surya.settings import settings


import pypdfium2
from PIL import Image
from fastapi import UploadFile


def get_page_image(
    pdf_file: UploadFile, page_num: int, dpi: int | None = None
) -> Image.Image:
    if dpi is None:
        dpi = settings.IMAGE_DPI_HIGHRES
    doc = pypdfium2.PdfDocument(pdf_file.file.read())
    renderred = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderred)[0]
    png_image = png.convert("RGB")
    doc.close()
    return png_image


def poligon_expand(polygon: list[list[float]], margin: float):
    """
    Expands a polygon by a certain margin (inplace).
                polygon = [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max],
                ]
    """
    if len(polygon) != 4:
        return polygon
    polygon[0][0] -= margin  # x_min
    polygon[0][1] -= margin  # y_min
    polygon[1][0] += margin  # x_max
    polygon[1][1] -= margin  # y_min
    polygon[2][0] += margin  # x_max
    polygon[2][1] += margin  # y_max
    polygon[3][0] -= margin  # x_min
    polygon[3][1] += margin  # y_max


def image_white_cnt_points(image: Image.Image, white_threshold: int = 200) -> int:
    """Counts the number of white points in b/w image."""
    import numpy as np

    array_img = np.array(image)
    count = int(np.sum(array_img > white_threshold))
    return count

def crop_by_percent(image: Image.Image, crop_percent: float) -> Image.Image:
    """Crops a percentage from each side of the image."""
    if crop_percent > 0.0:
        w, h = image.size
        crop_margin_h = int(h * crop_percent / 100)
        crop_margin_w = int(w * crop_percent / 100)
        cropped_image = image.crop((
            crop_margin_w,
            crop_margin_h,
            w - crop_margin_w,
            h - crop_margin_h
        ))
        return cropped_image
    return image

def entropy(values):
    values = np.asarray(values)
    _, counts = np.unique(values, return_counts=True)
    probs = counts / len(values)
    return -np.sum(probs * np.log2(probs))


def trim_white_background(image: Image.Image, threshold: int | None = None) -> Image.Image:

    gray = image.convert("L")

    if threshold is None:
        colors = [(int(x[0]), int(x[1])) for x in gray.getcolors(maxcolors=256) or []]
        if not colors or len(colors) < 2:
            return image
        sorted_colors = sorted(colors, key=lambda x: x[1], reverse=True)
        values = [c[1] for c in sorted_colors]
        weights = [c[0] for c in sorted_colors]
        threshold = int(np.average(values, weights=weights))
        light = sorted_colors[0][1]
        dark = sorted_colors[-1][1]
        if threshold == light or threshold == dark:
            return image

    inverse = gray.point(lambda x: 255 if x <= threshold else 0)
    # bg = Image.new(image.mode, image.size, bg_color)
    # diff = ImageChops.difference(image, bg)
    bbox = inverse.getbbox()
    if bbox:
        image = image.crop(bbox)
        gray = image.convert("L")
        inverse = gray.point(lambda x: 255 if x <= threshold else 0)
    n_points_total = image_white_cnt_points(inverse, white_threshold=threshold)
    if n_points_total == 0:
        return image
    w,h = image.size
    h_step = max(1, h // 10)
    w_step = max(1, w // 10)
    crop_top = 0
    crop_left = 0
    crop_bottom = h
    crop_right = w
    n_removed = 0
    for y in range(0, h, h_step):
        chank = inverse.crop((0, y, w, min(y + h_step, h)))
        chank_points = image_white_cnt_points(chank, white_threshold=threshold)
        n_removed += chank_points
        if n_removed / n_points_total < 0.01:
            crop_top = y + h_step
        else:
            break
    n_removed = 0
    for y in range(h, 0, -h_step):
        chank = inverse.crop((0, max(y - h_step, 0), w, y))
        chank_points = image_white_cnt_points(chank, white_threshold=threshold)
        n_removed += chank_points
        if n_removed / n_points_total < 0.01:
            crop_bottom = y - h_step
        else:
            break
    n_removed = 0
    for x in range(0, w, w_step):
        chank = inverse.crop((x, 0, min(x + w_step, w), h))
        chank_points = image_white_cnt_points(chank, white_threshold=threshold)
        n_removed += chank_points
        if n_removed / n_points_total < 0.01:
            crop_left = x + w_step
        else:
            break
    n_removed = 0
    for x in range(w, 0, -w_step):
        chank = inverse.crop((max(x - w_step, 0), 0, x, h))
        chank_points = image_white_cnt_points(chank, white_threshold=threshold)
        n_removed += chank_points
        if n_removed / n_points_total < 0.01:
            crop_right = x - w_step
        else:
            break
    if crop_left >= crop_right or crop_top >= crop_bottom:
        return image
    if crop_left > 0 or crop_top > 0 or crop_right < w or crop_bottom < h:
        image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    return image


def color_distance(pixel, bg, color_space='sRGB'):

    '''Calculates the color distance between two pixels according to a specified color space.  
    Returned values are normalised to be between 0 and 1.  
    Only sRGB is currently supported, but other color spaces could be added in the future.
    '''

    if color_space.lower() == 'srgb':
        norm = sqrt(3* 255**2)
        cdist = sqrt((pixel[0] - bg[0])**2 + (pixel[1] - bg[1])**2 + (pixel[2] - bg[2])**2)

        return cdist/norm
    return 0.0

def stripe_entropy(image: Image.Image, aggressive: bool = False) -> float:

    if aggressive:
        return entropy(image.getdata())
    w,h = image.size
    entropies = []
    if h > w:
        h_step = w
        for y in range(0, h, h_step):
            chank = image.crop((0, y, w, min(y + h_step, h)))
            entropies.append(entropy(chank.getdata()))
    else:
        w_step = h
        for x in range(0, w, w_step):
            chank = image.crop((x, 0, min(x + w_step, w), h))
            entropies.append(entropy(chank.getdata()))
    return max(entropies)

def trim_empty_background(image: Image.Image, threshold: float = 0.5) -> Image.Image:

    w,h = image.size
    h_step = max(1, h // 40)
    w_step = max(1, w // 40)
    crop_top = 0
    crop_left = 0
    crop_bottom = h
    crop_right = w
    pal_image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    for y in range(0, h, h_step):
        stripe = pal_image.crop((0, y, w, min(y + h_step, h)))
        stripe_ent = stripe_entropy(stripe)
        if stripe_ent < threshold:
            crop_top = y + h_step
        else:
            break
    for y in range(h, 0, -h_step):
        stripe = pal_image.crop((0, max(y - h_step, 0), w, y))
        stripe_ent = stripe_entropy(stripe)
        if stripe_ent < threshold:
            crop_bottom = y - h_step
        else:
            break
    for x in range(0, w, w_step):
        stripe = pal_image.crop((x, 0, min(x + w_step, w), h))
        stripe_ent = stripe_entropy(stripe)
        if stripe_ent < threshold:
            crop_left = x + w_step
        else:
            break
    for x in range(w, 0, -w_step):
        stripe = pal_image.crop((max(x - w_step, 0), 0, x, h))
        stripe_ent = stripe_entropy(stripe)
        if stripe_ent < threshold:
            crop_right = x - w_step
        else:
            break

    if crop_left >= crop_right or crop_top >= crop_bottom:
        return image
    if crop_left > 0 or crop_top > 0 or crop_right < w or crop_bottom < h:
        image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    return image
