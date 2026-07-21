import io
import os
import numpy as np


from surya.settings import settings

from bs4 import BeautifulSoup


import pypdfium2
from PIL import Image
from fastapi import UploadFile
from concurrent.futures import ProcessPoolExecutor


def get_bg_color(image: Image.Image) -> int|tuple[int, int, int]:
    """Returns the most common color in the image."""
    pixels = image.getcolors(image.size[0] * image.size[1])
    bg_color = max(pixels, key=lambda x: x[0])[1] if pixels else (255, 255, 255)
    return bg_color

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


def bbox_expand(bbox: tuple[int, ...], margin: int) -> tuple[int, int, int, int]:
    """
    Expands a bounding box by a certain margin.
                bbox = [x_min, y_min, x_max, y_max]
    """
    if len(bbox) != 4:
        return bbox
    return (bbox[0] - margin, bbox[1], bbox[2] + margin + margin, bbox[3] + margin)


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


def crop_by_side_percent(
    image: Image.Image,
    left: float = 0.0,
    right: float = 0.0,
    top: float = 0.0,
    bottom: float = 0.0,
) -> Image.Image:
    """Crops a percentage from each side of the image independently.

    Each parameter is a percentage (0..50) of the image dimension
    to crop from the corresponding side. 0 (or None) means no cropping
    on that side. When the sum of horizontal/vertical crops would
    cover the full image, the original image is returned.
    """
    left = max(0.0, float(left or 0.0))
    right = max(0.0, float(right or 0.0))
    top = max(0.0, float(top or 0.0))
    bottom = max(0.0, float(bottom or 0.0))
    if left == 0.0 and right == 0.0 and top == 0.0 and bottom == 0.0:
        return image
    w, h = image.size
    crop_left = int(w * left / 100)
    crop_right = w - int(w * right / 100)
    crop_top = int(h * top / 100)
    crop_bottom = h - int(h * bottom / 100)
    if crop_left >= crop_right or crop_top >= crop_bottom:
        return image
    return image.crop((crop_left, crop_top, crop_right, crop_bottom))

def entropy(values):
    values = np.asarray(values)
    _, counts = np.unique(values, return_counts=True)
    probs = counts / len(values)
    return -np.sum(probs * np.log2(probs))


def _entropy_from_chunks(image_bytes: bytes, chunk_boxes: list[tuple[int, int, int, int]]) -> list[float]:
    """
    Top-level helper for use with ProcessPoolExecutor (must be picklable).

    Reconstructs the image from a PNG bytes buffer once and computes the entropy
    of each chunk defined by ``chunk_boxes`` (left, top, right, bottom).

    Returns a list of entropy values, one per chunk box, in the same order.
    """
    img = Image.open(io.BytesIO(image_bytes))
    return [entropy(img.crop(box).getdata()) for box in chunk_boxes]


def _split_round_robin(items, n_slices):
    """Split ``items`` into ``n_slices`` lists using round-robin assignment."""
    n_slices = max(1, n_slices)
    slices: list[list] = [[] for _ in range(n_slices)]
    for idx, item in enumerate(items):
        slices[idx % n_slices].append(item)
    return slices


def stripe_entropy(
    image: Image.Image,
    aggressive: bool = False,
    executor: ProcessPoolExecutor | None = None,
) -> float:

    if aggressive:
        return entropy(image.getdata())
    w, h = image.size
    # Compute chunk boundary boxes (left, top, right, bottom) for each stripe.
    chunk_boxes: list[tuple[int, int, int, int]] = []
    if h > w:
        h_step = w
        for y in range(0, h, h_step):
            chunk_boxes.append((0, y, w, min(y + h_step, h)))
    else:
        w_step = h
        for x in range(0, w, w_step):
            chunk_boxes.append((x, 0, min(x + w_step, w), h))
    if not chunk_boxes:
        return 1.0
    # Serialize the entire image to a PNG bytes buffer once so workers can
    # reconstruct it independently without re-pickling per-stripe pixel data.
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    # Split the boxes across workers; each worker decodes ``image_bytes`` once
    # and computes entropy for its assigned subset of chunk boxes.
    n_slices = min(len(chunk_boxes), (os.cpu_count() or 1))
    splits = _split_round_robin(chunk_boxes, n_slices)
    # Use the caller-provided pool if given; otherwise spin up a transient one
    # so the function remains safe to call without external setup.
    owns_pool = executor is None
    pool = executor if executor is not None else ProcessPoolExecutor()
    try:
        per_worker_results = list(pool.map(
            _entropy_from_chunks,
            [image_bytes] * len(splits),
            splits,
        ))
    finally:
        if owns_pool:
            pool.shutdown()
    entropies: list[float] = []
    for result in per_worker_results:
        entropies.extend(result)
    return max(entropies)

def trim_empty_background(
    image: Image.Image,
    threshold: float = 0.5,
    executor: ProcessPoolExecutor | None = None,
) -> Image.Image:

    w, h = image.size
    h_step = max(1, h // 40)
    w_step = max(1, w // 40)
    crop_top = 0
    crop_left = 0
    crop_bottom = h
    crop_right = w
    pal_image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    # Share a single ProcessPoolExecutor across all four stripe scans so the
    # worker pool is created/used at most once per call (or reused entirely
    # if the caller passed one in).
    owns_pool = executor is None
    pool = executor if executor is not None else ProcessPoolExecutor()
    try:
        for y in range(0, h, h_step):
            stripe = pal_image.crop((0, y, w, min(y + h_step, h)))
            stripe_ent = stripe_entropy(stripe, executor=pool)
            if stripe_ent < threshold:
                crop_top = y + h_step
            else:
                break
        for y in range(h, 0, -h_step):
            stripe = pal_image.crop((0, max(y - h_step, 0), w, y))
            stripe_ent = stripe_entropy(stripe, executor=pool)
            if stripe_ent < threshold:
                crop_bottom = y - h_step
            else:
                break
        for x in range(0, w, w_step):
            stripe = pal_image.crop((x, 0, min(x + w_step, w), h))
            stripe_ent = stripe_entropy(stripe, executor=pool)
            if stripe_ent < threshold:
                crop_left = x + w_step
            else:
                break
        for x in range(w, 0, -w_step):
            stripe = pal_image.crop((max(x - w_step, 0), 0, x, h))
            stripe_ent = stripe_entropy(stripe, executor=pool)
            if stripe_ent < threshold:
                crop_right = x - w_step
            else:
                break
    finally:
        if owns_pool:
            pool.shutdown()

    if crop_left >= crop_right or crop_top >= crop_bottom:
        return image
    if crop_left > 0 or crop_top > 0 or crop_right < w or crop_bottom < h:
        image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    return image


# ---------------------------------------------------------------------------
# OCR block merging
# ---------------------------------------------------------------------------

def _block_bbox(block: dict) -> tuple[float, float, float, float] | None:
    """Return (x_min, y_min, x_max, y_max) for a block, or None if unavailable.

    Accepts either a ``polygon`` (list of [x, y] points) or a flat ``bbox``
    ([x_min, y_min, x_max, y_max]).
    """
    poly = block.get("polygon")
    if poly:
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        return (min(xs), min(ys), max(xs), max(ys))
    bbox = block.get("bbox")
    if bbox and len(bbox) == 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return None


def fix_html_chunk(html: str) -> str:
    """Repair missing closing tags in an OCR html fragment using lxml.

    BeautifulSoup with the lxml parser auto-closes unclosed tags and drops
    stray closing tags, producing a well-formed fragment. The surrounding
    ``<body>`` wrapper lxml adds is stripped so only the inner content is
    returned.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    body = soup.body
    if body is not None:
        return body.decode_contents()
    return str(soup)


def _block_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the (cx, cy) center of a bounding box."""
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def _pairwise_distances(centers: list[tuple[float, float]]) -> "np.ndarray":
    """Full pairwise Euclidean distance matrix via ``cv2.batchDistance``.

    Returns an ``(N, N)`` float32 matrix. Falls back to a numpy broadcast
    when cv2 is unavailable or the input is degenerate.
    """
    import numpy as np

    coords = np.asarray(centers, dtype=np.float32)
    n = coords.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=np.float32)
    try:
        import cv2

        dists, _idx = cv2.batchDistance(
            coords, coords, -1, None, normType=cv2.NORM_L2, K=n
        )
        return dists
    except Exception:
        # Fallback: pure numpy pairwise L2 distance.
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)


def _cluster_by_distance(
    centers: list[tuple[float, float]],
    *,
    distance_threshold: float,
) -> list[int]:
    """Cluster block centers by Euclidean distance using single linkage.

    Single-linkage clustering is implemented with a union-find over the
    pairwise distance matrix (computed via :func:`_pairwise_distances`): any
    two centers within ``distance_threshold`` of each other are merged into
    the same cluster. Returns a list of 0-based cluster labels (one per
    center).
    """
    import numpy as np

    n = len(centers)
    if n <= 1:
        return [0] * n
    dists = _pairwise_distances(centers)
    if dists.max() == 0.0:
        return [0] * n

    # Union-find (single linkage): merge any pair within the threshold.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if float(dists[i, j]) <= distance_threshold:
                union(i, j)

    # Normalize to contiguous 0-based labels.
    remap: dict[int, int] = {}
    labels: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in remap:
            remap[root] = len(remap)
        labels.append(remap[root])
    return labels


def merge_html_blocks(
    blocks: list[dict],
    *,
    distance_threshold: float | None = None,
) -> str:
    """Merge OCR html chunks into a single document respecting block adjacency.

    Each block is expected to carry either a ``polygon`` (list of [x, y]
    points) or a ``bbox`` ([x_min, y_min, x_max, y_max]) plus an ``html``
    string. ``reading_order`` is deliberately ignored — it is unreliable on
    multi-column / technical drawings.

    The blocks are ordered into a reading sequence as follows:

    1. The center of each block is computed from its geometry.
    2. Centers are clustered by Euclidean distance (single-linkage
       hierarchical clustering) so spatially adjacent blocks form a cluster.
       The distance threshold defaults to the median of the nearest-neighbour
       distances between centers, which adapts to the document's density.
    3. Clusters are ordered top-to-bottom by their median center y, then
       left-to-right by their median center x.
    4. Within a cluster, blocks are ordered top-to-bottom, then left-to-right
       by their centers.

    Each chunk is first passed through :func:`fix_html_chunk` so missing
    closing tags are repaired before concatenation. Blocks are joined with
    a newline separator.
    """
    if not blocks:
        return ""

    # Resolve geometry + center for every block.
    items: list[tuple[float, float, tuple[float, float, float, float], int, dict]] = []
    for idx, block in enumerate(blocks):
        bbox = _block_bbox(block)
        if bbox is None:
            # No geometry: defer to the end with a synthetic far center.
            items.append((float("inf"), float("inf"), (float("inf"), float("inf"), float("inf"), float("inf")), idx, block))
            continue
        cx, cy = _block_center(bbox)
        items.append((cx, cy, bbox, idx, block))

    centers = [(it[0], it[1]) for it in items]
    finite_centers = [c for c in centers if c[0] != float("inf")]

    if distance_threshold is None:
        # Adaptive threshold: median nearest-neighbour distance between
        # centers. This scales with the document's block density so tightly
        # packed columns cluster together while distant regions separate.
        import numpy as np
        import cv2

        if len(finite_centers) >= 2:
            coords = np.asarray(finite_centers, dtype=np.float32)
            # cv2.batchDistance with K=2 returns, for each point, the two
            # nearest neighbours (self at distance 0 + the true nearest).
            # d[:, 1] is therefore the nearest non-self distance per row.
            dists, _idx = cv2.batchDistance(
                coords, coords, -1, None, normType=cv2.NORM_L2, K=2
            )
            nearest = dists[:, 1]
            distance_threshold = float(np.median(nearest)) * 1.5
        else:
            distance_threshold = 1.0

    labels = _cluster_by_distance(centers, distance_threshold=distance_threshold)

    # Attach cluster label to each item.
    clustered = list(zip(labels, items))

    # Order clusters by their median center (y, then x).
    def _cluster_key(label: int) -> tuple[float, float]:
        members = [it for lbl, it in clustered if lbl == label]
        ys = [it[1] for it in members if it[1] != float("inf")]
        xs = [it[0] for it in members if it[0] != float("inf")]
        med_y = sorted(ys)[len(ys) // 2] if ys else float("inf")
        med_x = sorted(xs)[len(xs) // 2] if xs else float("inf")
        return (med_y, med_x)

    unique_labels = sorted(set(labels), key=_cluster_key)

    parts: list[str] = []
    for label in unique_labels:
        members = [it for lbl, it in clustered if lbl == label]
        # Within a cluster: top-to-bottom, then left-to-right by center.
        members.sort(key=lambda it: (it[1], it[0]))
        for _cx, _cy, _bbox, _idx, block in members:
            html = block.get("html") or ""
            parts.append(fix_html_chunk(html))
    return "\n".join(parts)
