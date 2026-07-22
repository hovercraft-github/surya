import base64
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from io import BytesIO
import requests
import json

from torch.cuda import temperature


from surya.table_rec import TableRecPredictor, _polygon_from_bbox, _intersect_bbox, logger
from surya.table_rec.schema import TableCell, TableCol, TableResult, TableRow
from surya.inference.prompts import PROMPT_TYPE_TABLE_REC, TABLE_REC_JSON_SCHEMA, TABLE_REC_LABEL_SET
from surya.inference.schema import PROMPT_TYPE_BLOCK, BatchInputItem, BatchOutputItem
from surya.inference.util import image_token_budget
from surya.inference import SuryaInferenceManager, get_default_manager
from surya.logging import get_logger
from surya.settings import settings
from surya.inference.parsers import clean_block_html, denorm_bbox, parse_table_rec

from crown.settings import crown_settings


BLOCK_PROMPT_TBL = ("OCR this image to HTML Each block is a div with data-label and data-bbox "
    "(x0 y0 x1 y1, normalized 0-1000)."
)

# TABLE_REC_PROMPT = (
#     "OCR this image to JSON. Each entry is a dict with "
#     '"label" ("Row" or "Col"), "text" (raw OCR text), and "bbox" (x0 y0 x1 y1, normalized 0-1000).'
# )

TABLE_REC_PROMPT = (
    "OCR this image to JSON. Each entry is a dict with "
    '"text" (raw OCR text), and "bbox" (x0 y0 x1 y1, normalized 0-1000).'
)

TABLE_REC_JSON_SCHEMA_EXT = {
    "type": "array",
    "maxItems": 200,
    "items": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": TABLE_REC_LABEL_SET},
            "text": {"type": "string", "description": "OCR text for this block."},
            "bbox": {
                "type": "string",
                "pattern": r"^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$",
            },
        },
        "required": ["bbox", "text"],
        "additionalProperties": False,
    },
}

GLMOCR_PROMPT = (
    "Extract all text from this image. "
    "Format any tables as HTML tables (<table><thead>...). "
)
    # "Ensure all keys in the extracted JSON contain unique values. "
    # "Deduplicate any overlapping line items in the source image."
    # "Preserve all numbers, units, and special characters exactly. "
GLMOCR_PROMPT_LAYOUT = (
    "Detect the bounding boxes of all individual table cells in this image. "
    "Please pinpoint the bounding box [[x1,y1,x2,y2], ...] in the image for every cell, matching their text content."
    #     "Extract all text from this image. "
    #     "Format any tables as HTML tables (<table><thead>...). "
    #     "Preserve the original layout of the text and tables as much as possible. "
)


def ollama_glm(image: Image.Image, prompt: str | None = None, num_predict: int | None = None) -> dict:
    """Call glm-ocr via Ollama chat API and return the full response JSON."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    OLLAMA_URL = crown_settings.OLLAMA_URL
    if not OLLAMA_URL:
        OLLAMA_URL = crown_settings.OLLAMA_URL_LAYOUT
    if not OLLAMA_URL:
        raise ValueError("OLLAMA_URL or OLLAMA_URL_LAYOUT must be set in your environment.")
    if not crown_settings.OLLAMA_GLM_MODEL:
        raise ValueError("OLLAMA_GLM_MODEL must be set in your environment.")
    if not prompt:
        prompt = GLMOCR_PROMPT

    payload = {
        "model": crown_settings.OLLAMA_GLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.0,
                    # "stop": ["\n", "\n\n", " \n \n \n", "---"],
                    "stop": ["\n\n", " \n \n \n", "---", "\n```\n```", "``````"],
                    "num_predict": 2048,
                    # "repeat_penalty": 1.4
                    },
    }

    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict

    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
    r.raise_for_status()

    result = r.json()
    return result

SURYA_OLLAMA_TBL_HTML_PROMPT = (
    # "OCR this table block image to HTML"
    "OCR this table to HTML"
)

SURYA_OLLAMA_STAMP_HTML_PROMPT = (
    "OCR this table to plain text"
)

SURYA_OLLAMA_STAMP_CORNER_PROMPT = (
    # "OCR this table block image to plain text"
    "OCR this block to HTML"
)

SURYA_OLLAMA_TBL_JSON_PROMPT = ( 
    # "OCR this block image to JSON. Each entry is a dict with "
    # '"text" (raw OCR text), and "bbox" (x0 y0 x1 y1, normalized 0-1000).'
    'Output the table rows then columns as JSON. Each entry is a dict with "label" ("Row" or "Col") '
    'and "bbox" (x0 y0 x1 y1, normalized 0-1000).'
)
# Output the table rows then columns as JSON. Each entry is a dict with "label" ("Row" or "Col") and "bbox" (x0 y0 x1 y1, normalized 0-1000).

def ollama_surya(image: Image.Image, prompt: str | None = None, num_predict: int | None = None, temperature: float | None = None) -> dict:
    """Call glm-ocr via Ollama chat API and return the full response JSON."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    OLLAMA_URL = crown_settings.OLLAMA_URL
    if not OLLAMA_URL:
        OLLAMA_URL = crown_settings.OLLAMA_URL_LAYOUT
    if not OLLAMA_URL:
        raise ValueError("OLLAMA_URL or OLLAMA_URL_LAYOUT must be set in your environment.")
    if not crown_settings.OLLAMA_SURYA_MODEL:
        raise ValueError("OLLAMA_SURYA_MODEL name must be set in your environment.")
    if not prompt:
        prompt = SURYA_OLLAMA_TBL_HTML_PROMPT

    payload = {
        "model": crown_settings.OLLAMA_SURYA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }
        ],
        "stream": False,
        "options": {
                    # "temperature": 0.0,
                    # "stop": ["\n", "\n\n", " \n \n \n", "---"],
                    # "stop": ["\n\n", " \n \n \n", "---", "\n```\n```", "``````"],
                    # "num_predict": 2048,
                    # "repeat_penalty": 1.4
                    },
    }

    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict
    if temperature is not None:
        payload["options"]["temperature"] = temperature

    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
    r.raise_for_status()

    result = r.json()
    return result


async def glm_ocr(image: Image.Image, prompt: str | None = None) -> str:
    """Call glm-ocr via Ollama chat API and return the extracted text content."""
    if not prompt:
        prompt = "Table Recognition:"
    result = ollama_glm(image, prompt=prompt)
    content = str(result.get("message", {}).get("content", ""))
    if "```json" in content or content.startswith("[["):
        content = ""
    if "```table" in content:
        content = content.split("```table", 1)[-1].rsplit("```", 1)[0].strip()
    return content

class TableExtPredictor(TableRecPredictor):
    def __init__(self, manager: Optional[SuryaInferenceManager] = None):
        super().__init__(manager)

    def predict_flexible(
        self, images: List[Image.Image],
        counts: Optional[List[int]] = None,
        mode: str = "div"
    ) -> List[TableResult]:
        """Full-HTML path: BLOCK_PROMPT on table crops. Use when complex
        structure (spanning cells, headers) matters and ground-truth-style
        HTML is preferred. `counts` (one per image) shapes max_tokens."""
        if not images:
            return []
        if crown_settings.OLLAMA_URL:
            results = []
            for img in images:
                result = ollama_glm(img)
                w, h = img.size
                page_bbox = [0, 0, float(w), float(h)]
                results.append(
                    TableResult(
                        rows=[],
                        cols=[],
                        cells=[],
                        image_bbox=page_bbox,
                        raw=None,
                        html=result.get("message", {}).get("content", ""),
                        mode="full",
                        error=False,
                    )
                )
            return results
        if counts is None:
            counts = [0] * len(images)
        batch = []
        temperature: float | None = None
        if mode == "td":
            prompt = None
        elif mode == "div":
            prompt = BLOCK_PROMPT_TBL
        elif mode == "stamp":
            prompt = SURYA_OLLAMA_STAMP_HTML_PROMPT
        elif mode == "corner":
            prompt = SURYA_OLLAMA_STAMP_CORNER_PROMPT
        if prompt:
            prompt_type = ""
        else:
            prompt_type = PROMPT_TYPE_BLOCK
        for img, count in zip(images, counts):
            batch.append(
                BatchInputItem(
                    image=img,
                    prompt=prompt,
                    prompt_type=prompt_type,
                    max_tokens=image_token_budget(
                        count,
                        ceiling=settings.SURYA_MAX_TOKENS_BLOCK_CEILING,
                        floor=1024,
                    ),
                )
            )
        if crown_settings.OLLAMA_URL_LAYOUT and crown_settings.OLLAMA_SURYA_MODEL:
            surya_outputs: list[BatchOutputItem] = []
            if mode == "td":
                prompt = SURYA_OLLAMA_TBL_HTML_PROMPT
                # temperature = 0.5
            for item in batch:
                result = ollama_surya(item.image, prompt=prompt, temperature=temperature)
                content = str(result.get("message", {}).get("content", ""))
                if ("<img/>" in content  or "\"bbox\"" in content): #and mode == "stamp":
                    result = ollama_surya(item.image, prompt="OCR this block image to HTML.", temperature=0.5)
                    content = str(result.get("message", {}).get("content", ""))
                surya_result:BatchOutputItem = BatchOutputItem(
                    raw=content,
                    error=False,
                    metadata=item.metadata,
                    token_count=len(content) if content else 0
                )
                surya_outputs.append(surya_result)
        else:
            manager = self.manager or get_default_manager()
            surya_outputs = manager.generate(batch)
        results: List[TableResult] = []
        for img, out in zip(images, surya_outputs):
            w, h = img.size
            page_bbox = [0, 0, float(w), float(h)]
            if out.error:
                results.append(
                    TableResult(
                        rows=[],
                        cols=[],
                        cells=[],
                        image_bbox=page_bbox,
                        raw=out.raw,
                        mode="full",
                        error=True,
                    )
                )
                continue
            html = clean_block_html(out.raw)
            results.append(
                TableResult(
                    rows=[],
                    cols=[],
                    cells=[],
                    image_bbox=page_bbox,
                    raw=out.raw,
                    html=html,
                    mode="full",
                    error=False,
                )
            )
        return results


    def predict_simple(self, images: List[Image.Image]) -> List[TableResult]:
        if not images:
            return []
        guided = TABLE_REC_JSON_SCHEMA_EXT if settings.SURYA_GUIDED_TABLE_REC else None
        batch = [
            BatchInputItem(
                image=img,
                # prompt=TABLE_REC_PROMPT,
                prompt_type=PROMPT_TYPE_TABLE_REC,
                max_tokens=settings.SURYA_MAX_TOKENS_TABLE_REC,
                guided_json=guided,
                metadata={"image_index": i},
            )
            for i, img in enumerate(images)
        ]
        glm_results: dict[int, str] = {}
        if crown_settings.OLLAMA_URL_LAYOUT and crown_settings.OLLAMA_SURYA_MODEL:
            surya_results: list[BatchOutputItem] = []
            for item in batch:
                result = ollama_glm(item.image, prompt=GLMOCR_PROMPT_LAYOUT)
                content = str(result.get("message", {}).get("content", ""))
                if "```json" in content or content.startswith("[["):
                    content = ""
                if "```table" in content:
                    content = content.split("```table", 1)[-1].rsplit("```", 1)[0].strip()
                # if not is_valid_html(content):
                #     content = ""
                glm_results[item.metadata["image_index"]] = content
                if crown_settings.DEBUG_FOLDER:
                    debug_path = Path(crown_settings.DEBUG_FOLDER) / "glm_ocr_layout.json"
                    with open(debug_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                result = ollama_surya(item.image, prompt=SURYA_OLLAMA_TBL_JSON_PROMPT)
                content = str(result.get("message", {}).get("content", ""))
                surya_result:BatchOutputItem = BatchOutputItem(
                    raw=content,
                    error=False,
                    metadata=item.metadata,
                    token_count=len(content) if content else 0
                )
                surya_results.append(surya_result)
        else:
            manager = self.manager or get_default_manager()
            surya_results = manager.generate(batch)

        results: List[TableResult] = []
        for out in surya_results:
            ix = out.metadata["image_index"]
            img = batch[ix].image
            w, h = img.size
            page_bbox = [0, 0, float(w), float(h)]
            html = glm_results.get(ix)
            if out.error or not out.raw:
                results.append(
                    TableResult(
                        rows=[],
                        cols=[],
                        cells=[],
                        image_bbox=page_bbox,
                        raw=html,
                        mode="simple",
                        error=True,
                    )
                )
                continue
            try:
                elements = parse_table_rec(out.raw)
            except Exception as e:
                logger.warning(
                    f"Table rec parse failed: {e}; raw[:200]={out.raw[:200]!r}"
                )
                results.append(
                    TableResult(
                        rows=[],
                        cols=[],
                        cells=[],
                        image_bbox=page_bbox,
                        raw=html,
                        mode="simple",
                        error=True,
                    )
                )
                continue

            rows: List[TableRow] = []
            cols: List[TableCol] = []
            for el in elements:
                pixel_bbox = denorm_bbox(el.bbox, w, h, scale=settings.BBOX_SCALE)
                poly = _polygon_from_bbox(pixel_bbox)
                if el.label == "Row":
                    rows.append(TableRow(polygon=poly, row_id=len(rows)))
                else:
                    cols.append(TableCol(polygon=poly, col_id=len(cols)))

            # Derive cells geometrically (row × column intersections)
            cells: List[TableCell] = []
            cell_id = 0
            for row in rows:
                for col in cols:
                    inter = _intersect_bbox(row.bbox, col.bbox)
                    if inter is None:
                        continue
                    cells.append(
                        TableCell(
                            polygon=_polygon_from_bbox(inter),
                            row_id=row.row_id,
                            col_id=col.col_id,
                            cell_id=cell_id,
                        )
                    )
                    cell_id += 1
            results.append(
                TableResult(
                    rows=rows,
                    cols=cols,
                    cells=cells,
                    image_bbox=page_bbox,
                    raw=html,
                    mode="simple",
                    error=False,
                )
            )
        return results


# ---------------------------------------------------------------------------
# HTML table reconstruction from the spatial grid
# ---------------------------------------------------------------------------

def _attr_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _bbox_of(obj):
    b = _attr_get(obj, "bbox")
    return list(b) if b is not None else None


class _TableCellParser(HTMLParser):
    """Parse a <table> HTML fragment into rows of cells.

    Each cell is a dict with keys: text, rowspan, colspan, header.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: List[List[dict]] = []
        self._cur: Optional[dict] = None
        self._buf: List[str] = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._in_cell = True
            self._buf = []
            try:
                rs = int(a.get("rowspan", "1") or 1)
            except (TypeError, ValueError):
                rs = 1
            try:
                cs = int(a.get("colspan", "1") or 1)
            except (TypeError, ValueError):
                cs = 1
            self._cur = {
                "rowspan": max(1, rs),
                "colspan": max(1, cs),
                "header": tag == "th",
                "text": "",
            }
        elif tag == "br" and self._in_cell:
            self._buf.append("\n")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cur is not None:
            self._cur["text"] = "".join(self._buf).strip()
            if self.rows:
                self.rows[-1].append(self._cur)
            self._cur = None
            self._in_cell = False

    def handle_data(self, data):
        if self._in_cell:
            self._buf.append(data)


def _parse_html_table(html: str):
    """Return (header_row_count, rows) from an HTML <table> string."""
    p = _TableCellParser()
    p.feed(html or "")
    header_rows = 0
    for row in p.rows:
        if row and all(c.get("header") for c in row):
            header_rows += 1
        else:
            break
    return header_rows, p.rows


def _grid_dims_from_spatial(rows: Sequence, cols: Sequence) -> Tuple[int, int]:
    """Return (n_rows, n_cols) from the spatial row/col bands."""
    row_b = sorted(
        [b for b in (_bbox_of(r) for r in rows) if b is not None],
        key=lambda b: b[1],
    )
    col_b = sorted(
        [b for b in (_bbox_of(c) for c in cols) if b is not None],
        key=lambda b: b[0],
    )
    return len(row_b), len(col_b)


def reconstruct_html_table(
    result: TableResult | Dict[str, Any],
    fill_text: bool = True,
    border: int = 1,
) -> str:
    """Reconstruct a proper rectangular ``<table>`` from a table-rec result.

    The spatial grid (``rows`` / ``cols`` / ``cells``) is treated as ground
    truth for the table dimensions. The original ``html`` (when present)
    supplies the cell text and the *intended* colspan/rowspan values, but its
    placement is broken — so we re-lay every cell into a clean ``R x C`` grid
    using the standard HTML cell-placement algorithm (skipping positions
    already occupied by rowspans), then fill any remaining holes with empty
    cells. Hole filling runs **right-to-left, bottom-to-top** so trailing
    gaps close against the rightmost placed cell.

    Args:
        result: dict (or TableResult) with rows/cols/cells and optionally
            ``html``.
        fill_text: pull text from ``result['html']`` when available.
        border: value for the ``border`` attribute of the emitted table.

    Returns:
        HTML string for a rectangular ``<table>`` with no broken spans.
    """
    rows = _attr_get(result, "rows") or []
    cols = _attr_get(result, "cols") or []
    html = _attr_get(result, "html")

    n_rows, n_cols = _grid_dims_from_spatial(rows, cols)

    # If we have no spatial bands, fall back to the html's own row/col count.
    header_rows: int = 0
    html_rows: List[List[dict]] = []
    if fill_text and html:
        header_rows, html_rows = _parse_html_table(html)
    if n_rows == 0:
        n_rows = len(html_rows)
    if n_cols == 0:
        n_cols = max(
            (sum(max(1, int(c.get("colspan", 1) or 1)) for c in row) for row in html_rows),
            default=0,
        )
    if n_rows == 0 or n_cols == 0:
        return ""

    # occ[r][c] = True once a cell covers that atomic position.
    occ = [[False] * n_cols for _ in range(n_rows)]
    # placed cells: (r, c, rowspan, colspan, text, header)
    placed: List[Tuple[int, int, int, int, str, bool]] = []

    # Standard left-to-right, top-to-bottom placement of the html cells.
    # A cell is placed at the first column where its *entire* rs x cs
    # rectangle is free (HTML table layout semantics).
    for r, row in enumerate(html_rows):
        if r >= n_rows:
            break
        c = 0
        for cell in row:
            rs = max(1, int(cell.get("rowspan", 1) or 1))
            cs = max(1, int(cell.get("colspan", 1) or 1))
            text = cell.get("text", "") if fill_text else ""
            header = bool(cell.get("header"))
            rs = min(rs, n_rows - r)
            if rs <= 0:
                continue
            # find first column where the whole rectangle is free
            placed_ok = False
            while c + cs <= n_cols:
                if all(not occ[r + dr][c + dc] for dr in range(rs) for dc in range(cs)):
                    placed_ok = True
                    break
                c += 1
            if not placed_ok:
                # shrink colspan to fit remaining width rather than drop text
                cs = min(cs, n_cols - c)
                if cs <= 0:
                    continue
            placed.append((r, c, rs, cs, text, header))
            for dr in range(rs):
                for dc in range(cs):
                    occ[r + dr][c + dc] = True
            c += cs

    # Hole filling, right-to-left, bottom-to-top: any atomic position not
    # covered becomes an empty 1x1 cell so the table is a full rectangle.
    for r in range(n_rows - 1, -1, -1):
        for c in range(n_cols - 1, -1, -1):
            if not occ[r][c]:
                placed.append((r, c, 1, 1, "", False))
                occ[r][c] = True

    # Index cells by anchor for emission, sorted in reading order.
    by_anchor = {(r, c): (rs, cs, text, header) for (r, c, rs, cs, text, header) in placed}

    out = [f'<table border="{border}">']
    thead_open = False
    tbody_open = False
    for r in range(n_rows):
        is_header = r < header_rows
        if is_header and not thead_open:
            out.append("<thead>")
            thead_open = True
        if not is_header and not tbody_open:
            if thead_open:
                out.append("</thead>")
                thead_open = False
            out.append("<tbody>")
            tbody_open = True
        out.append("<tr>")
        for c in range(n_cols):
            key = (r, c)
            if key in by_anchor:
                rs, cs, text, header = by_anchor[key]
                tag = "th" if (header or is_header) else "td"
                attrs = []
                if rs > 1:
                    attrs.append(f'rowspan="{rs}"')
                if cs > 1:
                    attrs.append(f'colspan="{cs}"')
                attr_str = (" " + " ".join(attrs)) if attrs else ""
                out.append(f"<{tag}{attr_str}>{text}</{tag}>")
            # else: covered by a span from an earlier anchor -> skip
        out.append("</tr>")
    if thead_open:
        out.append("</thead>")
    if tbody_open:
        out.append("</tbody>")
    out.append("</table>")
    return "\n".join(out)
