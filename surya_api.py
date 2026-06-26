#!/home/al/local/AI-tools/surya-api/.venv/bin/python
# -*- coding: utf-8 -*-

import multiprocessing
# This MUST be called before any other app setups or process creations
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import portalocker
from filelock import FileLock, Timeout
import logging
import logging.config
import copy

from fastapi import FastAPI, File, UploadFile, Query, Path, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import argparse
from time import perf_counter
from PIL import Image

import asyncio

from crown.table_rec import TableExtPredictor
from crown.utils import crop_by_percent, crop_by_side_percent, get_page_image, trim_empty_background
from crown.utils import poligon_expand
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition.schema import PageOCRResult
from surya.settings import settings
# from surya.inference import SuryaInferenceManager
from crown.inference import CrownSuryaInferenceManager, BatchBusyError
from surya.recognition import RecognitionPredictor

# from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.logging import get_logger
from surya.inference.backends.spawn import (
    _cache_dir,
    _lock_path,
    _read_sentinel,
    _write_sentinel,
    _delete_sentinel,
    _stop_docker_container,
    _stop_process,
    probe_health,
)

from contextlib import asynccontextmanager

from anyio.lowlevel import RunVar
from anyio import CapacityLimiter

from surya.table_rec.schema import TableCell, TableCol, TableResult, TableRow


logger = get_logger()
crown_logger = logging.getLogger("crown")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # Crucial: Don't kill third-party loggers
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s [%(name)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        # Explicitly configure your chosen third-party library logger here
        "surya": {
            # "handlers": ["console"],
            "level": "INFO",
            "propagate": True,
        },
        "crown": {
            "level": "DEBUG",
            "propagate": True,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

backend_type = settings.SURYA_INFERENCE_BACKEND or "vllm"
LOCK_FILE = _cache_dir() / "surya-api_server.lock"
LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
inference_manager = CrownSuryaInferenceManager()



def update_request_count(delta: int = 1) -> None:
    lock = FileLock(str(_lock_path(backend_type)))
    try:
        with lock.acquire(timeout=1):
            data = _read_sentinel(backend_type)
            if not data:
                return
            count = data.get("request_count", 0) + delta
            if count < 0:
                count = 0
            data["request_count"] = count
            data["last_updated"] = perf_counter()
            _write_sentinel(backend_type, data)
    except Timeout:
        crown_logger.warning("update_request_count lock timed out")
    except Exception as e:
        crown_logger.error(f"update_request_count failure: {e}")
        raise

def get_request_count() -> tuple[int, float | None, int | None]:
    lock = FileLock(str(_lock_path(backend_type)))
    try:
        with lock.acquire(timeout=1):
            data = _read_sentinel(backend_type) or {}
            if data and "request_count" not in data:
                data["request_count"] = 0
                data["last_updated"] = perf_counter()
                _write_sentinel(backend_type, data)
            return (
                data.get("request_count", 0),
                data.get("last_updated"),
                data.get("port", 0),
            )
    except Timeout:
        return 0, None, None


def cleanup():
    crown_logger.info("Cleaning up resources...")
    lock = FileLock(str(_lock_path(backend_type)))
    try:
        with lock.acquire(timeout=1):
            sentinel = _read_sentinel(backend_type)
            if not sentinel or not sentinel.get("cleanup_kind"):
                crown_logger.info("No active server detected; skipping cleanup.")
                return
            cleanup_id = sentinel.get("cleanup_id")
            pid = sentinel.get("pid")
            if sentinel.get("cleanup_kind") == "docker" and cleanup_id:
                _stop_docker_container(cleanup_id)
            elif sentinel.get("cleanup_kind") == "process" and pid:
                _stop_process(pid, backend_type)
    except Timeout:
        pass
    finally:
        _delete_sentinel(backend_type)
    crown_logger.info("Cleanup complete.")


async def resource_management_loop():
    try:
        timer = 60.0
        while True:
            await asyncio.sleep(timer)
            cnt, last_updated, _ = get_request_count()
            if (
                cnt == 0
                and last_updated is not None
                and (perf_counter() - last_updated) > timer
            ):
                crown_logger.warning(
                    f"No requests in the last {int(timer)} seconds; stopping inference server to save resources."
                )
                cleanup()
    except asyncio.CancelledError:
        pass
    cleanup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reduce FastAPI threads because we use separate worker processes.
    RunVar("_default_thread_limiter").set(CapacityLimiter(10))

    # Open or create the lock file
    file_descriptor = os.open(LOCK_FILE, os.O_CREAT | os.O_WRONLY)
    background_task = None

    try:
        # Attempt non-blocking exclusive lock
        portalocker.lock(file_descriptor, portalocker.LOCK_EX | portalocker.LOCK_NB)
        print("Server starting...")
        background_task = asyncio.create_task(resource_management_loop())
    except BlockingIOError:
        # Another worker process already has the lock
        pass

    yield

    # Gracefully close tasks and release the lock on shutdown
    if background_task:
        background_task.cancel()
    try:
        portalocker.unlock(file_descriptor)
        os.close(file_descriptor)
    except Exception:
        pass

    # atexit._run_exitfuncs()
    print("Done")


def load_and_preprocess_image(
    file: UploadFile,
    dpi: int | None,
    trim: float,
    crop: float,
    crop_left: float = 0.0,
    crop_right: float = 0.0,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
) -> Image.Image:
    """Load an uploaded file (PDF or image) into a PIL image and apply the
    shared trim/crop preprocessing used by every OCR endpoint.

    - PDF inputs are rendered at the given DPI to a single RGB image.
    - Image inputs are decoded and converted to RGB.
    - ``trim`` removes empty/solid borders by entropy threshold.
    - ``crop_*`` crop a percentage off the corresponding side before OCR;
      when any of the per-side values is non-zero, it is used; otherwise
      the symmetric ``crop`` value is applied to all four sides.
    """
    if file.content_type == "application/pdf":
        image = get_page_image(file, page_num=1, dpi=dpi or 300)
    else:
        image = Image.open(file.file).convert("RGB")
    if trim > 0.0:
        image = trim_empty_background(image, threshold=trim)
    if crop_left or crop_right or crop_top or crop_bottom:
        image = crop_by_side_percent(
            image,
            left=crop_left or crop,
            right=crop_right or crop,
            top=crop_top or crop,
            bottom=crop_bottom or crop,
        )
    elif crop > 0.0:
        image = crop_by_percent(image, crop)
    return image


# Initialize FastAPI
app = FastAPI(lifespan=lifespan)

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ocr/full/")
async def ocr_full_page(request: Request, file: UploadFile = File(...),
    dpi: int | None = Query(
        default=300,
        ge=72,
        le=600,
        description="Optional: DPI for rendering PDF pages to images. Higher DPI can improve OCR accuracy but increases processing time and memory usage. 300 (the default) is a common choice for good quality OCR."
        ),
    trim: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Optional entropy threshold to detect and trim empty/solid fill background from each side of the image before OCR. Can help with empty borders that can provoke hallucinations. 0 means no trimming, 0.2 .. 0.5 recommended value, default is 0.5."
        ),
    crop: float = Query(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Optional percent to crop from each side of the image before OCR. Can help with noisy borders that can provoke hallucinations. 0 means no cropping, 50 means crop half of the image from each side, 0.2 .. 0.5 recommended value."
        ),
    crop_left: float = Query(
        default=0.0,
        ge=0.0,
        le=99.0,
        description="Optional percent to crop from the LEFT side of the image before OCR. Overrides the 'crop' value for the left side. 0 means no cropping, 50 means crop half of the image width, 0.2 .. 0.5 recommended value."
        ),
    crop_right: float = Query(
        default=0.0,
        ge=0.0,
        le=99.0,
        description="Optional percent to crop from the RIGHT side of the image before OCR. Overrides the 'crop' value for the right side. 0 means no cropping, 50 means crop half of the image width, 0.2 .. 0.5 recommended value."
        ),
    crop_top: float = Query(
        default=0.0,
        ge=0.0,
        le=99.0,
        description="Optional percent to crop from the TOP side of the image before OCR. Overrides the 'crop' value for the top side. 0 means no cropping, 50 means crop half of the image height, 0.2 .. 0.5 recommended value."
        ),
    crop_bottom: float = Query(
        default=0.0,
        ge=0.0,
        le=99.0,
        description="Optional percent to crop from the BOTTOM side of the image before OCR. Overrides the 'crop' value for the bottom side. 0 means no cropping, 50 means crop half of the image height, 0.2 .. 0.5 recommended value."
        ),
    ):
    """Full-page OCR that extracts text and returns structured HTML.

    Can be inaccurate, slow and even unstable on large documents or complex tables.
    Returns HTML with block-level structure including bounding boxes and labels.
    For small, easy pages only.
    """
    try:
        crown_logger.info(
            f"Received file: {file.filename}, content_type: {file.content_type}"
        )
        if 100.0 - crop_left - crop_right < 1.0 or 100.0 - crop_top - crop_bottom < 1.0:
            raise HTTPException(
                status_code=400,
                detail="Inconsistent crop values",
            )
        recognizer = RecognitionPredictor(inference_manager)
        _, last_updated, port = get_request_count()
        if (
            last_updated is None
            or port is None
            or not probe_health(f"http://{settings.SURYA_INFERENCE_HOST}:{port}")
        ):
            inference_manager.stop()
            inference_manager.start()
        async with inference_manager.booking():
            update_request_count()
            try:
                start_time = perf_counter()
                image = await load_and_preprocess_image_async(
                    file,
                    dpi=dpi,
                    trim=trim,
                    crop=crop,
                    crop_left=crop_left,
                    crop_right=crop_right,
                    crop_top=crop_top,
                    crop_bottom=crop_bottom,
                )
                # Use full_page=True for direct HTML extraction with HIGH_ACCURACY_BBOX_PROMPT
                predictions = await asyncio.to_thread(recognizer, [image], full_page=True)

                if not predictions:
                    return {"html": "", "blocks": []}

                # Build structured block list
                blocks_data = []
                for prediction in predictions:
                    for block in prediction.blocks:
                        if block.html:  # Skip empty/skipped blocks
                            blocks_data.append(
                                {
                                    "label": block.label,
                                    "html": block.html,
                                    "polygon": block.polygon,
                                    "confidence": block.confidence,
                                    "reading_order": block.reading_order,
                                }
                            )

                # Assemble full-page HTML by combining all blocks in reading order
                html_parts = []
                for block in blocks_data:
                    # Each block already has HTML with proper structure from the model
                    html_parts.append(
                        f'<div class="block" data-label="{block["label"]}">{block["html"]}</div>'
                    )

                full_html = '<div id="ocr-page">' + "\n".join(html_parts) + "</div>"
                end_time = perf_counter()
                crown_logger.info(
                    f"OCR completed for {file.filename}, extracted {len(blocks_data)} blocks in {end_time - start_time:.2f} seconds."
                )
                if await request.is_disconnected():
                    crown_logger.warning(f"Client disconnected before OCR response for {file.filename}.")
                    return {"blocks": [], "html": ""}
                return {
                    "html": full_html,
                    "blocks": blocks_data,
                    "page_bbox": predictions[0].image_bbox if predictions else [],
                }
            finally:
                update_request_count(delta=-1)
    except BatchBusyError as e:
        crown_logger.warning(f"Batch busy for {file.filename}: {e}")
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": str(int(max(e.retry_after, 1)))},
        )
    except Exception as e:
        msg = str(e)
        crown_logger.error(f"OCR failed for {file.filename}: {msg}")
        raise HTTPException(
            status_code=500, detail=msg or "An error occurred during OCR processing."
        )

def text_recognition(
    img: Image.Image,
    layouts: list[LayoutResult],
) -> tuple[list[PageOCRResult], list[tuple[int, ...]]]:
    from surya.inference.prompts import LAYOUT_LABEL_SET
    # desired_labels = set(LAYOUT_LABEL_SET) - {"Image", "Figure", "Diagram", "Table", "Table-Of-Contents", "Complex-Block"}
    desired_labels = set(LAYOUT_LABEL_SET) - {"Image", "Figure", "Diagram", "Table", "Table-Of-Contents"}
    layout: LayoutResult = layouts[0]
    texts = [
        b
        for b in layout.bboxes
        if b.label in desired_labels or b.raw_label in desired_labels or b.label.startswith("Cell")
    ]
    if not texts:
        return [], []
    filterred_layout = copy.copy(layout)
    filterred_layout.bboxes = texts
    recognizer = RecognitionPredictor(inference_manager)
    predictions = recognizer([img], [filterred_layout])  # , full_page=False
    filtered_bboxes = [tuple(int(c) for c in b.bbox) for b in texts]
    return predictions, filtered_bboxes


def table_recognition(
    img: Image.Image,
    layout: LayoutResult,
    mode: str
) -> tuple[list[TableResult], list[tuple[int, ...]]]:
    tables = [b for b in layout.bboxes if b.label in ("Table", "Table-Of-Contents")]
    if not tables:
        return [], []
    table_bboxes = [tuple(int(c) for c in b.bbox) for b in tables]
    table_imgs = [img.crop(b) for b in table_bboxes]
    table_counts = [b.count for b in tables]
    table_rec_predictor = TableExtPredictor(inference_manager)
    table_preds = table_rec_predictor.predict_flexible(table_imgs, counts=table_counts, mode="td")
    if mode in ["td"]:
        table_preds2 = table_rec_predictor.predict_simple(table_imgs)
        if table_preds2:
            for pred, pred2 in zip(table_preds, table_preds2):
                if pred2.error:
                    continue
                pred.cells = pred2.cells
                pred.cols = pred2.cols
                pred.rows = pred2.rows
    return table_preds, table_bboxes


async def text_recognition_async(
    img: Image.Image,
    layouts: list[LayoutResult],
) -> tuple[list[PageOCRResult], list[tuple[int, ...]]]:
    """Async wrapper for :func:`text_recognition` that offloads the blocking
    inference work to a worker thread via :func:`asyncio.to_thread` so the
    event loop stays responsive while the recognizer runs."""
    return await asyncio.to_thread(text_recognition, img, layouts)


async def load_and_preprocess_image_async(
    file: UploadFile,
    dpi: int | None,
    trim: float,
    crop: float,
    crop_left: float = 0.0,
    crop_right: float = 0.0,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
) -> Image.Image:
    """Async wrapper for :func:`load_and_preprocess_image` that offloads the blocking
    image loading and preprocessing work to a worker thread via :func:`asyncio.to_thread`
    so the event loop stays responsive while processing images."""
    return await asyncio.to_thread(
        load_and_preprocess_image,
        file,
        dpi,
        trim,
        crop,
        crop_left,
        crop_right,
        crop_top,
        crop_bottom,
    )


async def table_recognition_async(
    img: Image.Image,
    layout: LayoutResult,
    mode: str,
) -> tuple[list[TableResult], list[tuple[int, ...]]]:
    """Async wrapper for :func:`table_recognition` that offloads the blocking
    inference work to a worker thread via :func:`asyncio.to_thread` so the
    event loop stays responsive while the table recognizer runs."""
    return await asyncio.to_thread(table_recognition, img, layout, mode)


@app.post("/ocr/block/")
async def ocr_blocks(request: Request, file: UploadFile = File(...),
    dpi: int | None = Query(
        default=300,
        ge=72,
        le=600,
        description="Optional: DPI for rendering PDF pages to images. Higher DPI can improve OCR accuracy but increases processing time and memory usage. 300 (the default) is a common choice for good quality OCR."
        ),
    trim: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Optional entropy threshold to detect and trim empty/solid fill background from each side of the image before OCR. Can help with empty borders that can provoke hallucinations. 0 means no trimming, 0.2 .. 0.5 recommended value, default is 0.5."
        ),
    crop: float = Query(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Optional percent to crop from each side of the image before OCR. Can help with noisy borders that can provoke hallucinations. 0 means no cropping, 50 means crop half of the image from each side, 0.2 .. 0.5 recommended value."
        ),
    crop_left: float = Query(
        default=0.0,
        ge=0.0,
        le=99.0,
        description="Optional percent to crop from the LEFT side of the image before OCR. Overrides the 'crop' value for the left side. 0 means no cropping, 50 means crop half of the image width, 0.2 .. 0.5 recommended value."
        ),
    crop_right: float = Query(
        default=0.0,
        ge=0.0,
        le=99.0,
        description="Optional percent to crop from the RIGHT side of the image before OCR. Overrides the 'crop' value for the right side. 0 means no cropping, 50 means crop half of the image width, 0.2 .. 0.5 recommended value."
        ),
    crop_top: float = Query(
        default=0.0,
        ge=0.0,
        le=99.0,
        description="Optional percent to crop from the TOP side of the image before OCR. Overrides the 'crop' value for the top side. 0 means no cropping, 50 means crop half of the image height, 0.2 .. 0.5 recommended value."
        ),
    crop_bottom: float = Query(
        default=0.0,
        ge=0.0,
        le=99.0,
        description="Optional percent to crop from the BOTTOM side of the image before OCR. Overrides the 'crop' value for the bottom side. 0 means no cropping, 50 means crop half of the image height, 0.2 .. 0.5 recommended value."
        ),
    ):
    """Block-based page OCR that extracts text and tables and returns structured HTML.  
    Faster than full-page OCR and able to process very large pages with a lot of content 
    that should be filtered out (like images), also is more accurate for complex table layouts.
    """
    try:
        crown_logger.info(
            f"Received file: {file.filename}, content_type: {file.content_type}"
        )
        if 100.0 - crop_left - crop_right < 1.0 or 100.0 - crop_top - crop_bottom < 1.0:
            raise HTTPException(
                status_code=400,
                detail="Inconsistent crop values",
            )
        _, last_updated, port = get_request_count()
        if (
            last_updated is None
            or port is None
            or not probe_health(f"http://{settings.SURYA_INFERENCE_HOST}:{port}")
        ):
            inference_manager.stop()
            inference_manager.start()
        async with inference_manager.booking():
            update_request_count()
            try:
                start_time = perf_counter()
                image = await load_and_preprocess_image_async(
                    file,
                    dpi=dpi,
                    trim=trim,
                    crop=crop,
                    crop_left=crop_left,
                    crop_right=crop_right,
                    crop_top=crop_top,
                    crop_bottom=crop_bottom,
                )
                layout_predictor = LayoutPredictor(inference_manager)
                layouts = await asyncio.to_thread(layout_predictor, [image])
                if not layouts or not layouts[0].bboxes:
                    return {"html": "", "blocks": []}
                width, height = image.size
                margin = int(max(width, height) / 200)  # Dynamic margin based on image size (e.g., 2px for 1000px image)
                for block in layouts[0].bboxes:
                    poligon_expand(block.polygon, margin=margin)
                # Run text and table recognition sequentially (texts first, then tables)
                # via asyncio.to_thread so each blocking inference call yields the event loop.
                texts, text_bboxes = await text_recognition_async(image, layouts)
                tables, table_bboxes = await table_recognition_async(image, layouts[0], mode="td")
                blocks_data = []
                for pred in texts:
                    for block in pred.blocks:
                        if block.html:  # Skip empty/skipped blocks
                            blocks_data.append(
                                {
                                    "label": block.label,
                                    "html": block.html,
                                    "polygon": block.polygon,
                                    "confidence": block.confidence,
                                    "reading_order": block.reading_order,
                                }
                            )
                for table, bbox in zip(tables, table_bboxes):
                    if table.html:  # Skip empty/skipped tables
                        blocks_data.append(
                            {
                                "label": "Table",
                                "html": table.html,
                                "rows": table.rows,
                                "cols": table.cols,
                                "cells": table.cells,
                                "bbox": bbox,
                            }
                        )

                html_parts = []
                for block in blocks_data:
                    # Each block already has HTML with proper structure from the model
                    html_parts.append(
                        f'<div class="block" data-label="{block["label"]}">{block["html"]}</div>'
                    )

                full_html = '<div id="ocr-page">' + "\n".join(html_parts) + "</div>"
                end_time = perf_counter()
                crown_logger.info(
                    f"OCR completed for {file.filename}, extracted {len(blocks_data)} blocks in {end_time - start_time:.2f} seconds."
                )
                if await request.is_disconnected():
                    crown_logger.warning(f"Client disconnected before OCR response for {file.filename}.")
                    return {"blocks": [], "html": ""}
                return {
                    "blocks": blocks_data,
                    "html": full_html,
                }
            finally:
                update_request_count(delta=-1)
    except BatchBusyError as e:
        crown_logger.warning(f"Batch busy for {file.filename}: {e}")
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": str(int(max(e.retry_after, 1)))},
        )
    except Exception as e:
        msg = str(e)
        crown_logger.error(f"OCR failed for {file.filename}: {msg}")
        raise HTTPException(
            status_code=500, detail=msg or "An error occurred during OCR processing."
        )


# Run the application
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Surya API Server")
    parser.add_argument(
        "--port", type=int, default=8522, help="Port number to run the server on"
    )
    args = parser.parse_args()

    # n_workers = settings.SURYA_INFERENCE_PARALLEL or 1
    uvicorn.run(
        "surya_api:app",
        host="0.0.0.0",
        port=args.port,
        workers=1,
        log_config=None,
    )  #  , reload=True
