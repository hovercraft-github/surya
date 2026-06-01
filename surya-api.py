#!/home/al/local/AI-tools/surya-api/.venv/bin/python
# -*- coding: utf-8 -*-

import fcntl
from filelock import FileLock, Timeout
import logging.config

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import argparse
from time import perf_counter
from PIL import Image

# import pypdfium2
import asyncio

from surya.settings import settings
from surya.inference import SuryaInferenceManager
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


logger = get_logger()

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
            "handlers": ["console"],
            "level": "INFO",
            "propagate": True,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

backend_type = settings.SURYA_INFERENCE_BACKEND or "vllm"
LOCK_FILE = _cache_dir() / "surya-api_server.lock"
LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)


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
        pass


def get_request_count() -> tuple[int, float | None, int | None]:
    lock = FileLock(str(_lock_path(backend_type)))
    try:
        with lock.acquire(timeout=1):
            data = _read_sentinel(backend_type) or {}
            if data and "request_count" not in data:
                data["request_count"] = 0
                data["last_updated"] = perf_counter()
                _write_sentinel(backend_type, data)
            return data.get("request_count", 0), data.get("last_updated"), data.get("port", 0)
    except Timeout:
        return 0, None, None


def cleanup():
    logger.info("Cleaning up resources...")
    lock = FileLock(str(_lock_path(backend_type)))
    try:
        with lock.acquire(timeout=1):
            sentinel = _read_sentinel(backend_type)
            if not sentinel or not sentinel.get("cleanup_kind"):
                logger.info("No active server detected; skipping cleanup.")
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
    logger.info("Cleanup complete.")


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
                logger.warning(
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
        fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
        fcntl.flock(file_descriptor, fcntl.LOCK_UN)
        os.close(file_descriptor)
    except Exception:
        pass

    # atexit._run_exitfuncs()
    print("Done")


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

# Load models once when the application starts
inference_manager = SuryaInferenceManager()
recognizer = RecognitionPredictor(inference_manager)
layout_predictor = LayoutPredictor(inference_manager)


@app.post("/ocr/full/")
async def ocr_full_page(file: UploadFile = File(...)):
    """Full-page OCR that extracts text and returns structured HTML.

    Uses HIGH_ACCURACY_BBOX prompt for accurate full-page OCR with layout detection.
    Returns HTML with block-level structure including bounding boxes and labels.
    """
    try:
        print(f"Received file: {file.filename}, content_type: {file.content_type}")
        _, last_updated, port = get_request_count()
        if last_updated is None or port is None or not probe_health(f"http://{settings.SURYA_INFERENCE_HOST}:{port}"):
            inference_manager.stop()
            inference_manager.start()
        update_request_count()
        start_time = perf_counter()
        image = Image.open(file.file)
        # Use full_page=True for direct HTML extraction with HIGH_ACCURACY_BBOX_PROMPT
        predictions = recognizer([image], full_page=True)

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
        print(
            f"OCR completed for {file.filename}, extracted {len(blocks_data)} blocks in {end_time - start_time:.2f} seconds."
        )
        return {
            "html": full_html,
            "blocks": blocks_data,
            "page_bbox": predictions[0].image_bbox if predictions else [],
        }
    except Exception as e:
        msg = str(e)
        logger.error(f"OCR failed for {file.filename}: {msg}")
        raise HTTPException(
            status_code=500, detail=msg or "An error occurred during OCR processing."
        )
    finally:
        update_request_count(delta=-1)


@app.post("/ocr/")
async def detect_text(file: UploadFile = File(...), lang: str = "en,ru"):
    try:
        print(f"Received file: {file.filename}, content_type: {file.content_type}")
        update_request_count()
        start_time = perf_counter()
        image = Image.open(file.file)
        # langs = lang.split(",")
        predictions = recognizer([image])  # , [langs], detector, layout_predictor)

        text_output = []
        for prediction in predictions:
            for block in prediction.blocks:
                text_output.append(block.html)

        end_time = perf_counter()
        print(
            f"OCR completed for {file.filename}, extracted {len(text_output)} blocks in {end_time - start_time:.2f} seconds."
        )
        return {"html": "\n".join(text_output), "predictions": predictions}
    except Exception as e:
        logger.error(f"OCR failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e.args))
    finally:
        update_request_count(delta=-1)


@app.post("/detect_layout/")
async def detect_layout(
    file: UploadFile = File(...), return_image: bool = Query(False)
):
    try:
        update_request_count()
        # Read the uploaded file
        contents = await file.read()
        file_type = file.content_type

        # Check if the file is a PDF or an image
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Perform layout detection
        # line_predictions = batch_text_detection([pil_image], model, processor)
        layout = LayoutPredictor(inference_manager)
        layout_predictions = layout([pil_image])

        # Create a result image with bounding boxes
        predictions = recognizer([pil_image.copy()], layout_predictions)

        # if return_image and layout_image:
        #     # Return the image with layout bounding boxes
        #     img_byte_arr = io.BytesIO()
        #     layout_image[0]. #.save(img_byte_arr, format='PNG')
        #     img_byte_arr.seek(0)
        #     return StreamingResponse(img_byte_arr, media_type="image/png")
        # else:
        #     # Return JSON response with layout data
        #     layout_data = [p.model_dump() for p in layout_predictions[0].bboxes]
        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
    finally:
        update_request_count(delta=-1)


# Run the application
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Surya API Server")
    parser.add_argument(
        "--port", type=int, default=8522, help="Port number to run the server on"
    )
    args = parser.parse_args()

    n_workers = settings.SURYA_INFERENCE_PARALLEL or 1
    uvicorn.run(
        "surya-api:app",
        host="0.0.0.0",
        port=args.port,
        workers=n_workers,
        log_config=None,
    )  #  , reload=True
