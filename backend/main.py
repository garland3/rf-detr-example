"""FastAPI backend exposing RF-DETR object detection.

Endpoints:
  GET  /api/health    -> liveness + hardware/model info
  POST /api/detect    -> run detection on an uploaded image
  GET  /              -> the single-page upload UI (static frontend)
"""
from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import inference, jobs
from .hardware import detect_device, device_label

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_BYTES = 15 * 1024 * 1024  # 15 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model up at startup so the first request isn't slow.
    inference.warmup()
    # Start the background queue workers for the async job API.
    await jobs.start_workers()
    yield
    await jobs.stop_workers()


app = FastAPI(title="RF-DETR Object Detection API", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "hardware": detect_device(),
        "hardware_label": device_label(),
        **inference.model_info(),
    }


async def _validate_upload(file: UploadFile, mode: str) -> bytes:
    """Shared validation for upload endpoints; returns the image bytes."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG/PNG/WebP/BMP.",
        )
    if mode not in inference.VALID_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mode: {mode!r}. Use one of {inference.VALID_MODES}.",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 15 MB).")
    return data


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    mode: str = Form("both"),
):
    """Synchronous detection: runs inference inline and returns the result."""
    data = await _validate_upload(file, mode)
    threshold = min(max(threshold, 0.0), 1.0)

    try:
        result = inference.detect(data, threshold=threshold, mode=mode)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the client
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    # Encode annotated image as a base64 data URL for easy display in the browser.
    annotated = result.pop("annotated_image")
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    result["annotated_image"] = "data:image/png;base64," + base64.b64encode(
        buf.getvalue()
    ).decode("ascii")

    return JSONResponse(result)


# --- Queued / async detection API -------------------------------------------
# Submit an image to a bounded queue and poll for the result, instead of running
# inference inline. This protects the single model from being overwhelmed by a
# burst of concurrent uploads. Workers drain the queue in the background.


@app.post("/api/jobs", status_code=202)
async def submit_job(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    mode: str = Form("both"),
):
    data = await _validate_upload(file, mode)
    threshold = min(max(threshold, 0.0), 1.0)
    try:
        job = await jobs.submit(data, threshold=threshold, mode=mode)
    except jobs.QueueFull:
        raise HTTPException(
            status_code=503,
            detail="Queue is full; please retry shortly.",
        )
    return JSONResponse(job, status_code=202)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return JSONResponse(job)


@app.get("/api/queue")
async def queue_stats():
    return jobs.stats()


# Serve the frontend (index.html + assets) at the root. Mounted last so it does
# not shadow the /api routes.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
