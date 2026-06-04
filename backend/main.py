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

from . import inference
from .hardware import detect_device, device_label

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_BYTES = 15 * 1024 * 1024  # 15 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model up at startup so the first request isn't slow.
    inference.warmup()
    yield


app = FastAPI(title="RF-DETR Object Detection API", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "hardware": detect_device(),
        "hardware_label": device_label(),
        **inference.model_info(),
    }


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG/PNG/WebP/BMP.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 15 MB).")

    threshold = min(max(threshold, 0.0), 1.0)

    try:
        result = inference.detect(data, threshold=threshold)
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


# Serve the frontend (index.html + assets) at the root. Mounted last so it does
# not shadow the /api routes.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
