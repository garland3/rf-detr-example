"""RF-DETR inference wrapper.

Loads an RF-DETR model once (singleton), runs object detection on uploaded
images, and returns structured detections plus an annotated image. The model is
placed on a GPU when one is usable and otherwise falls back to CPU; the device
that actually ran each request is reported back to the caller.
"""
from __future__ import annotations

import io
import os
import threading
import time
from typing import Any, Dict, List

import numpy as np
import supervision as sv
from PIL import Image

from .hardware import detect_device, device_label, device_string

# Model size is configurable; default to "nano" for snappy CPU inference in this
# proof-of-concept. Options: nano, small, medium, large.
MODEL_SIZE = os.environ.get("RFDETR_MODEL", "nano").lower()

_MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

_model = None
_model_lock = threading.Lock()
# RF-DETR / torch inference is not guaranteed thread-safe; serialize predicts.
_predict_lock = threading.Lock()


def _build_model():
    import rfdetr

    cls_name = _MODEL_CLASSES.get(MODEL_SIZE, "RFDETRNano")
    model_cls = getattr(rfdetr, cls_name)
    device = device_string()
    try:
        model = model_cls(device=device)
    except TypeError:
        # Older/newer signatures may not accept `device`; let it auto-detect.
        model = model_cls()

    # Optionally fuse/compile the model for lower latency. Best-effort: on some
    # devices (e.g. CPU-only) this can fail or be unsupported, so never fatal.
    if os.environ.get("RFDETR_OPTIMIZE", "1") == "1":
        optimize = getattr(model, "optimize_for_inference", None)
        if callable(optimize):
            try:
                optimize()
            except Exception:
                pass
    return model, cls_name


def get_model():
    """Return the lazily-initialised singleton model, building it if needed."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                model, cls_name = _build_model()
                _model = model
                _model.__rfdetr_class__ = cls_name  # type: ignore[attr-defined]
    return _model


def warmup() -> None:
    """Build the model and run a tiny inference so the first real request is fast."""
    model = get_model()
    dummy = Image.new("RGB", (640, 480), (32, 32, 32))
    with _predict_lock:
        try:
            model.predict(dummy, threshold=0.5)
        except Exception:
            # Warmup is best-effort; real requests will surface any error.
            pass


def model_info() -> Dict[str, Any]:
    cls_name = _MODEL_CLASSES.get(MODEL_SIZE, "RFDETRNano")
    return {
        "model": cls_name,
        "model_size": MODEL_SIZE,
        "loaded": _model is not None,
    }


def _coco_class_name(class_id: int) -> str:
    from rfdetr.assets.coco_classes import COCO_CLASSES

    if isinstance(COCO_CLASSES, dict):
        return COCO_CLASSES.get(int(class_id), str(class_id))
    try:
        return COCO_CLASSES[int(class_id)]
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def _annotate(image: Image.Image, detections: "sv.Detections", labels: List[str]) -> Image.Image:
    """Draw bounding boxes and labels onto a copy of the image."""
    annotated = image.copy()
    resolution = max(image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_padding=max(2, thickness),
    )
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    return annotated


def detect(image_bytes: bytes, threshold: float = 0.5) -> Dict[str, Any]:
    """Run detection on raw image bytes.

    Returns a dict with the detections, an annotated image (PIL), timing and the
    hardware that performed the inference.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = get_model()

    start = time.perf_counter()
    with _predict_lock:
        detections = model.predict(image, threshold=threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    results: List[Dict[str, Any]] = []
    labels: List[str] = []
    n = len(detections)
    for i in range(n):
        class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
        confidence = (
            float(detections.confidence[i]) if detections.confidence is not None else 0.0
        )
        xyxy = [float(v) for v in detections.xyxy[i]]
        name = _coco_class_name(class_id)
        labels.append(f"{name} {confidence:.2f}")
        results.append(
            {
                "class_id": class_id,
                "class_name": name,
                "confidence": round(confidence, 4),
                "box": {
                    "x1": round(xyxy[0], 1),
                    "y1": round(xyxy[1], 1),
                    "x2": round(xyxy[2], 1),
                    "y2": round(xyxy[3], 1),
                },
            }
        )

    annotated = _annotate(image, detections, labels)

    return {
        "detections": results,
        "count": len(results),
        "annotated_image": annotated,
        "inference_ms": round(elapsed_ms, 1),
        "image_size": {"width": image.width, "height": image.height},
        "hardware": detect_device(),
        "hardware_label": device_label(),
        "model": getattr(model, "__rfdetr_class__", _MODEL_CLASSES.get(MODEL_SIZE)),
    }
