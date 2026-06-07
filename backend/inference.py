"""RF-DETR inference wrapper.

Loads an RF-DETR model once (singleton), runs object detection -- or instance
segmentation -- on uploaded images, and returns structured results plus an
annotated image. The model is placed on a GPU when one is usable and otherwise
falls back to CPU; the device that actually ran each request is reported back to
the caller.
"""
from __future__ import annotations

import io
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import supervision as sv
from PIL import Image

from .hardware import detect_device, device_label, device_string

# Task is configurable: "segment" (instance masks + boxes) or "detect" (boxes
# only). Segmentation is the default so detected objects are also segmented.
TASK = os.environ.get("RFDETR_TASK", "segment").lower()
# Model size is configurable; default to "nano" for snappy CPU inference in this
# proof-of-concept. Options: nano, small, medium, large.
MODEL_SIZE = os.environ.get("RFDETR_MODEL", "nano").lower()

_DETECT_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}
_SEGMENT_CLASSES = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
}


def _model_class_name() -> str:
    table = _SEGMENT_CLASSES if TASK == "segment" else _DETECT_CLASSES
    return table.get(MODEL_SIZE, table["nano"])


_model = None
_model_lock = threading.Lock()
# RF-DETR / torch inference is not guaranteed thread-safe; serialize predicts.
_predict_lock = threading.Lock()


def _build_model():
    # On FIPS-enabled hosts, OpenSSL disables MD5 and RF-DETR's weight-download
    # integrity check (hashlib.md5) crashes startup. Apply the compatibility
    # shim before importing rfdetr so the checksum uses usedforsecurity=False.
    from .fips import enable as _enable_fips_md5

    _enable_fips_md5()

    import rfdetr

    cls_name = _model_class_name()
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
    return {
        "model": _model_class_name(),
        "model_size": MODEL_SIZE,
        "task": "segment" if TASK == "segment" else "detect",
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


def _polygon_area(p: np.ndarray) -> float:
    """Area of a polygon given as an (N, 2) point array (shoelace formula)."""
    if p is None or len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def _largest_polygon(mask: np.ndarray) -> Optional[List[List[int]]]:
    """Return the largest contour of a boolean mask as a list of [x, y] points."""
    try:
        polygons = sv.mask_to_polygons(mask)
    except Exception:
        return None
    if not polygons:
        return None
    largest = max(polygons, key=_polygon_area)
    # Downsample very dense polygons to keep the JSON payload reasonable.
    pts = largest.astype(int)
    if len(pts) > 200:
        step = len(pts) // 200 + 1
        pts = pts[::step]
    return [[int(x), int(y)] for x, y in pts]


def _annotate(
    image: Image.Image,
    detections: "sv.Detections",
    labels: List[str],
    draw_boxes: bool = True,
    draw_masks: bool = True,
) -> Image.Image:
    """Draw a light mask overlay and/or bounding boxes (plus labels) onto a copy."""
    annotated = image.copy()
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)

    # Light translucent mask overlay first, then boxes and labels on top.
    if draw_masks and getattr(detections, "mask", None) is not None:
        annotated = sv.MaskAnnotator(opacity=0.4).annotate(annotated, detections)

    if draw_boxes:
        annotated = sv.BoxAnnotator(thickness=thickness).annotate(annotated, detections)

    # Labels are always drawn (anchored to the box corner) so objects stay
    # identifiable even in mask-only mode.
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_padding=max(2, thickness),
    )
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    return annotated


VALID_MODES = ("box", "mask", "both")


def detect(image_bytes: bytes, threshold: float = 0.5, mode: str = "both") -> Dict[str, Any]:
    """Run detection/segmentation on raw image bytes.

    ``mode`` selects what is drawn on the annotated image and what each detection
    returns:
      - "box"  -> bounding boxes only
      - "mask" -> segmentation masks/polygons only
      - "both" -> boxes plus a light mask overlay (default)

    Returns a dict with the per-object results, an annotated image (PIL), timing
    and the hardware that performed the inference.
    """
    mode = mode if mode in VALID_MODES else "both"
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = get_model()

    start = time.perf_counter()
    with _predict_lock:
        detections = model.predict(image, threshold=threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    masks = getattr(detections, "mask", None)
    has_masks = masks is not None
    want_boxes = mode in ("box", "both")
    want_masks = mode in ("mask", "both") and has_masks

    results: List[Dict[str, Any]] = []
    labels: List[str] = []
    n = len(detections)
    for i in range(n):
        class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
        confidence = (
            float(detections.confidence[i]) if detections.confidence is not None else 0.0
        )
        name = _coco_class_name(class_id)
        labels.append(f"{name} {confidence:.2f}")
        item: Dict[str, Any] = {
            "class_id": class_id,
            "class_name": name,
            "confidence": round(confidence, 4),
        }
        if want_boxes:
            xyxy = [float(v) for v in detections.xyxy[i]]
            item["box"] = {
                "x1": round(xyxy[0], 1),
                "y1": round(xyxy[1], 1),
                "x2": round(xyxy[2], 1),
                "y2": round(xyxy[3], 1),
            }
        if want_masks:
            mask = np.asarray(masks[i]).astype(bool)
            item["mask_area"] = int(mask.sum())
            item["polygon"] = _largest_polygon(mask)
        results.append(item)

    annotated = _annotate(
        image, detections, labels, draw_boxes=want_boxes, draw_masks=want_masks
    )

    return {
        "mode": mode,
        "task": "segment" if has_masks else "detect",
        "has_masks": has_masks,
        "detections": results,
        "count": len(results),
        "annotated_image": annotated,
        "inference_ms": round(elapsed_ms, 1),
        "image_size": {"width": image.width, "height": image.height},
        "hardware": detect_device(),
        "hardware_label": device_label(),
        "model": getattr(model, "__rfdetr_class__", _model_class_name()),
    }
