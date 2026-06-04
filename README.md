# 🔎 RF-DETR Object Detection — FastAPI + Web UI

A small **proof of concept** that wraps [Roboflow's **RF-DETR**](https://github.com/roboflow/rf-detr)
real-time object detection transformer in a [FastAPI](https://fastapi.tiangolo.com/)
backend with a lightweight web frontend for uploading images and viewing results.

It **auto-detects a GPU and falls back to CPU**, and it always shows **which
hardware performed the inference** — both in the API responses and in the UI.

![Detection result](screenshots/02-results-bus.png)

---

## Features

- **FastAPI backend** exposing a clean JSON API (`/api/health`, `/api/detect`).
- **RF-DETR** inference (defaults to the `nano` model for fast CPU performance;
  `small`/`medium`/`large` selectable via env var).
- **GPU when available, CPU otherwise** — detection is automatic and safe even
  when a GPU is present but unusable (e.g. driver/library mismatch).
- **Hardware transparency** — every response and the UI report the exact device
  (e.g. `GPU (NVIDIA RTX 4090)` or `CPU (Intel Core i9-10900KF)`), model, and
  inference time in milliseconds.
- **Web UI** — drag-and-drop upload, confidence-threshold slider, annotated
  result image, and a detections table.
- **End-to-end tests** with `pytest` + FastAPI's `TestClient`.

## Architecture

```
┌──────────────┐   multipart upload    ┌───────────────────────────┐
│   Browser    │ ───────────────────▶  │   FastAPI  (backend/)     │
│ (frontend/)  │                       │  ┌─────────────────────┐  │
│  index.html  │ ◀───────────────────  │  │ inference.py        │  │
│  app.js      │   JSON: detections,   │  │  RF-DETR (singleton)│  │
│  style.css   │   annotated image,    │  │  hardware.py        │  │
└──────────────┘   hardware, timing    │  └─────────────────────┘  │
                                       └───────────────────────────┘
```

- `backend/hardware.py` — detects whether torch can actually use a CUDA GPU and
  produces a human-readable device label.
- `backend/inference.py` — loads RF-DETR once, runs detection, annotates the
  image (via [`supervision`](https://github.com/roboflow/supervision)), and
  returns structured results plus timing and hardware info.
- `backend/main.py` — FastAPI app; serves the API and the static frontend.
- `frontend/` — single-page UI (no build step, vanilla JS).

## Quick start

Requires [`uv`](https://github.com/astral-sh/uv) (Python 3.12+).

```bash
# 1. Install dependencies (creates a .venv)
uv sync

# 2. Run the app  (downloads the RF-DETR weights, ~349 MB, on first launch)
./run.sh
# or:  uv run uvicorn backend.main:app --host 127.0.0.1 --port 8011
```

Then open **http://127.0.0.1:8011** and upload an image.

### Configuration

| Env var           | Default | Description                                        |
| ----------------- | ------- | -------------------------------------------------- |
| `RFDETR_MODEL`    | `nano`  | Model size: `nano`, `small`, `medium`, `large`.    |
| `RFDETR_OPTIMIZE` | `1`     | Fuse/optimize the model for inference (best-effort).|
| `HOST` / `PORT`   | `127.0.0.1` / `8011` | Bind address for `run.sh`.            |

## API

### `GET /api/health`

```json
{
  "status": "ok",
  "hardware": {
    "device": "cpu",
    "device_type": "CPU",
    "name": "Intel(R) Core(TM) i9-10900KF CPU @ 3.70GHz",
    "cuda_available": false,
    "details": { "torch_version": "2.12.0+cu130" }
  },
  "hardware_label": "CPU (Intel(R) Core(TM) i9-10900KF CPU @ 3.70GHz)",
  "model": "RFDETRNano",
  "model_size": "nano",
  "loaded": true
}
```

### `POST /api/detect`

Multipart form fields: `file` (image), `threshold` (float, default `0.5`).

```bash
curl -s -X POST http://127.0.0.1:8011/api/detect \
  -F "file=@tests/images/people.jpg;type=image/jpeg" \
  -F "threshold=0.5" | jq '{count, inference_ms, hardware_label, detections}'
```

```json
{
  "count": 5,
  "inference_ms": 92.9,
  "hardware_label": "CPU (Intel(R) Core(TM) i9-10900KF CPU @ 3.70GHz)",
  "detections": [
    { "class_id": 6, "class_name": "bus",    "confidence": 0.9534,
      "box": { "x1": 9.0, "y1": 229.5, "x2": 806.8, "y2": 734.5 } },
    { "class_id": 1, "class_name": "person", "confidence": 0.9484,
      "box": { "x1": 50.6, "y1": 394.9, "x2": 245.7, "y2": 907.6 } }
  ]
}
```

The response also includes `annotated_image` — a base64 PNG data URL with boxes
and labels drawn on — `image_size`, `model`, and the full `hardware` object.

## Tests

```bash
uv run pytest -q
```

Two layers of tests:

- **API end-to-end** (`tests/test_api.py`) — loads the real model and exercises
  the HTTP API via FastAPI's `TestClient`: health, detection on a sample image,
  and rejection of non-image / empty uploads.
- **Browser end-to-end with Playwright** (`tests/test_e2e_playwright.py`) —
  spawns the real uvicorn server in a subprocess, drives a headless Chromium
  browser through the actual UI (upload → detect → read results), and asserts on
  what the user sees (hardware badge, detection count vs. table rows, annotated
  image, "Computed on" device). It also **regenerates the screenshots** below.

```bash
# One-time browser download, then run the browser E2E test:
uv run playwright install chromium
uv run pytest tests/test_e2e_playwright.py -q
```

## Docker (RHEL 9)

A [`Dockerfile`](Dockerfile) based on Red Hat's **UBI 9** Python 3.12 image is
included.

```bash
# Build (pre-downloads model weights by default; pass --build-arg PREFETCH_WEIGHTS=0 to skip)
docker build -t rf-detr-example .          # or: podman build -t rf-detr-example .

# Run (CPU)
docker run --rm -p 8011:8011 rf-detr-example

# Run with a GPU (requires the NVIDIA Container Toolkit)
docker run --rm --gpus all -p 8011:8011 rf-detr-example
# podman:  podman run --rm --device nvidia.com/gpu=all -p 8011:8011 rf-detr-example
```

Then open <http://localhost:8011>. The container reports the active device on
`/api/health` and in the UI just like the local app. The image bundles CUDA
torch (~7 GB); the same image runs on CPU or GPU.

> **podman note:** the `HEALTHCHECK` is honored only with the Docker image
> format — build with `podman build --format docker -t rf-detr-example .` if you
> want it. It is ignored (with a harmless warning) under the default OCI format.

This image has been built and validated end-to-end on RHEL 9 / UBI 9 (health +
detection inside the container, with automatic CPU fallback).

## Screenshots

| Upload screen | Result (bus) | Result (dog) |
| --- | --- | --- |
| ![home](screenshots/01-home.png) | ![bus](screenshots/02-results-bus.png) | ![dog](screenshots/03-results-dog.png) |

> The badge at the top and the "Computed on" chip make the active hardware
> explicit. In this environment the host GPU has a driver/library version
> mismatch, so inference correctly falls back to **CPU** — on a machine with a
> working CUDA GPU the same code reports and uses the **GPU** automatically.

## Notes & limitations

- This is a proof of concept: single-process, in-memory model, no auth, no batching.
- RF-DETR is COCO-pretrained (80 classes). For custom classes, fine-tune RF-DETR
  and point the loader at your weights.
- First launch downloads model weights to `~/.roboflow/models/`.

## License

Proof-of-concept code in this repo: MIT. RF-DETR and its Apache-licensed weights
are © Roboflow under their respective licenses.
