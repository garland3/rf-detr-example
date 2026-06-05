"""End-to-end tests for the RF-DETR FastAPI backend.

Run with:  uv run pytest -q
These tests load the real model (CPU or GPU) and exercise the HTTP API via
FastAPI's TestClient, so the first run downloads model weights (~349 MB).
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app

IMAGES = Path(__file__).parent / "images"


@pytest.fixture(scope="module")
def client():
    # `with` triggers lifespan startup (model warmup) and shutdown.
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["hardware"]["device_type"] in {"CPU", "GPU"}
    assert data["hardware_label"]
    assert data["model"].startswith("RFDETR")


def test_detect_bus(client):
    img = (IMAGES / "people.jpg").read_bytes()
    resp = client.post(
        "/api/detect",
        files={"file": ("people.jpg", img, "image/jpeg")},
        data={"threshold": "0.5"},  # mode defaults to "both"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    assert data["mode"] == "both"
    assert data["annotated_image"].startswith("data:image/png;base64,")
    assert data["hardware"]["device_type"] in {"CPU", "GPU"}
    classes = {d["class_name"] for d in data["detections"]}
    # The bus image reliably contains a bus and people.
    assert "person" in classes
    for d in data["detections"]:
        assert 0.0 <= d["confidence"] <= 1.0
        b = d["box"]
        assert b["x2"] >= b["x1"] and b["y2"] >= b["y1"]


def test_detect_modes(client):
    """box / mask / both control both the drawing and the returned fields."""
    img = (IMAGES / "people.jpg").read_bytes()

    def run(mode):
        resp = client.post(
            "/api/detect",
            files={"file": ("people.jpg", img, "image/jpeg")},
            data={"threshold": "0.5", "mode": mode},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()

    box = run("box")
    assert box["mode"] == "box"
    assert all("box" in d and "polygon" not in d for d in box["detections"])

    mask = run("mask")
    assert mask["mode"] == "mask"
    assert mask["has_masks"] is True  # segmentation model is the default
    for d in mask["detections"]:
        assert "box" not in d
        assert "mask_area" in d and d["mask_area"] >= 0
        assert d["polygon"] is None or len(d["polygon"]) >= 3

    both = run("both")
    assert both["mode"] == "both"
    assert all("box" in d and "mask_area" in d for d in both["detections"])


def test_detect_rejects_bad_mode(client):
    img = (IMAGES / "people.jpg").read_bytes()
    resp = client.post(
        "/api/detect",
        files={"file": ("people.jpg", img, "image/jpeg")},
        data={"mode": "nonsense"},
    )
    assert resp.status_code == 422


def test_queue_job_end_to_end(client):
    """Submit to the queue, poll, and get a completed result."""
    import time

    img = (IMAGES / "people.jpg").read_bytes()
    resp = client.post(
        "/api/jobs",
        files={"file": ("people.jpg", img, "image/jpeg")},
        data={"threshold": "0.5", "mode": "both"},
    )
    assert resp.status_code == 202
    job = resp.json()
    assert job["status"] == "queued"
    job_id = job["job_id"]

    data = job
    for _ in range(160):
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("done", "error"):
            break
        time.sleep(0.25)

    assert data["status"] == "done", data
    result = data["result"]
    assert result["count"] >= 1
    assert result["mode"] == "both"
    assert result["annotated_image"].startswith("data:image/png;base64,")


def test_queue_unknown_job(client):
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_queue_stats(client):
    r = client.get("/api/queue")
    assert r.status_code == 200
    s = r.json()
    assert "queued" in s and "workers" in s and "max_queue" in s


def test_detect_rejects_non_image(client):
    resp = client.post(
        "/api/detect",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 415


def test_detect_rejects_empty(client):
    resp = client.post(
        "/api/detect",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert resp.status_code == 400
