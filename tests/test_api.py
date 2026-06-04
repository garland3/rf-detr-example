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
        data={"threshold": "0.5"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    assert data["annotated_image"].startswith("data:image/png;base64,")
    assert data["hardware"]["device_type"] in {"CPU", "GPU"}
    classes = {d["class_name"] for d in data["detections"]}
    # The bus image reliably contains a bus and people.
    assert "person" in classes
    for d in data["detections"]:
        assert 0.0 <= d["confidence"] <= 1.0
        b = d["box"]
        assert b["x2"] >= b["x1"] and b["y2"] >= b["y1"]


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
