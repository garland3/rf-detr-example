"""Browser-driven end-to-end test using Playwright.

Spawns the real FastAPI/uvicorn server in a subprocess, drives a headless
Chromium browser through the actual UI (upload -> detect -> read results), and
asserts on what the user sees. As a side effect it (re)generates the screenshots
embedded in the project documentation.

Run with:
    uv run playwright install chromium    # one-time browser download
    uv run pytest tests/test_e2e_playwright.py -q

Or standalone (also refreshes screenshots/):
    uv run python tests/test_e2e_playwright.py
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "tests" / "images"
SCREENSHOTS = ROOT / "screenshots"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/api/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # server not up yet
            last_err = exc
        time.sleep(1.0)
    raise RuntimeError(f"Server did not become healthy in {timeout}s: {last_err}")


def _start_server():
    """Start uvicorn on a free port and return (process, base_url)."""
    port = _free_port()
    env = {**os.environ, "RFDETR_OPTIMIZE": "0"}
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "backend.main:app",
            "--host", "127.0.0.1", "--port", str(port),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(base_url)
    except Exception:
        proc.terminate()
        raise
    return proc, base_url


@pytest.fixture(scope="module")
def server():
    proc, base_url = _start_server()
    try:
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _set_threshold(page, value: float) -> None:
    """Set the range slider and fire the input event the UI listens for."""
    page.eval_on_selector(
        "#threshold",
        "(el, v) => { el.value = v; el.dispatchEvent(new Event('input', {bubbles: true})); }",
        value,
    )


def _detect(page, image_path: Path, threshold: float):
    """Upload an image, run detection, and return the parsed result chips."""
    page.set_input_files("#file-input", str(image_path))
    _set_threshold(page, threshold)
    page.click("#detect-btn")
    # Wait until the status line reports completion.
    page.wait_for_function(
        "() => document.getElementById('status').textContent.includes('object(s) found')",
        timeout=60_000,
    )
    page.wait_for_selector("#results:not([hidden])")


def test_e2e_browser(server):
    from playwright.sync_api import sync_playwright

    SCREENSHOTS.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 1400})
        try:
            # --- Home screen -------------------------------------------------
            page.goto(server, wait_until="networkidle")
            assert "RF-DETR" in page.title()

            # Hardware badge must resolve to a real device (not the placeholder).
            page.wait_for_function(
                "() => { const t = document.getElementById('hardware-text').textContent;"
                " return t && !t.includes('detecting'); }",
                timeout=30_000,
            )
            badge = page.inner_text("#hardware-text")
            assert ("CPU" in badge) or ("GPU" in badge), f"unexpected badge: {badge!r}"
            badge_class = page.get_attribute("#hardware-badge", "class")
            assert ("badge-cpu" in badge_class) or ("badge-gpu" in badge_class)
            page.screenshot(path=str(SCREENSHOTS / "01-home.png"), full_page=True)

            # --- Detection on the bus image ---------------------------------
            _detect(page, IMAGES / "people.jpg", 0.5)

            count = int(page.inner_text("#det-count"))
            assert count >= 1, "expected at least one detection on people.jpg"
            rows = page.locator("#det-table tbody tr").count()
            assert rows == count, f"table rows ({rows}) != count chip ({count})"

            # Annotated image is rendered as a data URL.
            img_src = page.get_attribute("#result-image", "src")
            assert img_src and img_src.startswith("data:image/png;base64,")

            # "Computed on" chip names the same device family as the badge.
            meta = page.inner_text("#result-meta")
            assert ("CPU" in meta) or ("GPU" in meta)
            assert "RFDETR" in meta  # model is reported
            assert "ms" in meta      # timing is reported

            classes = page.inner_text("#det-table").lower()
            assert "person" in classes, "expected a person detection on the bus image"

            page.screenshot(path=str(SCREENSHOTS / "02-results-bus.png"), full_page=True)

            # --- Detection on the dog image (lower threshold) ---------------
            _detect(page, IMAGES / "dog.jpg", 0.4)
            assert int(page.inner_text("#det-count")) >= 1
            page.screenshot(path=str(SCREENSHOTS / "03-results-dog.png"), full_page=True)
        finally:
            browser.close()


if __name__ == "__main__":
    # Standalone runner: start the server, run the test, refresh screenshots.
    proc, base_url = _start_server()
    try:
        class _Wrap:  # minimal shim so test_e2e_browser can take the URL
            pass

        test_e2e_browser(base_url)
        print("E2E passed. Screenshots written to", SCREENSHOTS)
    finally:
        proc.terminate()
