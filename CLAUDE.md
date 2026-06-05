# Project guidance for Claude

## Conventions

- **No emojis.** Do not use emojis anywhere in this repository — not in code,
  comments, UI, documentation, commit messages, or any other file. Use plain
  text (and SVG/icons where a graphic is genuinely needed). This applies to all
  contributions going forward.

## What this project is

A proof-of-concept object detection service: a FastAPI backend wrapping
Roboflow's RF-DETR, with a vanilla-JS web UI for uploading images and viewing
results. It auto-detects a usable GPU and falls back to CPU, and reports the
active hardware in both the API and the UI.

## Layout

- `backend/` — FastAPI app (`main.py`), inference wrapper (`inference.py`),
  hardware detection (`hardware.py`).
- `frontend/` — single-page UI (`index.html`, `app.js`, `style.css`), no build step.
- `tests/` — `test_api.py` (API end-to-end) and `test_e2e_playwright.py`
  (browser end-to-end; also regenerates the screenshots in `screenshots/`).
- `Dockerfile` — RHEL 9 / UBI 9 image.

## Common commands

- Install: `uv sync`
- Run: `./run.sh` (or `uv run uvicorn backend.main:app --host 127.0.0.1 --port 8011`)
- Test: `uv run pytest -q`
- Browser test / refresh screenshots: `uv run playwright install chromium` then
  `uv run pytest tests/test_e2e_playwright.py -q`

## Notes

- Use `uv` for all Python dependency and run management.
- Model size is configurable via `RFDETR_MODEL` (`nano`/`small`/`medium`/`large`).
