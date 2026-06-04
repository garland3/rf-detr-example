# RHEL 9 based image for the RF-DETR FastAPI proof of concept.
#
# Uses Red Hat's UBI 9 (Universal Base Image) Python 3.12 stream, which is the
# supported RHEL 9 container base. Builds a CPU/GPU-capable image; to use a GPU
# at runtime, run with the NVIDIA Container Toolkit (e.g. `--gpus all` / podman
# `--device nvidia.com/gpu=all`). With no usable GPU it falls back to CPU.
#
# Build:  docker build -t rf-detr-example .
# Run:    docker run --rm -p 8011:8011 rf-detr-example
#         (then open http://localhost:8011)
FROM registry.access.redhat.com/ubi9/python-312:latest

# --- Build as root --------------------------------------------------------
# The venv and app are installed root-owned but world-readable, so the
# unprivileged runtime user (uid 1001) can execute them without an expensive
# `chown -R` over the multi-GB torch/CUDA tree. Only small, runtime-writable
# cache dirs are handed to 1001 below.
USER 0
WORKDIR /opt/app-root/src

# OpenCV (pulled in by `supervision`) needs libGL and glib at runtime, which the
# minimal UBI image does not ship. mesa-libGL provides libGL.so.1.
RUN dnf install -y mesa-libGL glib2 && dnf clean all

# Bring in the uv binary from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install third-party deps into an in-project .venv (cached unless the lockfile
# changes). --no-dev skips pytest/playwright; --no-install-project installs only
# dependencies (the app is imported as plain modules). The uv build cache is
# removed afterwards to keep the image smaller.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_CACHE_DIR=/tmp/uv-cache
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project && rm -rf /tmp/uv-cache

# Application code (root-owned, world-readable).
COPY backend ./backend
COPY frontend ./frontend

# Runtime-writable cache dirs for the unprivileged user. RF-DETR caches weights
# under $HOME/.roboflow; redirect other library caches to writable locations too
# so nothing tries to write into the root-owned tree at runtime.
ENV HOME=/opt/app-root/src \
    XDG_CACHE_HOME=/opt/app-root/src/.rtcache \
    HF_HOME=/opt/app-root/src/.hf \
    MPLCONFIGDIR=/opt/app-root/src/.mpl \
    PATH="/opt/app-root/src/.venv/bin:${PATH}" \
    RFDETR_MODEL=nano
RUN mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR" /opt/app-root/src/.roboflow \
    && chown -R 1001:0 "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR" /opt/app-root/src/.roboflow

# --- Run as the unprivileged user -----------------------------------------
USER 1001

# Optionally pre-download the RF-DETR weights (~349 MB) at build time so the
# image is self-contained and the first request is fast. Set to 0 to skip and
# fetch weights on first run instead.
ARG PREFETCH_WEIGHTS=1
RUN if [ "$PREFETCH_WEIGHTS" = "1" ]; then \
        python -c "from backend import inference; inference.get_model()"; \
    fi

ENV HOST=0.0.0.0 \
    PORT=8011 \
    RFDETR_OPTIMIZE=1

EXPOSE 8011

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8011/api/health').getcode()==200 else 1)"

CMD ["sh", "-c", "uvicorn backend.main:app --host ${HOST} --port ${PORT}"]
