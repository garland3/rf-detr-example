"""Hardware detection helpers.

Determines whether a CUDA GPU is actually usable by torch (not just present in
the system) and produces a human-readable description that the API and UI can
surface so it is always clear *what* ran the inference.
"""
from __future__ import annotations

import platform
from functools import lru_cache
from typing import Any, Dict


@lru_cache(maxsize=1)
def _cpu_name() -> str:
    """Best-effort human-readable CPU model name."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "Unknown CPU"


@lru_cache(maxsize=1)
def detect_device() -> Dict[str, Any]:
    """Detect the device torch will actually use.

    Returns a dict describing the selected device. ``torch.cuda.is_available()``
    returns ``False`` (rather than raising) when a GPU is present but unusable,
    e.g. a driver/library version mismatch, so CPU fallback is automatic.
    """
    info: Dict[str, Any] = {
        "device": "cpu",
        "device_type": "CPU",
        "name": _cpu_name(),
        "cuda_available": False,
        "details": {},
    }

    try:
        import torch
    except Exception as exc:  # torch missing/broken -> CPU
        info["details"]["torch_error"] = str(exc)
        return info

    info["details"]["torch_version"] = torch.__version__

    cuda_available = False
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        info["details"]["cuda_error"] = str(exc)

    info["cuda_available"] = cuda_available

    if cuda_available:
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info.update(
                {
                    "device": "cuda",
                    "device_type": "GPU",
                    "name": torch.cuda.get_device_name(idx),
                }
            )
            info["details"].update(
                {
                    "cuda_version": getattr(torch.version, "cuda", None),
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        except Exception as exc:  # detected but unusable -> stay on CPU
            info["details"]["gpu_init_error"] = str(exc)
            info["cuda_available"] = False

    return info


def device_string() -> str:
    """Return 'cuda' or 'cpu' for passing to the model."""
    return detect_device()["device"]


def device_label() -> str:
    """Short label like 'GPU (NVIDIA ...)' or 'CPU (Intel ...)'."""
    info = detect_device()
    return f"{info['device_type']} ({info['name']})"
