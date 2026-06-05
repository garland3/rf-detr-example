"""A minimal in-process job queue for the detection API.

Instead of running inference inline on every request (which lets many concurrent
uploads pile up and overwhelm the single model), callers can submit an image to
a bounded queue and poll for the result. One or more background worker tasks
drain the queue, running the blocking inference in a thread so the event loop
stays responsive. This is intentionally simple -- in-memory, single process --
and is meant for the API, not for durable/distributed work.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from . import inference

# Bounded queue so a flood of uploads is rejected with 503 rather than consuming
# unbounded memory. Workers process jobs serially (per worker) which naturally
# rate-limits inference.
MAX_QUEUE = int(os.environ.get("RFDETR_QUEUE_MAX", "32"))
NUM_WORKERS = max(1, int(os.environ.get("RFDETR_WORKERS", "1")))
# Cap on retained job records to bound memory; oldest finished jobs are pruned.
MAX_JOBS = int(os.environ.get("RFDETR_MAX_JOBS", "500"))

_jobs: Dict[str, Dict[str, Any]] = {}
_queue: Optional[asyncio.Queue] = None
_workers: List[asyncio.Task] = []


class QueueFull(Exception):
    """Raised when the queue is at capacity."""


def _now() -> float:
    return time.time()


def _prune() -> None:
    """Drop the oldest finished jobs once we exceed MAX_JOBS."""
    if len(_jobs) <= MAX_JOBS:
        return
    finished = sorted(
        (j for j in _jobs.values() if j["status"] in ("done", "error")),
        key=lambda j: j.get("finished_at") or 0.0,
    )
    for job in finished:
        if len(_jobs) <= MAX_JOBS:
            break
        _jobs.pop(job["id"], None)


def public_view(job: Dict[str, Any]) -> Dict[str, Any]:
    """A JSON-safe view of a job (never exposes the raw image bytes)."""
    view: Dict[str, Any] = {
        "job_id": job["id"],
        "status": job["status"],
        "mode": job["mode"],
        "threshold": job["threshold"],
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    }
    if job["status"] == "queued":
        view["queue_position"] = job.get("queue_position")
    if job["status"] == "done":
        view["result"] = job["result"]
    if job["status"] == "error":
        view["error"] = job.get("error")
    return view


async def submit(image_bytes: bytes, threshold: float, mode: str) -> Dict[str, Any]:
    """Enqueue an image for processing; returns the initial job view."""
    if _queue is None:
        raise RuntimeError("Queue not started")
    if _queue.full():
        raise QueueFull()

    job_id = uuid.uuid4().hex[:12]
    job: Dict[str, Any] = {
        "id": job_id,
        "status": "queued",
        "mode": mode,
        "threshold": threshold,
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "_image": image_bytes,
    }
    _jobs[job_id] = job
    _queue.put_nowait(job_id)
    job["queue_position"] = _queue.qsize()
    _prune()
    return public_view(job)


def get(job_id: str) -> Optional[Dict[str, Any]]:
    job = _jobs.get(job_id)
    return public_view(job) if job else None


def stats() -> Dict[str, Any]:
    qsize = _queue.qsize() if _queue is not None else 0
    statuses: Dict[str, int] = {}
    for j in _jobs.values():
        statuses[j["status"]] = statuses.get(j["status"], 0) + 1
    return {
        "queued": qsize,
        "max_queue": MAX_QUEUE,
        "workers": NUM_WORKERS,
        "jobs_by_status": statuses,
        "total_jobs": len(_jobs),
    }


def _run_inference(image_bytes: bytes, threshold: float, mode: str) -> Dict[str, Any]:
    """Blocking inference + image encoding; runs in a worker thread."""
    result = inference.detect(image_bytes, threshold=threshold, mode=mode)
    annotated = result.pop("annotated_image")
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    result["annotated_image"] = "data:image/png;base64," + base64.b64encode(
        buf.getvalue()
    ).decode("ascii")
    return result


async def _worker(worker_id: int) -> None:
    assert _queue is not None
    while True:
        job_id = await _queue.get()
        try:
            job = _jobs.get(job_id)
            if job is None:
                continue
            job["status"] = "processing"
            job["started_at"] = _now()
            image_bytes = job.pop("_image", b"")
            try:
                result = await asyncio.to_thread(
                    _run_inference, image_bytes, job["threshold"], job["mode"]
                )
                job["result"] = result
                job["status"] = "done"
            except Exception as exc:  # noqa: BLE001 - record and move on
                job["status"] = "error"
                job["error"] = str(exc)
            finally:
                job["finished_at"] = _now()
        finally:
            _queue.task_done()


async def start_workers() -> None:
    """Create the queue and spawn worker tasks (call once at startup)."""
    global _queue
    if _queue is not None:
        return
    _queue = asyncio.Queue(maxsize=MAX_QUEUE)
    for i in range(NUM_WORKERS):
        _workers.append(asyncio.create_task(_worker(i)))


async def stop_workers() -> None:
    """Cancel worker tasks (call at shutdown)."""
    for task in _workers:
        task.cancel()
    for task in _workers:
        try:
            await task
        except asyncio.CancelledError:
            pass
    _workers.clear()
