"""Unit tests for the bounded-concurrency processing queue.

Verifies the two properties that matter for an 8-doc burst on a small GPU:
  1. At most ``max_concurrent`` documents are processed at once.
  2. The rest wait FIFO and report an accurate queue position.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from src.api.routers.upload import ProcessingQueue, task_store


def _seed_task(task_id: str) -> None:
    task_store[task_id] = {"task_id": task_id, "filename": f"{task_id}.pdf", "file_path": ""}


async def _stop_workers(queue: ProcessingQueue) -> None:
    """Cancel the queue's worker tasks so they don't leak into the test loop."""
    for worker in queue._workers:
        worker.cancel()
    await asyncio.gather(*queue._workers, return_exceptions=True)


async def test_queue_limits_concurrency_and_is_fifo() -> None:
    """max_concurrent=2 → exactly 2 run at once, completion is FIFO."""
    started: list[str] = []
    finished: list[str] = []

    async def fake_process(task_id: str, file_path: str) -> None:
        started.append(task_id)
        await asyncio.sleep(0.05)  # make each task "take a while"
        finished.append(task_id)

    try:
        with patch("src.api.routers.upload._process_document", fake_process):
            queue = ProcessingQueue(max_concurrent=2)
            for tid in ("a", "b", "c", "d"):
                _seed_task(tid)
                queue.submit(tid)

            # Let workers pick up the first batch
            await asyncio.sleep(0.02)
            assert queue.running_count == 2, f"expected 2 running, got {queue.running_count}"
            assert queue.waiting_count == 2

            # Running tasks have no position; waiting tasks report FIFO position
            assert queue.is_processing("a")
            assert queue.position("a") is None
            assert queue.position("c") == 1
            assert queue.position("d") == 2

            # Let everything drain
            while queue.waiting_count or queue.running_count:
                await asyncio.sleep(0.02)

            # FIFO guarantees START order (which task grabs a free slot), not
            # finish order — concurrent tasks finish based on their own duration.
            assert started == ["a", "b", "c", "d"], f"start order {started}"
            assert sorted(finished) == ["a", "b", "c", "d"], f"finish set {finished}"
            # Bounded concurrency: c could not start until a's slot freed up
            assert started.index("c") > finished.index("a")
    finally:
        await _stop_workers(queue)
        task_store.clear()


async def test_queue_position_reports_none_when_not_waiting() -> None:
    """A task that is neither running nor waiting reports no position."""
    async def _noop(task_id: str, file_path: str) -> None:
        pass

    try:
        with patch("src.api.routers.upload._process_document", _noop):
            queue = ProcessingQueue(max_concurrent=1)
            _seed_task("x")
            queue.submit("x")
            await asyncio.sleep(0.02)  # worker grabs it and clears the waiting line
            assert queue.position("x") is None
            assert queue.position("never-submitted") is None
            assert queue.waiting_count == 0
    finally:
        await _stop_workers(queue)
        task_store.clear()
