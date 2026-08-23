"""Small non-blocking queue operations shared by runtime workers."""

from __future__ import annotations

from queue import Empty, Queue
from typing import TypeVar


Item = TypeVar("Item")


def discard_queued(target: Queue[Item]) -> Item | None:
    """Discard and return one queued item, or ``None`` when empty."""
    try:
        return target.get_nowait()
    except Empty:
        return None


def replace_latest(target: Queue[Item], item: Item) -> Item | None:
    """Replace the pending item in a latest-only queue."""
    displaced = discard_queued(target)
    target.put_nowait(item)
    return displaced


def poll_latest(target: Queue[Item]) -> Item | None:
    """Drain a queue and return its newest pending item."""
    latest = None
    while True:
        try:
            latest = target.get_nowait()
        except Empty:
            return latest


def drain_queue(target: Queue[object]) -> int:
    """Remove all pending items without waiting and return their count."""
    drained = 0
    while True:
        try:
            target.get_nowait()
            drained += 1
        except Empty:
            return drained
