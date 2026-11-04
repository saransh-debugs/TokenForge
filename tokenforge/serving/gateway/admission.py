"""Admission control: a bounded queue in front of a bounded number of in-flight requests.

A request is shed with 429 + ``Retry-After`` when the engine's own waiting queue (scraped
from its metrics) is at the threshold or the gateway queue is full. A queued request that
cannot start before the deadline, including while the engine is cold-starting, gets 503.
"""

from __future__ import annotations

import asyncio
import time


class OverloadedError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class QueueTimeoutError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Ticket:
    """One in-flight slot. Releasing twice is a no-op."""

    def __init__(self, ctl: AdmissionController) -> None:
        self._ctl = ctl
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._ctl._release()


class AdmissionController:
    def __init__(
        self,
        max_inflight: int,
        max_queue: int,
        waiting_threshold: float,
        queue_timeout_s: float,
    ) -> None:
        self.max_inflight = max_inflight
        self.max_queue = max_queue
        self.waiting_threshold = waiting_threshold
        self.queue_timeout_s = queue_timeout_s
        self._sem = asyncio.Semaphore(max_inflight)
        self.queued = 0
        self.inflight = 0
        self.engine_waiting = 0.0
        self.engine_ready = asyncio.Event()

    def set_engine_state(self, *, ready: bool, waiting: float | None) -> None:
        if ready:
            self.engine_ready.set()
        else:
            self.engine_ready.clear()
        if waiting is not None:
            self.engine_waiting = waiting

    async def acquire(self) -> Ticket:
        if self.engine_ready.is_set() and self.engine_waiting >= self.waiting_threshold:
            raise OverloadedError("engine_queue")
        if self.queued >= self.max_queue and self._sem.locked():
            raise OverloadedError("gateway_queue")
        deadline = time.monotonic() + self.queue_timeout_s
        self.queued += 1
        try:
            try:
                await asyncio.wait_for(self._sem.acquire(), self.queue_timeout_s)
            except TimeoutError:
                raise QueueTimeoutError("queue_timeout") from None
            try:
                remaining = max(0.0, deadline - time.monotonic())
                await asyncio.wait_for(self.engine_ready.wait(), remaining)
            except TimeoutError:
                self._sem.release()
                raise QueueTimeoutError("engine_not_ready") from None
        finally:
            self.queued -= 1
        self.inflight += 1
        return Ticket(self)

    def _release(self) -> None:
        self.inflight -= 1
        self._sem.release()
