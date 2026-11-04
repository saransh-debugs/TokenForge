"""A continuous-batching engine simulator with vLLM's API and metrics surface.

It reproduces what the gateway, autoscaler and chaos suite observe from a real engine:
queueing behind ``max_num_seqs``, KV-cache pressure with preemption, TTFT growing with
prompt length, TPOT growing with batch size, aborts on disconnect, and injectable faults.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
import zlib
from collections import deque
from dataclasses import dataclass, field

WORDS = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "that",
    "for",
    "it",
    "as",
    "with",
    "was",
    "on",
    "be",
    "by",
    "this",
    "are",
    "or",
    "from",
    "at",
    "an",
    "but",
    "not",
    "have",
    "which",
    "they",
    "you",
    "were",
    "all",
    "one",
    "their",
    "there",
    "been",
    "has",
    "more",
    "if",
    "will",
    "can",
    "when",
    "who",
    "no",
    "may",
    "into",
    "kernel",
    "tensor",
    "cache",
    "token",
    "graph",
    "batch",
    "queue",
    "memory",
    "latency",
    "throughput",
]


@dataclass
class EngineConfig:
    max_num_seqs: int = 64
    kv_blocks: int = 4096
    block_size: int = 16
    max_model_len: int = 8192
    prefill_s_per_token: float = 2e-5
    tpot_base_s: float = 0.010
    tpot_per_seq_s: float = 2e-4


@dataclass
class Faults:
    hang: bool = False
    fail_next: int = 0
    fail_status: int = 503
    crash_after_tokens: int | None = None
    first_token_delay_s: float = 0.0


@dataclass
class Request:
    rid: str
    prompt_tokens: int
    max_tokens: int
    seed: int
    stop: tuple[str, ...] = ()
    out: asyncio.Queue[tuple[str, str | None] | BaseException] = field(
        default_factory=asyncio.Queue
    )
    generated: int = 0
    text: str = ""
    needs_prefill: bool = True
    arrival: float = field(default_factory=time.monotonic)
    first_token_at: float | None = None
    last_token_at: float | None = None
    done: bool = False

    def blocks(self, block_size: int) -> int:
        return -(-(self.prompt_tokens + self.generated + 1) // block_size)


def _find_stop(text: str, stops: tuple[str, ...], prev_len: int) -> int | None:
    """Earliest stop-sequence start that ends in the newly appended text, if any."""
    best = None
    for s in stops:
        idx = text.find(s, max(0, prev_len - len(s) + 1))
        if idx != -1 and (best is None or idx < best):
            best = idx
    return best


class EngineCrashedError(RuntimeError):
    """Raised into every active stream when the crash fault fires."""


class MockEngine:
    def __init__(self, config: EngineConfig, metrics: EngineMetrics) -> None:
        self.cfg = config
        self.m = metrics
        self.faults = Faults()
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._ids = itertools.count()
        self.total_generated = 0
        self.last_progress = time.monotonic()
        self.crashed = False

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    # ---------------------------------------------------------------- API
    def submit(self, prompt_tokens: int, max_tokens: int, stop: tuple[str, ...]) -> Request:
        rid = f"cmpl-{next(self._ids)}"
        req = Request(rid, prompt_tokens, max_tokens, seed=zlib.crc32(rid.encode()), stop=stop)
        self.waiting.append(req)
        self.m.prompt_tokens.inc(prompt_tokens)
        self._publish()
        self._wake.set()
        return req

    def abort(self, req: Request) -> None:
        if req.done:
            return
        req.done = True
        if req in self.running:
            self.running.remove(req)
        elif req in self.waiting:
            self.waiting.remove(req)
        self.m.aborts.inc()
        self._publish()

    def _hung(self) -> bool:
        return self.faults.hang

    def kv_used(self) -> int:
        return sum(r.blocks(self.cfg.block_size) for r in self.running)

    def progress_stalled(self, stall_s: float) -> bool:
        busy = bool(self.running or self.waiting)
        return busy and time.monotonic() - self.last_progress > stall_s

    # ---------------------------------------------------------------- scheduler
    def _publish(self) -> None:
        self.m.running.set(len(self.running))
        self.m.waiting.set(len(self.waiting))
        self.m.kv_usage.set(self.kv_used() / self.cfg.kv_blocks)

    def _admit(self) -> list[Request]:
        admitted = []
        used = self.kv_used()
        while self.waiting and len(self.running) < self.cfg.max_num_seqs:
            nxt = self.waiting[0]
            need = nxt.blocks(self.cfg.block_size)
            if used + need > self.cfg.kv_blocks:
                break
            self.waiting.popleft()
            self.running.append(nxt)
            admitted.append(nxt)
            used += need
        return admitted

    def _preempt_for_growth(self) -> None:
        """vLLM-style recompute preemption: evict the newest request until the step fits."""
        while self.running:
            growth = sum(
                1
                for r in self.running
                if (r.prompt_tokens + r.generated) % self.cfg.block_size == 0
            )
            if self.kv_used() + growth <= self.cfg.kv_blocks:
                return
            victim = self.running.pop()
            victim.needs_prefill = True
            self.waiting.appendleft(victim)
            self.m.preemptions.inc()

    def _next_piece(self, req: Request) -> str:
        return WORDS[(req.seed + req.generated * 2654435761) % len(WORDS)]

    async def _loop(self) -> None:
        while True:
            if self.faults.hang:
                await asyncio.sleep(0.05)
                continue
            admitted = self._admit()
            self._publish()
            if not self.running:
                self._wake.clear()
                await self._wake.wait()
                continue
            prefill = [r for r in self.running if r.needs_prefill]
            step = self.cfg.tpot_base_s + self.cfg.tpot_per_seq_s * len(self.running)
            step += sum(
                (r.prompt_tokens + r.generated) * self.cfg.prefill_s_per_token for r in prefill
            )
            if admitted and self.faults.first_token_delay_s:
                step += self.faults.first_token_delay_s
            await asyncio.sleep(step)
            if self._hung():  # the fault may have been set while this step slept
                continue
            self._preempt_for_growth()
            now = time.monotonic()
            for req in list(self.running):
                req.needs_prefill = False
                piece = self._next_piece(req) + " "
                req.generated += 1
                self.total_generated += 1
                self.m.generation_tokens.inc()
                if req.first_token_at is None:
                    req.first_token_at = now
                    self.m.ttft.observe(now - req.arrival)
                elif req.last_token_at is not None:
                    self.m.tpot.observe(now - req.last_token_at)
                req.last_token_at = now
                finish = None
                stop_at = _find_stop(req.text + piece, req.stop, len(req.text))
                if stop_at is not None:
                    piece = (req.text + piece)[len(req.text) : stop_at]
                    finish = "stop"
                elif req.generated >= req.max_tokens:
                    finish = "length"
                req.text += piece
                req.out.put_nowait((piece, finish))
                if finish is not None:
                    req.done = True
                    self.running.remove(req)
                    self.m.e2e.observe(now - req.arrival)
            self.last_progress = now
            crash_at = self.faults.crash_after_tokens
            if crash_at is not None and self.total_generated >= crash_at:
                self._crash()
                return
            self._publish()

    def _crash(self) -> None:
        self.crashed = True
        for req in [*self.running, *self.waiting]:
            req.done = True
            req.out.put_nowait(EngineCrashedError("engine process died"))
        self.running.clear()
        self.waiting.clear()


class EngineMetrics:
    """vLLM v1 metric names, on a private registry so several engines can share a process."""

    def __init__(self, model: str) -> None:
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

        self.registry = CollectorRegistry()
        lbl = {"model_name": model}

        def gauge(name: str, doc: str) -> Gauge:
            return Gauge(name, doc, list(lbl), registry=self.registry)

        def counter(name: str, doc: str) -> Counter:
            return Counter(name, doc, list(lbl), registry=self.registry)

        def hist(name: str, doc: str, buckets: tuple[float, ...]) -> Histogram:
            return Histogram(name, doc, list(lbl), buckets=buckets, registry=self.registry)

        lat = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
        self.running = gauge("vllm:num_requests_running", "Requests running").labels(**lbl)
        self.waiting = gauge("vllm:num_requests_waiting", "Requests waiting").labels(**lbl)
        self.kv_usage = gauge("vllm:kv_cache_usage_perc", "KV cache usage (0-1)").labels(**lbl)
        self.preemptions = counter("vllm:num_preemptions", "Preemptions").labels(**lbl)
        self.aborts = counter("vllm:request_abort", "Aborted requests").labels(**lbl)
        self.prompt_tokens = counter("vllm:prompt_tokens", "Prefill tokens").labels(**lbl)
        self.generation_tokens = counter("vllm:generation_tokens", "Generated tokens").labels(**lbl)
        self.ttft = hist("vllm:time_to_first_token_seconds", "TTFT", lat).labels(**lbl)
        self.tpot = hist("vllm:inter_token_latency_seconds", "Inter-token latency", lat).labels(
            **lbl
        )
        self.e2e = hist("vllm:e2e_request_latency_seconds", "End-to-end latency", lat).labels(**lbl)
