"""T0: the mock engine's scheduler behaves like a continuous-batching engine."""

from __future__ import annotations

import asyncio

import pytest

from tokenforge.serving.mock_engine.engine import EngineConfig, EngineMetrics, MockEngine


def _engine(**cfg: int | float) -> MockEngine:
    return MockEngine(
        EngineConfig(tpot_base_s=0.001, tpot_per_seq_s=0.0, **cfg),  # type: ignore[arg-type]
        EngineMetrics("m"),
    )


async def _drain(req) -> list[str | None]:  # type: ignore[no-untyped-def]
    finishes = []
    while True:
        item = await asyncio.wait_for(req.out.get(), 5)
        assert not isinstance(item, BaseException)
        finishes.append(item[1])
        if item[1] is not None:
            return finishes


async def test_max_num_seqs_queues_the_rest() -> None:
    eng = _engine(max_num_seqs=2)
    eng.start()
    try:
        reqs = [eng.submit(10, 50, ()) for _ in range(5)]
        await asyncio.sleep(0.02)
        assert len(eng.running) == 2
        assert len(eng.waiting) == 3
        for r in reqs:
            assert (await _drain(r))[-1] == "length"
        assert not eng.running
        assert not eng.waiting
        assert eng.kv_used() == 0
    finally:
        await eng.stop()


async def test_kv_pressure_preempts_and_still_completes() -> None:
    eng = _engine(kv_blocks=12, block_size=4, max_num_seqs=8)
    eng.start()
    try:
        reqs = [eng.submit(8, 12, ()) for _ in range(3)]
        results = await asyncio.gather(*(_drain(r) for r in reqs))
        assert all(r[-1] == "length" for r in results)
        assert eng.m.preemptions._value.get() > 0  # type: ignore[attr-defined]
        assert eng.kv_used() == 0
    finally:
        await eng.stop()


async def test_abort_frees_kv_and_slot() -> None:
    eng = _engine(max_num_seqs=1)
    eng.start()
    try:
        a = eng.submit(10, 10_000, ())
        b = eng.submit(10, 3, ())
        await asyncio.sleep(0.02)
        assert eng.running == [a]
        eng.abort(a)
        assert (await _drain(b))[-1] == "length"
        assert eng.kv_used() == 0
    finally:
        await eng.stop()


async def test_hang_fault_stalls_progress() -> None:
    eng = _engine()
    eng.start()
    try:
        eng.submit(10, 10_000, ())
        await asyncio.sleep(0.02)
        eng.faults.hang = True
        await asyncio.sleep(0.15)
        assert eng.progress_stalled(0.1)
        eng.faults.hang = False
        await asyncio.sleep(0.05)
        assert not eng.progress_stalled(0.1)
    finally:
        await eng.stop()


@pytest.mark.parametrize(("prompt", "max_tokens"), [(1, 1), (100, 7)])
async def test_usage_counts(prompt: int, max_tokens: int) -> None:
    eng = _engine()
    eng.start()
    try:
        r = eng.submit(prompt, max_tokens, ())
        await _drain(r)
        assert r.generated == max_tokens
    finally:
        await eng.stop()
