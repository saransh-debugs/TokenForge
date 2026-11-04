"""T0 + F4: the autotune cache is written atomically and never trusted when corrupt."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch

from tokenforge.kernels import autotune
from tokenforge.kernels.config import GemmConfig, default_config
from tokenforge.kernels.pack import awq_quantize, pack_awq

GPU = "test-gpu-sm89"
N, K, G = 64, 256, 128
TUNED_1 = GemmConfig(block_m=1, block_n=32, block_k=128, split_k=2, num_warps=2, num_stages=2)
TUNED_16 = GemmConfig(block_m=16, block_n=32, block_k=64, split_k=1, num_warps=4, num_stages=2)


def _fresh_cache() -> autotune.AutotuneCache:
    autotune.reset_caches()
    return autotune.get_cache(GPU)


def _populated(path_cache: autotune.AutotuneCache) -> None:
    path_cache.put(1, N, K, G, TUNED_1)
    path_cache.put(16, N, K, G, TUNED_16)
    path_cache.save()


def test_roundtrip_and_atomic_write() -> None:
    cache = _fresh_cache()
    assert len(cache) == 0
    _populated(cache)
    files = sorted(p.name for p in cache.path.parent.iterdir())
    assert files == [cache.path.name], "temporary files must not survive a save"
    doc = json.loads(cache.path.read_text())
    assert doc["schema"] == autotune.SCHEMA_VERSION
    assert doc["gpu"] == GPU
    reloaded = _fresh_cache()
    assert reloaded.discarded_reason is None
    assert reloaded.get(1, N, K, G) == TUNED_1
    assert autotune.lookup(GPU, 16, N, K, G) == TUNED_16
    assert autotune.lookup(GPU, 32, N, K, G) == default_config(32, N, K, G)


def test_cache_path_is_keyed_by_gpu_and_triton() -> None:
    p = autotune.cache_path(GPU, "3.8.0")
    assert p.name == f"{GPU}-triton3.8.0.json"
    assert p.parent == autotune.cache_dir()


def _corruptions(path: Path, cache: autotune.AutotuneCache) -> dict[str, str]:
    good = json.loads(path.read_text())
    entry_key = autotune.entry_key(1, N, K, G)

    def with_entry(cfg: object) -> str:
        return json.dumps({**good, "entries": {entry_key: cfg}})

    text = path.read_text()
    return {
        "truncated": text[: len(text) // 2],
        "garbled": "\x00\xff" + text[5:],
        "empty": "",
        "not_object": "[1, 2, 3]",
        "wrong_schema": json.dumps({**good, "schema": 999}),
        "wrong_gpu": json.dumps({**good, "gpu": "nvidia-tesla-t4-sm75"}),
        "wrong_triton": json.dumps({**good, "triton": "0.0.1"}),
        "bad_key": json.dumps({**good, "entries": {"1:x:3:4": TUNED_1.to_dict()}}),
        "bad_block_k": with_entry({**TUNED_1.to_dict(), "block_k": 96}),
        "unknown_field": with_entry({**TUNED_1.to_dict(), "evil": 1}),
        "bool_value": with_entry({**TUNED_1.to_dict(), "split_k": True}),
        "block_m_too_big": with_entry({**TUNED_1.to_dict(), "block_m": 64}),
    }


@pytest.mark.parametrize(
    "kind",
    [
        "truncated",
        "garbled",
        "empty",
        "not_object",
        "wrong_schema",
        "wrong_gpu",
        "wrong_triton",
        "bad_key",
        "bad_block_k",
        "unknown_field",
        "bool_value",
        "block_m_too_big",
    ],
)
def test_f4_corrupt_cache_is_discarded(kind: str, caplog: pytest.LogCaptureFixture) -> None:
    cache = _fresh_cache()
    _populated(cache)
    cache.path.write_bytes(_corruptions(cache.path, cache)[kind].encode("utf-8", "surrogatepass"))
    with caplog.at_level(logging.WARNING, logger="tokenforge.kernels.autotune"):
        reloaded = _fresh_cache()
    assert reloaded.discarded_reason is not None
    assert len(reloaded) == 0, "a corrupt cache must be discarded whole"
    assert any("discarded" in r.getMessage() for r in caplog.records)
    assert autotune.lookup(GPU, 1, N, K, G) == default_config(1, N, K, G)


def _fake_bench_prefers(winner: GemmConfig):  # type: ignore[no-untyped-def]
    """Deterministic stand-in for CUDA-event timing: ``winner`` is fastest."""
    calls: list[object] = []

    def bench(fn):  # type: ignore[no-untyped-def]
        fn()
        calls.append(fn)
        return 1.0 if fn.args[-1] == winner else 2.0

    return bench, calls


def test_f4_retune_after_corruption_restores_outputs(caplog: pytest.LogCaptureFixture) -> None:
    """Corrupt -> discard + warn -> retune -> same winner -> bit-identical kernel output."""
    from tokenforge.kernels.triton.w4a16 import w4a16_gemm

    candidates = [TUNED_1, GemmConfig(1, 64, 128, 1, 4, 2)]
    bench, _ = _fake_bench_prefers(TUNED_1)
    cache = _fresh_cache()
    winners = autotune.tune_layer(
        N,
        K,
        G,
        device=torch.device("cpu"),
        gpu=GPU,
        buckets=(1,),
        bench_fn=bench,
        candidates=lambda b, g: candidates,
    )
    assert winners == {1: TUNED_1}

    gen = torch.Generator().manual_seed(0)
    w = pack_awq(*awq_quantize(torch.randn((N, K), generator=gen) * 0.02, G), G)
    x = torch.randn((1, K), generator=gen).to(torch.float16)
    before = w4a16_gemm(x, w.qweight_t, w.scales, w.qzeros, G, autotune.lookup(GPU, 1, N, K, G))

    cache.path.write_text(cache.path.read_text()[:17])  # truncate
    with caplog.at_level(logging.WARNING):
        corrupt = _fresh_cache()
    assert corrupt.discarded_reason is not None
    assert "discarded" in caplog.text

    bench2, calls = _fake_bench_prefers(TUNED_1)
    autotune.tune_layer(
        N,
        K,
        G,
        device=torch.device("cpu"),
        gpu=GPU,
        buckets=(1,),
        bench_fn=bench2,
        candidates=lambda b, g: candidates,
    )
    assert len(calls) == len(candidates), "every candidate is re-benchmarked after a discard"
    assert _fresh_cache().get(1, N, K, G) == TUNED_1
    after = w4a16_gemm(x, w.qweight_t, w.scales, w.qzeros, G, autotune.lookup(GPU, 1, N, K, G))
    assert torch.equal(before, after)


def test_tune_skips_cached_buckets_and_needs_bench_off_gpu() -> None:
    cache = _fresh_cache()
    _populated(cache)
    bench, calls = _fake_bench_prefers(TUNED_1)
    winners = autotune.tune_layer(
        N,
        K,
        G,
        device=torch.device("cpu"),
        gpu=GPU,
        buckets=(1, 16),
        bench_fn=bench,
        candidates=lambda b, g: [TUNED_1],
    )
    assert winners == {1: TUNED_1, 16: TUNED_16}
    assert calls == []
    with pytest.raises(RuntimeError, match="CUDA"):
        autotune.tune_layer(N, K, G, device=torch.device("cpu"), gpu=GPU, buckets=(2,))


def test_save_failure_leaves_previous_file(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _fresh_cache()
    _populated(cache)
    original = cache.path.read_text()

    def boom(*_a, **_k):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(autotune.os, "replace", boom)
    cache.put(2, N, K, G, GemmConfig(2, 32, 64))
    with pytest.raises(OSError, match="disk full"):
        cache.save()
    assert cache.path.read_text() == original
    assert sorted(p.name for p in cache.path.parent.iterdir()) == [cache.path.name]
