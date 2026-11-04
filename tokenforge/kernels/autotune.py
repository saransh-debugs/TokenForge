"""Persistent autotune cache for the W4A16 Triton kernel.

Tuning runs eagerly at weight-load time (``process_weights_after_loading``), before vLLM
captures CUDA graphs. The hot path only does a dict lookup and falls back to the heuristic
:func:`default_config` on a miss, so a call never benchmarks, allocates for tuning, or syncs.

The cache lives at ``$TOKENFORGE_CACHE_DIR`` (default ``~/.cache/tokenforge``) as
``<gpu>-triton<version>.json``. Writes are atomic (tmp -> fsync -> rename -> fsync dir).
Anything unexpected on read (bad JSON, wrong schema, another GPU or Triton version, an
invalid config) discards the whole file with a warning. A cache is never partially trusted.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Final

import torch

from tokenforge.kernels.config import M_BUCKETS, GemmConfig, candidate_configs, default_config

log = logging.getLogger(__name__)

SCHEMA_VERSION: Final = 1
CACHE_DIR_ENV: Final = "TOKENFORGE_CACHE_DIR"


def triton_version() -> str:
    import triton

    return str(triton.__version__)


def cache_dir() -> Path:
    env = os.environ.get(CACHE_DIR_ENV)
    return Path(env) if env else Path.home() / ".cache" / "tokenforge"


def cache_path(gpu: str, triton_ver: str) -> Path:
    return cache_dir() / f"{gpu}-triton{triton_ver}.json"


def entry_key(bucket: int, n: int, k: int, group_size: int) -> str:
    return f"{bucket}:{n}:{k}:{group_size}"


class CacheRejectedError(ValueError):
    """The on-disk cache failed validation."""


def _parse_entries(raw: Any) -> dict[str, GemmConfig]:
    if not isinstance(raw, dict):
        raise CacheRejectedError("'entries' is not an object")
    entries: dict[str, GemmConfig] = {}
    for key, value in raw.items():
        parts = key.split(":") if isinstance(key, str) else []
        if len(parts) != 4 or not all(p.isdigit() for p in parts):
            raise CacheRejectedError(f"malformed key {key!r}")
        bucket, n, k, group = (int(p) for p in parts)
        if bucket not in M_BUCKETS or n <= 0 or k <= 0 or k % group:
            raise CacheRejectedError(f"impossible key {key!r}")
        try:
            cfg = GemmConfig.from_dict(value)
            cfg.validate(group)
        except (TypeError, ValueError) as exc:
            raise CacheRejectedError(f"invalid config for {key!r}: {exc}") from exc
        if cfg.block_m > max(bucket, 16) or (bucket < 16 and cfg.block_m not in (1, bucket)):
            raise CacheRejectedError(f"block_m={cfg.block_m} does not fit bucket {bucket}")
        entries[key] = cfg
    return entries


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


class AutotuneCache:
    def __init__(self, path: Path, gpu: str, triton_ver: str) -> None:
        self.path = path
        self.gpu = gpu
        self.triton_ver = triton_ver
        self._entries: dict[str, GemmConfig] = {}
        self._lock = threading.Lock()
        self.discarded_reason: str | None = None

    def load(self) -> None:
        """Populate from disk. Missing file: empty cache. Invalid file: discard and warn."""
        self.discarded_reason = None
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._entries = {}
            return
        except OSError as exc:
            self._discard(f"unreadable: {exc}")
            return
        try:
            doc = json.loads(text)
            if not isinstance(doc, dict):
                raise CacheRejectedError("top level is not an object")
            if doc.get("schema") != SCHEMA_VERSION:
                raise CacheRejectedError(f"schema {doc.get('schema')!r} != {SCHEMA_VERSION}")
            if doc.get("gpu") != self.gpu:
                raise CacheRejectedError(
                    f"built for gpu {doc.get('gpu')!r}, running on {self.gpu!r}"
                )
            if doc.get("triton") != self.triton_ver:
                raise CacheRejectedError(
                    f"built for triton {doc.get('triton')!r}, running {self.triton_ver!r}"
                )
            entries = _parse_entries(doc.get("entries"))
        except (json.JSONDecodeError, UnicodeDecodeError, CacheRejectedError) as exc:
            self._discard(str(exc))
            return
        with self._lock:
            self._entries = entries

    def _discard(self, reason: str) -> None:
        log.warning("autotune cache %s discarded (%s); configs will be retuned", self.path, reason)
        self.discarded_reason = reason
        with self._lock:
            self._entries = {}

    def get(self, bucket: int, n: int, k: int, group_size: int) -> GemmConfig | None:
        return self._entries.get(entry_key(bucket, n, k, group_size))

    def put(self, bucket: int, n: int, k: int, group_size: int, cfg: GemmConfig) -> None:
        cfg.validate(group_size)
        with self._lock:
            self._entries[entry_key(bucket, n, k, group_size)] = cfg

    def __len__(self) -> int:
        return len(self._entries)

    def save(self) -> None:
        with self._lock:
            payload = {
                "schema": SCHEMA_VERSION,
                "gpu": self.gpu,
                "triton": self.triton_ver,
                "torch": torch.__version__,
                "entries": {k: v.to_dict() for k, v in sorted(self._entries.items())},
            }
        atomic_write_json(self.path, payload)


_CACHES: dict[tuple[str, str], AutotuneCache] = {}
_CACHES_LOCK = threading.Lock()


def get_cache(gpu: str) -> AutotuneCache:
    ver = triton_version()
    with _CACHES_LOCK:
        cache = _CACHES.get((gpu, ver))
        if cache is None:
            cache = AutotuneCache(cache_path(gpu, ver), gpu, ver)
            cache.load()
            _CACHES[(gpu, ver)] = cache
        return cache


def reset_caches() -> None:
    with _CACHES_LOCK:
        _CACHES.clear()


def lookup(gpu: str, bucket: int, n: int, k: int, group_size: int) -> GemmConfig:
    """Hot-path lookup: tuned config if cached, otherwise the heuristic default."""
    cfg = get_cache(gpu).get(bucket, n, k, group_size)
    return cfg if cfg is not None else default_config(bucket, n, k, group_size)


BenchFn = Callable[[Callable[[], object]], float]


def cuda_bench(fn: Callable[[], object]) -> float:
    """Median runtime in ms with an L2 flush between iterations."""
    import triton.testing

    return float(triton.testing.do_bench(fn, warmup=10, rep=40, return_mode="median"))


def tune_layer(
    n: int,
    k: int,
    group_size: int,
    *,
    device: torch.device,
    gpu: str,
    buckets: Iterable[int] = M_BUCKETS,
    bench_fn: BenchFn | None = None,
    candidates: Callable[[int, int], list[GemmConfig]] = candidate_configs,
    save: bool = True,
) -> dict[int, GemmConfig]:
    """Tune every M bucket for one (N, K) layer shape and persist the winners.

    Each candidate is checked against the default config's output before it can win, so a
    config that miscompiles is never cached. Buckets already in the cache are skipped.
    """
    from tokenforge.kernels.pack import awq_quantize, pack_awq
    from tokenforge.kernels.triton.w4a16 import w4a16_gemm

    if bench_fn is None:
        if device.type != "cuda":
            raise RuntimeError("tuning needs a CUDA device or an explicit bench_fn")
        bench_fn = cuda_bench

    cache = get_cache(gpu)
    gen = torch.Generator(device="cpu").manual_seed(0)
    weight = torch.randn((n, k), generator=gen, dtype=torch.float32) * 0.02
    w = pack_awq(*awq_quantize(weight, group_size), group_size)
    qweight_t, scales, qzeros = (t.to(device) for t in (w.qweight_t, w.scales, w.qzeros))

    winners: dict[int, GemmConfig] = {}
    for bucket in buckets:
        cached = cache.get(bucket, n, k, group_size)
        if cached is not None:
            winners[bucket] = cached
            continue
        x = (torch.randn((bucket, k), generator=gen) * 0.5).to(device=device, dtype=torch.float16)
        base_cfg = default_config(bucket, n, k, group_size)
        ref = w4a16_gemm(x, qweight_t, scales, qzeros, group_size, base_cfg).float()
        tol = 1e-2 * float(ref.abs().max()) + 1e-3
        best: tuple[float, GemmConfig] | None = None
        for cfg in candidates(bucket, group_size):
            try:
                out = w4a16_gemm(x, qweight_t, scales, qzeros, group_size, cfg)
                if float((out.float() - ref).abs().max()) > tol:
                    log.warning("config %s miscomputes for %s; skipped", cfg, (bucket, n, k))
                    continue
                t = bench_fn(
                    functools.partial(w4a16_gemm, x, qweight_t, scales, qzeros, group_size, cfg)
                )
            except Exception as exc:  # out of shared memory / registers
                log.debug("config %s failed to run: %s", cfg, exc)
                continue
            if best is None or t < best[0]:
                best = (t, cfg)
        chosen = best[1] if best is not None else base_cfg
        cache.put(bucket, n, k, group_size, chosen)
        winners[bucket] = chosen
    if save:
        cache.save()
    return winners
