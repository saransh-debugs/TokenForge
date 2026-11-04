"""Kernel microbenchmarks with a regression gate.

    python -m tokenforge.bench.micro [--suite full] [--check] [--update-baseline --reason ...]

Method: CUDA events around each launch, 25 warmup iterations, median of 200, L2 flushed
between iterations so cold-cache numbers are what is measured. Each row records achieved
DRAM bandwidth (bytes actually moved: packed weights, scales, zeros, activations, output)
and FLOP/s, plus the environment fingerprint. ``--check`` fails if any shape is > 5% slower
than ``bench/results/<gpu>/micro_baseline.json``.
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from tokenforge.bench.env import ROOT, fingerprint
from tokenforge.kernels.device import gpu_identity

WARMUP = 25
ITERS = 200
REGRESSION = 0.05
PEAK_GBPS = {"nvidia-l4-sm89": 300.0, "tesla-t4-sm75": 320.0, "nvidia-tesla-t4-sm75": 320.0}


@dataclass(frozen=True)
class Case:
    layer: str
    m: int
    n: int
    k: int


def suite(name: str) -> list[Case]:
    shapes = {
        "qwen2.5-0.5b": [
            ("qkv_proj", 1152, 896),
            ("o_proj", 896, 896),
            ("gate_up_proj", 9728, 896),
            ("down_proj", 896, 4864),
        ],
        "qwen2.5-7b": [
            ("qkv_proj", 4608, 3584),
            ("o_proj", 3584, 3584),
            ("gate_up_proj", 37888, 3584),
            ("down_proj", 3584, 18944),
        ],
    }
    models = ["qwen2.5-7b"] if name == "l4" else ["qwen2.5-0.5b", "qwen2.5-7b"]
    ms = (1, 8, 32) if name in ("l4", "t4") else (1, 2, 4, 8, 16, 32, 64, 128)
    return [
        Case(f"{model}/{layer}", m, n, k)
        for model in models
        for layer, n, k in shapes[model]
        for m in ms
    ]


def time_cuda(fn: Callable[[], object], flush: torch.Tensor) -> float:
    """Median milliseconds over ITERS, L2 flushed before each timed launch."""
    for _ in range(WARMUP):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]  # type: ignore[no-untyped-call]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]  # type: ignore[no-untyped-call]
    for s, e in zip(starts, ends, strict=True):
        flush.zero_()
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return float(statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)))


def bytes_moved(c: Case, group: int = 128) -> int:
    weights = c.n * c.k // 2
    scales = (c.k // group) * c.n * 2
    zeros = (c.k // group) * c.n // 2
    return weights + scales + zeros + c.m * c.k * 2 + c.m * c.n * 2


def run(cases: list[Case], tune: bool = True) -> list[dict[str, Any]]:
    import tokenforge.kernels.ops  # noqa: F401
    from tokenforge.kernels import autotune
    from tokenforge.kernels.pack import awq_quantize, pack_awq

    dev = torch.device("cuda")
    gpu = gpu_identity(dev)
    if tune:  # same as process_weights_after_loading: tune before anything is timed
        for n, k in sorted({(c.n, c.k) for c in cases}):
            autotune.tune_layer(n, k, 128, device=dev, gpu=gpu)
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device=dev)  # 256 MB > L2
    fp = fingerprint()
    rows = []
    cache: dict[tuple[int, int], Any] = {}
    for c in cases:
        if (c.n, c.k) not in cache:
            w = pack_awq(*awq_quantize(torch.randn(c.n, c.k) * 0.02, 128), 128)
            cache[(c.n, c.k)] = (w.qweight_t.to(dev), w.scales.to(dev), w.qzeros.to(dev))
        qw, s, z = cache[(c.n, c.k)]
        x = torch.randn(c.m, c.k, device=dev, dtype=torch.float16)
        dense = torch.randn(c.n, c.k, device=dev, dtype=torch.float16) * 0.02
        impls: dict[str, Callable[[], object]] = {
            "tokenforge-triton": functools.partial(
                torch.ops.tokenforge.w4a16_linear, x, qw, s, z, None, 128, "triton"
            ),
            "cublas-fp16": functools.partial(torch.nn.functional.linear, x, dense),
        }
        for impl, fn in impls.items():
            ms = time_cuda(fn, flush)
            moved = (
                bytes_moved(c)
                if impl.startswith("tokenforge")
                else (c.n * c.k * 2 + c.m * c.k * 2 + c.m * c.n * 2)
            )
            gbps = moved / (ms * 1e-3) / 1e9
            rows.append(
                {
                    **asdict(c),
                    "impl": impl,
                    "gpu": gpu,
                    "ms": ms,
                    "gbps": gbps,
                    "pct_peak_bw": 100 * gbps / PEAK_GBPS.get(gpu, float("nan")),
                    "tflops": 2 * c.m * c.n * c.k / (ms * 1e-3) / 1e12,
                    "env": fp,
                }
            )
            print(
                f"{c.layer:28s} M={c.m:<4d} {impl:18s} {ms * 1e3:9.1f} us {gbps:7.1f} GB/s",
                flush=True,
            )
    return rows


def _key(r: dict[str, Any]) -> str:
    return f"{r['impl']}|{r['layer']}|{r['m']}"


def check(rows: list[dict[str, Any]], baseline: dict[str, float]) -> list[str]:
    failures = []
    for r in rows:
        base = baseline.get(_key(r))
        if base is not None and r["ms"] > base * (1 + REGRESSION):
            failures.append(
                f"{_key(r)}: {r['ms']:.4f} ms vs baseline {base:.4f} ms "
                f"(+{100 * (r['ms'] / base - 1):.1f}%)"
            )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--suite", default="l4", choices=["l4", "t4", "full"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-tune", action="store_true", help="use cached/default configs only")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--reason", default="", help="required with --update-baseline")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("microbenchmarks need a CUDA device", file=sys.stderr)
        return 1
    rows = run(suite(args.suite), tune=not args.no_tune)
    out_dir = ROOT / "bench" / "results" / rows[0]["gpu"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "micro.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    baseline_path = out_dir / "micro_baseline.json"
    if args.update_baseline:
        if not args.reason:
            print("--update-baseline needs --reason", file=sys.stderr)
            return 1
        doc = {
            "reason": args.reason,
            "git_sha": rows[0]["env"]["git_sha"],
            "ms": {_key(r): r["ms"] for r in rows},
        }
        baseline_path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return 0
    if args.check:
        if not baseline_path.exists():
            print(f"no baseline at {baseline_path}", file=sys.stderr)
            return 1
        failures = check(rows, json.loads(baseline_path.read_text())["ms"])
        for line in failures:
            print("REGRESSION " + line, file=sys.stderr)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
