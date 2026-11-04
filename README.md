# TokenForge

One 7B model, Qwen2.5-7B-Instruct, from raw weights to a fault-tested, benchmarked
production deployment on a single NVIDIA L4, for under $75 of GPU time. Every layer of the
inference stack is built or integrated: custom Triton and CUDA kernels, a vLLM quantization
plugin, TensorRT-LLM engines, Triton Inference Server on Kubernetes, and a goodput-under-SLO
benchmark.

The deliverable is a decision: which engine and precision to run for which traffic shape,
at what cost per 1M tokens, and how each configuration fails. Design and execution plan:
[`docs/design.md`](docs/design.md). Decisions: [`docs/adr/`](docs/adr/).

## Status

| Milestone | State |
| --- | --- |
| M0 Scaffolding and CI | Tooling, CPU CI, `versions.lock`, ADRs D1–D7, gateway + mock engine, Helm charts, kind `make up` |
| M1 Triton kernel | W4A16 GEMM/GEMV, dequant, fused RMSNorm + FP8 quant, AWQ repack, dispatch, custom ops, autotune cache; T0–T2 and F1/F4 pass under the Triton interpreter; T4 run pending |
| M2–M6 | Not started |

Every performance claim in this README will trace to a committed JSONL row under
`bench/results/` and the commit that produced it.

## Layout

```
tokenforge/               Python package (ADR 0009)
  kernels/                triton/ (w4a16, rmsnorm_fp8), pack.py, dispatch.py, ops.py, autotune.py
  serving/gateway/        OpenAI-compatible gateway: validation, admission, retries, SSE relay
  serving/mock_engine/    continuous-batching engine simulator with vLLM's API and metrics
  serving/progress.py     progress-watchdog liveness probe (F12)
  bench/                  microbenchmarks, environment fingerprint
  ci/                     GPU-tier runner and report gate (ADR 0008)
  ledger.py               GPU spend ledger and the $70 provisioning stop
serving/                  helm/, docker/, kind/, keda/
bench/results/            committed results (<gpu>/*.jsonl, ci/*.json)
infra/                    provision.sh, watchdog.sh, up.sh, ledger.csv, versions.lock
tests/                    unit/, numerics/, contract/, integration/
docs/                     design.md, adr/, runbooks/
```

## Quickstart

```bash
make setup        # uv env from uv.lock
make test-cpu     # unit + kernel numerics (Triton interpreter) + gateway contract tests
make lint typecheck
make up           # gateway + mock engine in kind on http://127.0.0.1:8080
```

GPU tiers:

```bash
make gpu-dev                                   # Kaggle T4: numerics, sanitizers, microbench
TF_PROVIDER=runpod TF_RATE=0.44 make provision # rented L4: budget gate, env, watchdog
make gpu-ci                                    # L4: GPU-full tier, writes bench/results/ci/
```

## Kernel layer at a glance

* **Weights:** standard AWQ GEMM checkpoints. `pack_awq` repacks qweight once at load time
  to `[N, K/8]`, packed along K with the AWQ nibble interleave; scales and zero-points are
  used as stored. Dequant is `fp16(q - z) * s`, one rounding, identical in every path.
* **W4A16 Triton kernel (M = 1–128):** BLOCK_M = 1 GEMV with a deferred reduction,
  broadcast FMA for BLOCK_M < 16, `tl.dot` for BLOCK_M ≥ 16; in-register dequant;
  deterministic split-K. M > 128 routes to dequant + cuBLAS.
* **RMSNorm + FP8:** one read of the hidden state, per-token dynamic or static e4m3 scale,
  saturating cast with explicit round-to-nearest-even onto the e4m3 grid (bit-exact against
  PyTorch at every midpoint).
* **Tolerances are derived:** `atol = c · sqrt(K) · eps_fp16 · max|x| · max|w|`, with `c`
  calibrated against the vendor fp16 GEMM and frozen in `tests/numerics/tolerances.json`.
