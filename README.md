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
| M0 Scaffolding and CI | Built: tooling, CPU CI workflow, `versions.lock`, ADRs D1–D7 (+0008, 0009), gateway + mock engine, Helm charts, kind `make up`. 232 CPU-tier tests green in < 2 min |
| M1 Triton kernel | Built: W4A16 GEMM/GEMV, dequant, fused RMSNorm + FP8 quant, AWQ repack, dispatch, custom ops, autotune cache. T0–T2, F1, F4 green under the Triton interpreter |
| M2 CUDA kernel on L4 | Not started |
| M3 vLLM plugin | Not started |
| M4 Serving stack | Gateway, mock engine and charts built; real engines not started |
| M5 Chaos and observability | Gateway-level F11, F13–F17, F19 and the F12 probe built; cluster suite not started |
| M6 Benchmark and write-up | Microbenchmark harness built; load generator and matrix not started |

GPU spend to date: $0 of $75 (`make budget-report`).

Every performance claim in this README will trace to a committed JSONL row under
`bench/results/` and the commit that produced it.

## What remains

Each milestone closes when every exit criterion in [`docs/design.md`](docs/design.md) holds
and its git tag points at a commit with a passing GPU-full report.

### M0 Scaffolding and CI

- [ ] First green run of `.github/workflows/ci.yml` on GitHub, including the `kind` job
      (`make up` + smoke test on a runner)
- [ ] Resolve the `resolve-on-l4` pins in `infra/versions.lock` (Triton Inference Server
      TRT-LLM and vLLM images, `vllm/vllm-openai`, DCGM exporter, device plugin, k3s, KEDA,
      Hugging Face model revisions); `make lock-check` passes
- [ ] Tag `m0-scaffolding`

### M1 Triton kernel (Kaggle T4)

- [ ] `make calibrate-cuda` against cuBLAS; commit the `cuda` entry in
      `tests/numerics/tolerances.json`
- [ ] `make gpu-dev`: full shape suite, compute-sanitizer (memcheck, racecheck, initcheck,
      synccheck), microbenchmarks; commit the report under `bench/results/ci/`
- [ ] Tune configs per M bucket; reach >= 60% of T4 peak DRAM bandwidth at M = 1
- [ ] Commit `bench/results/tesla-t4-sm75/micro_baseline.json`; tag `m1-kernels`

### M2 CUDA kernel on L4

- [ ] `tokenforge/kernels/csrc/w4a16_gemv.cu` (M = 1–4): warp per N tile, 128-bit loads,
      LOP3 + magic-number int4 -> fp16 unpack, fp32 accumulate, warp-shuffle reduce, split-K
      for small N; `-lineinfo -Werror -Wall -Wextra`, clang-format, clang-tidy
- [ ] Torch extension build and `tokenforge.kernels._C` registration behind
      `TOKENFORGE_BACKEND=cuda`
- [ ] >= 75% of 300 GB/s at M = 1; Triton within 15% of CUDA; within 1.25x of `awq_marlin`
      at M in {1, 8, 32}
- [ ] FP8 kernel T2 on sm89; compute-sanitizer clean on the full shape suite
- [ ] Nsight Compute roofline (CSV + plot) committed; L4 baseline committed; tag `m2-cuda`

### M3 vLLM plugin

- [ ] `vllm_plugin/tokenforge_vllm`: `TokenForgeConfig` + `TokenForgeLinearMethod`
      (`create_weights`, `process_weights_after_loading` with repack and tuning, `apply`),
      `vllm.general_plugins` entry point
- [ ] Version guard for `>=0.33.0,<0.34.0` (F6); `TOKENFORGE_BACKEND=marlin` kill switch;
      per-layer fallback to `awq_marlin` with `tokenforge_fallback_total` (F2)
- [ ] T3: WikiText-2 perplexity delta <= 0.02 vs `awq_marlin`; >= 95% of 200 prompts
      greedy-identical for 32 tokens; GSM8K-250 within 1 point (0.5B on T4, 7B on L4)
- [ ] T4: identical outputs with CUDA graphs on/off and torch.compile on/off, chunked
      prefill, prefix caching, forced preemption; zero host syncs in `apply` (F5, F7)
- [ ] Tag `m3-plugin`

### M4 Serving stack

- [ ] `make engines`: TensorRT-LLM FP8 and INT4-AWQ engines keyed by (model, quant, sm,
      TensorRT version), SHA-256 manifest in object storage; init container verifies hash
      and key (F8, F9)
- [ ] `serving/triton_model_repo/` for the TRT-LLM and vLLM backends; vLLM OpenAI server as
      the control; Helm charts for each engine; gateway wired to the HF tokenizer
- [ ] k3s on the L4 with the NVIDIA device plugin and DCGM exporter
- [ ] All 7 configurations serve behind the gateway; T5 on real engines; F10, F16, F19 pass
- [ ] `make up` from a fresh L4 in < 20 min, timed; tag `m4-serving`

### M5 Chaos and observability

- [ ] Prometheus, one Grafana dashboard per layer in `serving/helm/dashboards/`,
      OpenTelemetry spans gateway -> engine (CMPE 273 trace-aggregator SDK)
- [ ] Alerts with 5 min / 1 h burn rates: `TTFTBudgetBurn`, `KVCacheSaturated`,
      `EngineNotProgressing`, `TokenForgeFallbackHigh`, `GPUXidError`, `EngineCrashLoop`,
      `MetricsAbsent`; a runbook for each in `docs/runbooks/`
- [ ] `chaos/scenarios/` for F1–F23 (Chaos Mesh, Toxiproxy), each asserting steady state ->
      fault -> expected behavior -> recovery; every alert fires in its scenario and each
      runbook is used once
- [ ] KEDA autoscaling in kind with the mock engine, then the one-day 2x L4 run; tag
      `m5-chaos`

### M6 Benchmark and write-up

- [ ] Open-loop load generator: Poisson and gamma (CV = 3) arrivals; every stream verified
      to end in `[DONE]` or an error event
- [ ] Workloads pinned by SHA-256: ShareGPT chat, RAG 4k-in / 256-out, short 128 / 128
- [ ] TTFT, TPOT, ITL, end-to-end, goodput under SLO, cost per 1M output tokens; NVML
      throttle sampling with discard-and-rerun (F21); resume from the last cell (F22)
- [ ] Quality per precision: WikiText-2 perplexity, GSM8K-250
- [ ] Full matrix (7 configs x 3 workloads x 6 rates), 3 repeats, CV < 5%, one JSONL row per
      cell with the environment fingerprint
- [ ] README results and write-up from committed results; tag `m6-bench`

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
