# TokenForge — Design & Execution Plan

Oct 30, 2026 · @S.S.

## Summary

TokenForge takes one 7B model from raw weights to a fault-tested, benchmarked production deployment on a single NVIDIA L4. It runs in 9 weeks for under $75 of GPU time. Every layer of the inference stack is built or integrated: custom Triton and CUDA kernels, a vLLM quantization plugin, TensorRT-LLM engines, Triton Inference Server on Kubernetes, and a goodput-under-SLO benchmark.

The deliverable is a decision, not a demo: which engine and precision to run for which traffic shape, at what cost per 1M tokens, and how each configuration fails.

### Goals

1. **Kernels.** Ship a W4A16 GEMV/GEMM in both Triton and CUDA, plus a fused RMSNorm + FP8 quant kernel in Triton. Each kernel is roofline-characterized and sanitizer-clean.
2. **Engine integration.** Load the kernels into vLLM as an out-of-tree quantization backend that is CUDA-graph and torch.compile safe.
3. **Production serving.** Serve vLLM and TensorRT-LLM behind Triton Inference Server on k3s, with a gateway, admission control, autoscaling, metrics, traces and runbooks.
4. **Failure-mode coverage.** Every failure mode in the matrix below has an automated injection test with an asserted outcome.
5. **Benchmark.** Produce a reproducible engine × precision × workload matrix: goodput under SLO, cost per 1M tokens, quality delta.

### Non-goals

- Training or fine-tuning.
- Multi-GPU tensor parallelism, or Hopper/Blackwell-specific code (wgmma, TMA). SparseBench's H100 port covers that.
- Writing a new serving engine. PrefixServe already demonstrates scheduler and KV-cache design.
- Beating TensorRT-LLM or Marlin in absolute speed. The bar is to match them within a stated margin and explain the gap.
- A new quantization algorithm. TokenForge consumes standard AWQ checkpoints.

### Success metrics

| Area | Metric | Target |
| --- | --- | --- |
| Kernel | W4A16 GEMV achieved DRAM bandwidth, M=1, L4 | ≥ 75% of 300 GB/s |
| Kernel | CUDA vs Triton W4A16, M=1 | Triton within 15% of CUDA |
| Kernel | vs vLLM awq\_marlin, M ∈ {1, 8, 32} | Within 1.25× of Marlin latency |
| Kernel | compute-sanitizer (memcheck, racecheck, initcheck, synccheck) | 0 errors over the full shape suite |
| Plugin | WikiText-2 perplexity vs awq\_marlin, same weights | Δ ≤ 0.02 |
| Plugin | Greedy outputs identical to awq\_marlin, first 32 tokens, 200 prompts | ≥ 95% of prompts |
| Serving | Failure-mode tests passing | 100%, or the gap documented with a reason |
| Serving | Silent stream truncations under chaos | 0 |
| Benchmark | Run-to-run coefficient of variation, 3 repeats | < 5% |
| Budget | Total GPU spend | ≤ $75 |

## Context and constraints

The portfolio already proves CUDA tensor-core kernels (SparseBench), KV-cache scheduling (PrefixServe), lock-free systems code (FluxIngest, RustLog) and LLM tracing (CMPE 273 trace aggregator). TokenForge fills the gaps an AI infra hiring loop probes: Triton, vLLM internals, TensorRT-LLM, quantization, Kubernetes GPU serving and SLO-driven benchmarking.

### Hardware and budget

| Resource | Use | Cost |
| --- | --- | --- |
| Kaggle T4 (sm\_75, 16 GB) | Kernel dev, unit and numerics tests, Qwen2.5-0.5B integration | Free, 30 h/week |
| GitHub Actions (CPU) | Lint, types, Triton interpreter tests, Helm and kind tests | Free |
| Rented 1× L4 (sm\_89, 24 GB, 300 GB/s) | FP8, INT4-AWQ, TensorRT-LLM, serving, chaos, benchmarks | \~$0.39–0.80/h ([Runpod](https://computeprices.com/providers/runpod/gpus/l4), [Jarvislabs](https://jarvislabs.ai/blog/l4-gpu-price)) |
| Rented 2× L4, one day | The only real 1→2 replica autoscaling test | \~$8 |

The L4 is the target because TensorRT-LLM gates key paths by architecture. INT4 AWQ/GPTQ is unsupported below SM 80, and INT4 with FP8 activations needs SM ≥ 89 ([TensorRT-LLM support matrix](https://github.com/JidongZhang-THU/TensorRT-LLM)). A T4 cannot run the comparison.

### Constraints

- **Budget cap $75.** A hard stop, tracked in `infra/ledger.csv` after every session.
- **One GPU for most work.** Autoscaling logic is proven with a mock engine on CPU, then validated once on 2× L4.
- **Rented boxes are ephemeral and may be interrupted.** All state lives in git or object storage, and every long job checkpoints.
- **Host RAM ≥ 64 GB** on the L4 box. TensorRT-LLM engine builds for 7B models are host-memory heavy.

### Assumptions

- Model: **Qwen2.5-7B-Instruct**, with its official AWQ checkpoint. Apache-2.0, no gating. FP16 weights (\~15 GB) fit a 24 GB L4 with room left for KV cache.
- Dev model: **Qwen2.5-0.5B-Instruct**, same architecture family, fits on a T4 and in CI.
- Versions of vLLM, TensorRT-LLM, the Triton server container, Triton, PyTorch and the driver are pinned in week 1 and recorded in `versions.lock`. Upgrades follow the policy in Engineering hygiene.

## Architecture

&#91;embedded content: TokenForge architecture · 4 layers on one L4\]

Requests flow top-down: load generator → gateway → Triton Inference Server → one of two backends. The TokenForge plugin swaps vLLM's linear layers for the custom kernels and falls back to awq\_marlin on unsupported shapes. Dashed lines are metrics and traces. The chaos suite injects faults into the cluster.

## Detailed design

### L1: Kernels (`kernels/`)

**Weight format.** Kernels consume the standard AWQ GEMM layout: int4 packed 8 per int32, group size 128, fp16 scales, packed zero-points. Reusing the format means the same checkpoint runs through Marlin, TensorRT-LLM and TokenForge. Quality differences then come from kernels, not from quantization.

| Kernel | Language | Shapes | Key techniques |
| --- | --- | --- | --- |
| `w4a16_gemv` | CUDA | M = 1–4 | Warp per N-tile, 128-bit loads, LOP3 + magic-number int4→fp16 unpack, fp32 accumulate, warp-shuffle reduce, split-K for small N |
| `w4a16_gemm` | Triton | M = 1–128 | Autotuned block sizes per M bucket, in-register dequant, split-K, `tl.dot` on fp16 |
| `rmsnorm_fp8_quant` | Triton | Any token count | Fused RMSNorm, per-token dynamic e4m3 scale, saturating cast; one read of the hidden state |
| `pack_awq` | Python + Triton | Load time only | Repack AWQ to the kernel's interleaved layout; inverse provided for tests |

**Dispatch.** A pure-Python dispatcher picks a kernel from (M, N, K, GPU sm). It routes M > 128 to dequant + cuBLAS, and any unsupported shape to vLLM's awq\_marlin. Every fallback increments a labeled counter. Silent fallbacks would invalidate benchmarks.

**Op registration.** Each kernel is a `torch.library.custom_op` with a fake (meta) implementation. That makes it traceable by torch.compile and capturable in CUDA graphs, both of which vLLM uses. Kernels never allocate, never sync with the host, and never read tensor values on the host.

**Autotune cache.** Triton autotune results persist to `~/.cache/tokenforge/<gpu>-<triton_ver>.json`, written atomically (tmp → fsync → rename). A corrupt or mismatched cache is discarded with a warning, never trusted.

### L2: vLLM plugin (`vllm_plugin/`)

- **Packaging.** A separate pip package, `tokenforge-vllm`, registered through the `vllm.general_plugins` entry point. No vLLM fork, so upgrades are a version bump, not a rebase.
- **Classes.** `TokenForgeConfig(QuantizationConfig)`, registered with `register_quantization_config("tokenforge")`, and `TokenForgeLinearMethod(LinearMethodBase)` implementing `create_weights`, `process_weights_after_loading` (repack once) and `apply` (dispatch).
- **Version guard.** At registration the plugin checks the installed vLLM against a supported range and fails loudly outside it. API churn surfaces at startup, not as wrong outputs.
- **Backend switch.** `TOKENFORGE_BACKEND` = `triton` | `cuda` | `marlin`. The `marlin` setting is a kill switch to vLLM's built-in path, with no redeploy of weights.
- **Must work with:** CUDA graphs on and off, torch.compile, chunked prefill, prefix caching, and preemption.

### L3: Serving (`serving/`)

- **Cluster.** k3s on the L4 host, NVIDIA device plugin and DCGM exporter. Helm chart per component, values per engine config.
- **Engines.** Triton Inference Server with two backends: TensorRT-LLM (in-flight batching, paged KV) and vLLM. vLLM's own OpenAI server also runs as a control, so Triton's overhead is measured, not assumed.
- **Gateway.** A small FastAPI/uvicorn service with an OpenAI-compatible API in front of the engines. It owns:
  - request IDs and trace context propagation
  - validation: prompt vs max context, `max_tokens` bounds
  - admission control: a bounded queue with a waiting-requests threshold read from engine metrics; beyond it, 429 with `Retry-After`
  - retries only before the first token is streamed; after that, a terminal SSE error event, never a silent close
  - client-disconnect propagation to engine abort
- **Autoscaling.** KEDA scales on queue depth (`num_requests_waiting`). Scale logic is tested on CPU in kind with PrefixServe's mock engine, then once on 2× L4.
- **Artifacts.** TensorRT-LLM engines are built by `make engines` and keyed by (model, quant, sm, TensorRT version). They are stored in object storage with a SHA-256 manifest. An init container verifies the hash and the key before the engine starts.

### L4: Benchmark harness (`bench/`)

- **Load generator.** Async Python client, open-loop Poisson arrivals and a bursty gamma process with CV = 3. Open-loop matters: closed-loop clients hide queueing collapse.
- **Workloads.** ShareGPT chat; RAG-shaped 4k-in / 256-out; short 128 / 128. Prompt sets are pinned by SHA-256, and download scripts verify them.
- **Metrics.** TTFT, TPOT, inter-token latency, end-to-end latency, throughput, and goodput = requests/s meeting both the TTFT and TPOT SLOs. Cost per 1M output tokens = hourly price ÷ output tokens per hour at max goodput.
- **Quality.** WikiText-2 perplexity and a 250-question GSM8K subset per precision.
- **Method.** Warmup, then 3 repeats per cell; report median and IQR. NVML samples clocks, temperature, power and throttle reasons every 100 ms, and any run with a thermal or power throttle flag is discarded and rerun.
- **Records.** One JSONL row per cell with a full environment fingerprint: git SHA, container digests, driver, GPU UUID, `versions.lock` hash.

**Configuration matrix:** vLLM {FP16, FP8, awq\_marlin, tokenforge-triton, tokenforge-cuda} and TensorRT-LLM {FP8, INT4-AWQ}, each behind Triton Inference Server, × 3 workloads × 6 arrival rates. SGLang is optional (see Open questions).

## Engineering hygiene

One monorepo, one `Makefile` entry point, and nothing that only works on one person's machine. Every claim in the README traces back to a committed JSONL result and the commit that produced it.

### Repo layout

```
tokenforge/
  kernels/        triton/, csrc/, dispatch.py, ops.py (torch.library registration)
  vllm_plugin/    tokenforge_vllm/ (own pyproject, entry point)
  serving/        helm/, triton_model_repo/, gateway/, keda/
  bench/          loadgen/, workloads/, analysis/, results/<gpu>/*.jsonl
  chaos/          scenarios/*.py, chaos-mesh/*.yaml, toxiproxy/
  infra/          provision.sh, watchdog.sh, ledger.csv, versions.lock
  tests/          unit/, numerics/, model/, integration/, contract/
  docs/           adr/, runbooks/, design.md
```

### Code quality

- **Python:** `uv` with a lockfile, `ruff` (lint + format), `mypy --strict` on gateway, bench and plugin code. Kernel code gets `mypy` with Triton stubs relaxed.
- **CUDA/C++:** `clang-format`, `clang-tidy`, `-Werror -Wall -Wextra`, and `-lineinfo` builds so Nsight and compute-sanitizer report source lines.
- **Infra:** `helm lint`, `kubeconform` against the cluster version, `shellcheck`, `hadolint`.
- **pre-commit** runs all of the above locally, and CI runs the same config.

### CI tiers

| Tier | Where | Trigger | Runs | Budget |
| --- | --- | --- | --- | --- |
| CPU | GitHub Actions | Every push and PR | Lint, types, unit and property tests, Triton kernels under `TRITON_INTERPRET=1`, Helm and kind tests with the mock engine, gateway contract tests | < 10 min |
| GPU-dev | Kaggle T4 notebook | Any PR touching `kernels/` or `vllm_plugin/` | Numerics suite, compute-sanitizer, microbench vs T4 baseline, 0.5B model tests | < 30 min |
| GPU-full | Rented L4, `make gpu-ci` | Before merging a milestone, and any dependency bump | Everything above + FP8 + TensorRT-LLM + 7B quality gates + chaos suite | < 2 h |

GPU tiers cannot run on free GitHub runners, so `make gpu-ci` writes a report JSON (git SHA, environment fingerprint, pass/fail per test) to `bench/results/ci/`. The PR template requires a link to it for kernel or plugin changes. CI fails if the report's SHA does not match the PR head.

### Reproducibility

- Containers pinned by digest, never by tag. `versions.lock` lists every one.
- Seeds fixed for prompt sampling, arrival processes and Hypothesis. Datasets are fetched by script and verified by SHA-256.
- `make up` brings the whole stack up from a fresh L4 box in under 20 minutes. This is timed and tracked as a regression metric.

### Change management

- **ADRs** in `docs/adr/` for each decision in the Decision log, using the template: context, decision, consequences, alternatives.
- **Conventional commits**, a generated `CHANGELOG.md`, and a git tag per milestone (`m1-kernels`, `m2-cuda`, ...).
- **Dependency upgrades:** one dependency per PR, with a full GPU-full run and a before/after benchmark diff in the PR description.
- **Self-review:** every PR has a description stating what changed, how it was tested and its perf impact. Open it as a draft, wait 24 hours, review it cold, then merge.

### Cost and security

- `infra/watchdog.sh` powers the rented box off when GPU utilization stays under 5% for 20 minutes. Every session is appended to `ledger.csv`, and `make` refuses to provision once spend reaches $70.
- Hugging Face token and storage credentials come only from environment variables. Never baked into images; `gitleaks` runs in pre-commit.
- Gateway logs never contain prompt or completion text by default, only lengths and IDs.

## Testing strategy

Correctness is proven bottom-up in tiers. A tier runs only after the one below it is green, so a model-level failure is never debugged from the top down.

| Tier | Scope | What it asserts | Runs on |
| --- | --- | --- | --- |
| T0 Unit and property | Packing, dispatch, gateway logic | `unpack(pack(w)) == w` for random shapes, group boundaries, zero-points at 0 and 15; dispatcher picks the expected kernel per shape; Hypothesis-generated inputs | CPU CI |
| T1 Kernel numerics | Each kernel vs an fp32 reference on dequantized weights | Error within a K-scaled tolerance (below); NaN and Inf propagate; no fp16 overflow with activations at ±6e4; zero and denormal scales handled | Interpreter on CPU, then T4 |
| T2 Cross-implementation | TokenForge vs awq\_marlin on identical AWQ weights; FP8 kernel vs a PyTorch e4m3 reference | Max relative error ≤ tolerance; e4m3 saturation (±448) matches the reference exactly | T4, L4 for FP8 |
| T3 Model quality | Full model in vLLM | Perplexity Δ ≤ 0.02 vs awq\_marlin; ≥ 95% of 200 prompts greedy-identical for 32 tokens; GSM8K-250 within 1 point | T4 (0.5B), L4 (7B) |
| T4 Engine integration | Plugin inside vLLM | Identical outputs with CUDA graphs on vs off, torch.compile on vs off, chunked prefill, prefix caching, forced preemption; zero host syncs in `apply` | T4, L4 |
| T5 Serving contract | Gateway + engines | OpenAI schema conformance, SSE framing, `usage` token counts equal engine counts, stop sequences, cancellation, 4xx vs 5xx mapping | CPU CI (mock), L4 (real) |
| T6 Failure modes | Whole stack | Every row of the failure matrix, below | kind (mock), L4 |

### Numerical tolerance

Tolerances are derived, not guessed. With fp32 accumulation, the expected error scales with sqrt(K). The test uses atol = c · sqrt(K) · eps\_fp16 · max|x| · max|w|, with c calibrated once against cuBLAS fp16 on random data and then frozen in `tests/numerics/tolerances.json`. A kernel that needs a looser constant than cuBLAS fails.

### Shape suite

- **M:** 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 127, 128, 129. Odd values catch tail-handling bugs; 129 crosses the fallback boundary.
- **N, K:** every linear layer shape in Qwen2.5-0.5B and 7B (q/k/v/o, gate/up/down), plus N not divisible by the tile size.
- **Non-contiguous and misaligned inputs:** the kernel asserts or handles them. A misaligned pointer never produces silent garbage.

### Sanitizers and fuzzing

- `compute-sanitizer` with memcheck, racecheck, initcheck and synccheck over the full shape suite, for every CUDA kernel and for Triton kernels compiled with `-lineinfo`.
- Gateway request fuzzing with Schemathesis against its OpenAPI spec: malformed JSON, empty and huge prompts, invalid UTF-8, `max_tokens` of 0 and 10^9, unknown fields.

### Performance regression

- Microbenchmarks use CUDA events: 25 warmup iterations, median of 200, L2 flushed between iterations, so cold-cache numbers are what is measured.
- Baselines live in `bench/results/<gpu>/micro_baseline.json`. A change that is > 5% slower than baseline on any shape fails GPU-dev or GPU-full. Updating a baseline needs a commit that says why.
- Roofline: achieved bandwidth and FLOP/s per kernel and shape, from Nsight Compute, committed as a plot and raw CSV.

## Failure modes and fault injection

Every failure mode is a test, not a paragraph. Each scenario in `chaos/scenarios/` follows the same four steps: assert steady state, inject the fault, assert the expected behavior while it is active, then assert recovery back to steady state within a deadline. M5 does not exit until every row passes, or is listed in Known gaps with a reason.

**Steady state** means: gateway 5xx rate = 0, p99 TTFT under SLO at 50% of max goodput, KV cache usage back at baseline after load stops, and `num_requests_running` = 0 when idle.

### Kernel and engine

| # | Failure | Injection | Expected behavior | Detection |
| --- | --- | --- | --- | --- |
| F1 | Activation outliers overflow fp16 | T1 inputs at ±6e4 and with Inf | fp32 accumulate, no spurious Inf; a real Inf propagates | T1 assertion |
| F2 | Unsupported shape reaches plugin | Synthetic layer with K not a multiple of 128 | Falls back to awq\_marlin, counter increments, outputs correct | `tokenforge_fallback_total` |
| F3 | Illegal memory access in kernel | Test-only build flag forces an out-of-bounds write | CUDA context is poisoned, so the process exits instead of serving garbage; pod restarts; gateway marks it unready and returns retriable 503 | Pod restart count, gateway 503 rate |
| F4 | Corrupt or stale autotune cache | Truncate and garble the cache file | Cache discarded with a warning, retuned, outputs unchanged | Log line, T1 rerun |
| F5 | CUDA graph capture breaks with plugin | Run T4 with graphs on vs `enforce_eager` | Identical outputs; any difference fails CI | T4 assertion |
| F6 | vLLM version outside supported range | Install a newer vLLM in a scratch env | Plugin refuses to register with a clear message; never loads and runs | Startup log, exit code |
| F7 | Quality regression from a kernel change | Deliberately flip one scale's sign in a test build | Perplexity gate fails | T3 gate |
| F8 | TensorRT engine built for another sm or TensorRT version | Deploy an engine with a mismatched manifest | Init container refuses; engine never starts | Init container exit code |
| F9 | Model artifact corrupted in storage | Flip one byte in a weight shard | SHA-256 mismatch, init container fails | Same |
| F10 | GPU OOM at startup | Set `gpu_memory_utilization` to 0.99 | Startup probe fails, pod in CrashLoopBackOff, alert fires; no half-ready pod serving | Startup probe, alert |

### Serving and load

| # | Failure | Injection | Expected behavior | Detection |
| --- | --- | --- | --- | --- |
| F11 | KV cache exhaustion | Flood with 4k-token prompts at 3× max goodput | Engine preempts and recomputes, never OOMs; gateway sheds with 429 + `Retry-After` before p99 TTFT exceeds 5× SLO | KV usage, 429 rate, TTFT p99 |
| F12 | Engine hangs | `SIGSTOP` the engine process | Progress watchdog (queue > 0 and no new tokens for 30 s) fails liveness; pod restarts within 60 s | Liveness failures, restart count |
| F13 | Pod killed mid-stream | `kill -9` the engine with 64 active streams | Requests with no tokens sent are retried once and succeed; requests already streaming get a terminal SSE error event; 0 silent truncations | Loadgen verifies every stream ends with `[DONE]` or an error event |
| F14 | Clients disconnect mid-stream | 500 clients drop at random points | Engine aborts each request within one step; KV usage and running count return to baseline (no leak) | `num_requests_running`, KV usage after 10 s |
| F15 | Slow reader | Client reads SSE at 1 byte/s | Bounded per-request buffer; request aborted after the write timeout; other requests' TPOT unaffected | TPOT p99 of the other requests |
| F16 | Invalid requests | Schemathesis + prompt > context, `max_tokens` = 10^9 | 4xx with a reason, never 5xx, engine never sees the request | 5xx count = 0 |
| F17 | Cold start after scale-from-zero | KEDA scales 0→1 under a waiting burst | Gateway holds requests in a bounded queue up to the timeout, then 503s; cold-start time recorded | Queue depth, cold-start histogram |
| F18 | Gateway ↔ engine network faults | Toxiproxy: 200 ms latency, connection resets, 1 KB/s bandwidth | Timeouts fire as configured; resets before the first token are retried; no hung gateway workers | Gateway RED metrics |
| F19 | Tokenizer mismatch between gateway and engine | Deploy the gateway with a different tokenizer revision | Contract test fails on the `usage` token count | T5 assertion |

### Infrastructure

| # | Failure | Injection | Expected behavior | Detection |
| --- | --- | --- | --- | --- |
| F20 | GPU becomes unhealthy (Xid) | Simulated: mark the device unhealthy in the device plugin, then cordon the node | Pod evicted and shown unschedulable; alert fires; runbook followed | DCGM Xid metric, alert |
| F21 | Thermal or power throttling during a benchmark | Observed, not injected; also forced with a power cap where the host allows it | The run is flagged and discarded, never reported | NVML throttle reasons in the run record |
| F22 | Rented instance interrupted mid-campaign | Kill the box mid-run | Harness resumes from the last completed cell; no duplicate or partial rows | Row count and uniqueness check |
| F23 | Prometheus down | Delete the Prometheus pod | Serving unaffected; `absent()` alert fires | Alert |

### Known gaps (accepted)

- Real Xid errors, ECC faults and NVLink failures cannot be injected on rented hardware. F20 tests the response path only.
- Multi-node failures are out of scope: there is a single node, by design.

## Observability, SLOs and runbooks

Any incident in the chaos suite must be diagnosable from dashboards and traces alone, without SSH. That is the acceptance test for this section.

### Signals

| Source | Signals | Why |
| --- | --- | --- |
| Gateway | Request rate, errors by code, duration (RED); queue depth; 429 and retry counts; stream terminations by type | User-facing truth |
| vLLM / Triton metrics | Running and waiting requests, KV cache usage, preemptions, TTFT and TPOT histograms | Engine saturation |
| TokenForge plugin | `tokenforge_kernel_calls_total{kernel,bucket}`, `tokenforge_fallback_total{reason}` | Proves the custom path is actually running |
| DCGM exporter | SM utilization, DRAM bandwidth utilization, power, temperature, clocks, Xid errors | Hardware health and roofline context |
| Traces | OpenTelemetry spans gateway → engine: queue wait, prefill, decode, stream write. Reuses the CMPE 273 trace-aggregator SDK | Where the time went for one request |
| Logs | Structured JSON, request ID and trace ID on every line, no prompt text | Correlation |

One Grafana dashboard JSON per layer lives in `serving/helm/dashboards/` and is provisioned with the stack.

### SLOs (for the reference deployment)

| SLO | Target | Window |
| --- | --- | --- |
| Availability: non-5xx responses | 99.5% | Each 1 h benchmark run |
| TTFT p99 at the target rate | < 1 s (chat), < 2.5 s (RAG 4k) | Same |
| TPOT p99 | < 60 ms | Same |
| Streams ending in `[DONE]` or an explicit error event | 100% | Same |

Alerts use multi-window burn rates (5 min and 1 h) against the error budget, so a short spike pages less than a sustained burn.

### Runbooks (`docs/runbooks/`)

Each alert links to a runbook with the same five headings: symptom, dashboard panel, likely causes ranked, mitigation steps, follow-up. Runbooks are tested: each chaos scenario names the runbook it should trigger, and the runbook's steps are followed during that test.

| Alert | Fired by | First mitigation |
| --- | --- | --- |
| `TTFTBudgetBurn` | F11, F17 | Lower the admission threshold; check for preemption storms |
| `KVCacheSaturated` | F11 | Shed load; lower `max_num_seqs`; check for leaked requests (F14) |
| `EngineNotProgressing` | F12 | Confirm liveness restarted the pod; capture `py-spy` dump before restart if possible |
| `TokenForgeFallbackHigh` | F2, F6 | Check shape logs; flip `TOKENFORGE_BACKEND=marlin` if outputs are suspect |
| `GPUXidError` | F20 | Cordon the node; drain; reprovision the box |
| `EngineCrashLoop` | F3, F8, F9, F10 | Read the init and startup logs; verify manifest and memory settings |
| `MetricsAbsent` | F23 | Restart Prometheus; serving needs no action |

## Milestones and exit criteria

&#91;embedded content: Milestones by week · planned GPU spend per milestone\]

Planned spend is about $50 against the $75 cap; the rest is buffer for reruns. A milestone closes only when every exit criterion holds and the milestone's git tag points at a commit with a passing GPU-full report.

| Milestone | Exit criteria |
| --- | --- |
| M0 Scaffolding and CI | CPU CI green; `make up` runs the mock stack in kind; `versions.lock` committed; ADRs D1–D7 written |
| M1 Triton kernel | T0–T2 pass on T4; ≥ 60% of T4 peak bandwidth at M = 1; F4 passes |
| M2 CUDA kernel on L4 | ≥ 75% of L4 peak bandwidth at M = 1; Triton within 15%; compute-sanitizer clean; roofline committed |
| M3 vLLM plugin | T3 and T4 pass on 0.5B (T4) and 7B (L4); F2, F5, F6, F7 pass; kill switch verified |
| M4 Serving stack | All 7 configurations serve behind the gateway; T5 passes; F8–F10, F16, F19 pass; `make up` under 20 min |
| M5 Chaos and observability | F1–F23 pass or are listed as known gaps; every alert fires in its scenario; each runbook used once; 2× L4 scaling test done |
| M6 Benchmark and write-up | Full matrix, 3 repeats, CV < 5%; README and write-up published; resume numbers taken from committed results |

## Risks, open questions and decisions

### Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| vLLM internal API changes break the plugin mid-project | High | Medium | Pin the version; version guard (F6); keep the plugin surface to `QuantizationConfig` and `LinearMethodBase` only |
| TensorRT-LLM engine build fails or runs out of host memory on the L4 box | Medium | High | Rent ≥ 64 GB RAM hosts; build engines once, store them by key; fall back to NVIDIA's prebuilt container examples |
| Custom kernel loses to Marlin by more than 1.25× | Medium | Low | Report the gap with the roofline explanation. The Triton-vs-CUDA comparison and the integration still stand |
| L4 rentals unavailable or interrupted | Medium | Medium | Two providers configured; harness checkpointing (F22); all state in git and object storage |
| Budget overrun | Low | Medium | Watchdog auto-poweroff, ledger, hard stop at $70 |
| Kaggle T4 quota runs out in a heavy week | Medium | Low | Triton interpreter on CPU for logic; batch T4 work into one session per day |
| Scope creep (SGLang, more models, H100) | High | Medium | Anything not in the milestone exit criteria goes to a `later.md` list |

### Open questions

- Include SGLang as a third engine? Default: only if M6 finishes with ≥ 2 days and ≥ $15 left.
- Add Llama-3.1-8B as a second model? It is gated and adds a full benchmark pass. Default: no.
- Run the Triton server's TensorRT-LLM backend, or TensorRT-LLM's own OpenAI server? Default: the Triton server, for parity with the vLLM backend.

### Decision log

| ID | Decision | Rationale | Alternative rejected |
| --- | --- | --- | --- |
| D1 | Qwen2.5-7B-Instruct + its official AWQ checkpoint | Apache-2.0, no gating, fits a 24 GB L4 at FP16 | Llama-3.1-8B (gated, tighter on memory) |
| D2 | Consume the AWQ format; no custom quantizer | Same weights across every engine, so quality deltas come from kernels | Own quantization scheme (confounds the comparison) |
| D3 | Out-of-tree vLLM plugin, not a fork | Upgrades are a version bump; reviewers can read the plugin alone | Fork vLLM (rebase cost, harder to review) |
| D4 | `torch.library.custom_op` with fake implementations | Works with torch.compile and CUDA graphs, both used by vLLM | Raw pybind calls (graph breaks) |
| D5 | Single L4, plus a one-day 2× L4 test for autoscaling | Fits the budget; scale logic proven on CPU with the mock engine | Multi-GPU node for the whole project (3–4× cost) |
| D6 | Open-loop load generation | Closed-loop clients hide queueing collapse and overstate goodput | Closed-loop fixed concurrency only |
| D7 | Gateway retries only before the first token | Retrying after tokens are streamed would duplicate output | Transparent retries (corrupts streams) |
