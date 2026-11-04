# ADR 0001: Qwen2.5-7B-Instruct with its official AWQ checkpoint

Status: accepted · Date: 2026-11-04 · Decision log: D1

## Context
The project needs one 7B model that every engine (vLLM, TensorRT-LLM) and precision (FP16,
FP8, INT4-AWQ) can serve on a single 24 GB L4, plus a small sibling for T4 and CI.

## Decision
Qwen2.5-7B-Instruct and Qwen/Qwen2.5-7B-Instruct-AWQ. Dev model: Qwen2.5-0.5B-Instruct
(-AWQ), same architecture family. Revisions are pinned by commit SHA in
`infra/versions.lock` at first download.

## Consequences
Apache-2.0, no gating, so CI and rented boxes download without tokens. FP16 weights (~15 GB)
fit with room for KV cache. Every shape in the kernel suite comes from these two models.

## Alternatives considered
Llama-3.1-8B: gated, tighter on memory at FP16, and a second full benchmark pass.
