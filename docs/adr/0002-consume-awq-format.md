# ADR 0002: Consume the AWQ GEMM format; no custom quantizer

Status: accepted · Date: 2026-11-04 · Decision log: D2

## Context
Quality deltas between engines are only meaningful if every engine runs the same weights.

## Decision
Kernels consume the standard AWQ GEMM layout (int4 packed 8 per int32, group 128, fp16
scales, packed zero-points). `pack_awq` repacks qweight once at load time into a K-packed
layout with the same nibble interleave; scales and zero-points are used as-is.
`awq_quantize` exists only to generate test fixtures.

## Consequences
The same checkpoint runs through awq_marlin, TensorRT-LLM INT4-AWQ and TokenForge, so a
perplexity or GSM8K delta is attributable to kernels. Dequantization rounds once
(`fp16(q - z) * s`) in every implementation, which is what makes bit-level comparison
against the reference possible.

## Alternatives considered
A custom quantization scheme would confound kernel quality with quantization quality.
