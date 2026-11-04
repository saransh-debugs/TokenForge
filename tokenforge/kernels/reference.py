"""High-precision references used by the numerics tiers (T1, T2)."""

from __future__ import annotations

import torch

from tokenforge.kernels.pack import W4A16Weights, dequantize_tf

FP8_E4M3_MAX = 448.0
MIN_SCALE = 1.0 / (FP8_E4M3_MAX * 512.0)


def w4a16_reference(
    x: torch.Tensor, weights: W4A16Weights, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """fp64 ``x @ W^T (+ bias)`` on the fp16-dequantized weights. Returns fp64."""
    w = dequantize_tf(weights).to(torch.float64)
    y = x.to(torch.float64) @ w.t()
    if bias is not None:
        y = y + bias.to(torch.float64)
    return y


def w4a16_torch(
    x: torch.Tensor, weights: W4A16Weights, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Device-agnostic fp16 path (dequant + matmul) used when Triton cannot run."""
    w = dequantize_tf(weights).to(x.device)
    y = x.to(torch.float32) @ w.to(torch.float32).t()
    if bias is not None:
        y = y + bias.to(torch.float32)
    return y.to(x.dtype)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF Qwen2RMSNorm semantics: fp32 normalize, round to input dtype, then scale."""
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(x.dtype) * weight


def fp8_quant_reference(
    y: torch.Tensor, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic (or static per-tensor) saturating e4m3 quantization of ``y``."""
    yf = y.to(torch.float32)
    if scale is None:
        scale = (yf.abs().amax(dim=-1, keepdim=True) / FP8_E4M3_MAX).clamp(min=MIN_SCALE)
    q = (yf / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def rmsnorm_fp8_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return fp8_quant_reference(rmsnorm(x, weight, eps), scale)
