"""Fused RMSNorm + FP8 (e4m3) activation quantization in Triton.

One program per token. The row is read from DRAM once, normalized, scaled and cast in
registers. Semantics follow the HF Qwen2 RMSNorm (normalize in fp32, round to the input
dtype, multiply by the weight), then:

* dynamic (default): ``scale = max(amax(|y|) / 448, MIN_SCALE)`` per token;
* static: a caller-provided per-tensor ``scale``.

``q = e4m3(clamp(y / scale, -448, 448))`` with an IEEE round-to-nearest division, so the
result is bit-identical to the PyTorch reference given the same ``y`` and ``scale``.

Rounding onto the e4m3 grid is done explicitly (round-to-nearest-even in fp32 bit space)
before the cast, so the cast itself is exact. This matches the hardware ``cvt.rn`` on sm89
and sidesteps the Triton interpreter's fp8 downcast, which drops the mantissa carry into
the exponent (127.9 -> 64 instead of 128).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_E4M3_MAX = 448.0
MIN_SCALE = 1.0 / (FP8_E4M3_MAX * 512.0)


@triton.jit
def _round_to_e4m3_grid(v):
    """Round fp32 ``v`` (|v| <= 448) to the nearest e4m3fn value, ties to even."""
    # Normals: keep 3 of fp32's 23 mantissa bits; the add carries into the exponent.
    bits = v.to(tl.int32, bitcast=True)
    lsb = (bits >> 20) & 1
    normal = ((bits + 0x7FFFF + lsb) & -0x100000).to(tl.float32, bitcast=True)
    # Subnormals (|v| < 2**-6) have a fixed quantum of 2**-9; the 1.5 * 2**23 trick is RNE.
    # Rounded on |v| and re-signed from the sign bit, so -0.0 and tiny negatives keep it.
    t = tl.abs(v) * 512.0
    sub = ((t + 12582912.0) - 12582912.0) * (1.0 / 512.0)
    sub = tl.where(bits < 0, -sub, sub)
    return tl.where(tl.abs(v) < 0.015625, sub, normal)


@triton.jit
def _rmsnorm_fp8_quant_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    scale_ptr,
    H,
    stride_x,
    stride_o,
    eps,
    STATIC_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    x_raw = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0)
    x = x_raw.to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt_rn(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = ((x * rstd).to(x_raw.dtype) * w).to(tl.float32)
    if STATIC_SCALE:
        scale = tl.load(scale_ptr)
    else:
        amax = tl.max(tl.where(mask, tl.abs(y), 0.0), axis=0)
        scale = tl.maximum(amax / 448.0, 1.0 / (448.0 * 512.0))
        tl.store(scale_ptr + row, scale)
    q = _round_to_e4m3_grid(tl.clamp(tl.math.div_rn(y, scale), -448.0, 448.0))
    tl.store(out_ptr + row * stride_o + offs, q.to(tl.float8e4nv), mask=mask)


def rmsnorm_fp8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x [T, H]`` -> ``(q [T, H] float8_e4m3fn, scale)``.

    ``scale`` is ``[T, 1]`` fp32 for dynamic per-token quantization, or the given ``[1]``
    fp32 tensor for static per-tensor quantization.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D [T, H], got {tuple(x.shape)}")
    t, h = x.shape
    if tuple(weight.shape) != (h,):
        raise ValueError(f"weight shape {tuple(weight.shape)} != {(h,)}")
    if x.stride(1) != 1:
        raise ValueError("x must be contiguous along H")
    out = torch.empty((t, h), dtype=torch.float8_e4m3fn, device=x.device)
    static = scale is not None
    if scale is None:
        scale_out = torch.empty((t, 1), dtype=torch.float32, device=x.device)
    else:
        if scale.numel() != 1 or scale.dtype != torch.float32:
            raise ValueError("static scale must be a single fp32 element")
        scale_out = scale
    block = triton.next_power_of_2(h)
    num_warps = 4 if block <= 2048 else 8 if block <= 8192 else 16
    _rmsnorm_fp8_quant_kernel[(t,)](
        x,
        weight,
        out,
        scale_out,
        h,
        x.stride(0),
        out.stride(0),
        eps,
        STATIC_SCALE=static,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out, scale_out
