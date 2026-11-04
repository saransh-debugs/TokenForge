"""W4A16 GEMM / GEMV and full-weight dequantization in Triton.

One kernel covers M = 1..128 with three code paths selected at compile time:

* ``BLOCK_M == 1``: GEMV. Products accumulate into a ``[BLOCK_N, BLOCK_K]`` fp32 tile and are
  reduced once at the end, so the K loop is pure load + unpack + FMA.
* ``1 < BLOCK_M < 16``: broadcast FMA with a per-iteration K reduction.
* ``BLOCK_M >= 16``: ``tl.dot`` on fp16 operands with fp32 accumulation.

Weights are dequantized in registers. ``BLOCK_K`` divides the group size, so each K block
reads one scale and one zero-point per output column. Split-K writes fp32 partials to a
``[SPLIT_K, M, N]`` workspace that the host reduces in a fixed order: results are
deterministic, which keeps CUDA-graph and eager outputs bit-identical (F5).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tokenforge.kernels.config import GemmConfig


@triton.jit
def _unpack_k(qw, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """int32 [BLOCK_N, BLOCK_K // 8] packed along K -> int32 [BLOCK_N, BLOCK_K] in [0, 15]."""
    j = tl.arange(0, 8)
    shifts = ((j % 2) * 4 + j // 2) * 4  # AWQ_REVERSE_ORDER * 4
    q = (qw[:, :, None] >> shifts[None, None, :]) & 0xF
    return tl.reshape(q, (BLOCK_N, BLOCK_K))


@triton.jit
def _load_group(s_ptr, z_ptr, g, offs_n, mask_n, stride_sg, stride_zg):
    """Scales (fp16) and zero-points (int32) for group ``g`` and columns ``offs_n``."""
    s = tl.load(s_ptr + g * stride_sg + offs_n, mask=mask_n, other=0.0)
    zw = tl.load(z_ptr + g * stride_zg + offs_n // 8, mask=mask_n, other=0)
    c = offs_n % 8
    z = (zw >> (((c % 2) * 4 + c // 2) * 4)) & 0xF
    return s, z


@triton.jit
def _w4a16_gemm_kernel(
    x_ptr,
    qw_ptr,
    s_ptr,
    z_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_qn,
    stride_sg,
    stride_zg,
    stride_os,
    stride_om,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_k8 = tl.arange(0, BLOCK_K // 8)
    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    qw_ptrs = qw_ptr + offs_n[:, None] * stride_qn + offs_k8[None, :]

    if BLOCK_M == 1:
        acc2 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_kb = K // BLOCK_K
    for kb in tl.range(pid_k, num_kb, SPLIT_K):
        k0 = kb * BLOCK_K
        x = tl.load(x_ptrs + k0 * stride_xk, mask=mask_m[:, None], other=0.0)
        qw = tl.load(qw_ptrs + k0 // 8, mask=mask_n[:, None], other=0)
        q = _unpack_k(qw, BLOCK_N, BLOCK_K)
        s, z = _load_group(s_ptr, z_ptr, k0 // GROUP_SIZE, offs_n, mask_n, stride_sg, stride_zg)
        w = (q - z[:, None]).to(tl.float16) * s[:, None]  # [BLOCK_N, BLOCK_K], one rounding
        if BLOCK_M == 1:
            acc2 += x.to(tl.float32) * w.to(tl.float32)
        elif BLOCK_M < 16:
            prod = x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :]
            acc += tl.sum(prod, axis=2)
        else:
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)

    if BLOCK_M == 1:
        acc = tl.reshape(tl.sum(acc2, axis=1), (1, BLOCK_N))

    out_ptrs = out_ptr + pid_k * stride_os + offs_m[:, None] * stride_om + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        if HAS_BIAS:
            b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
            acc += b.to(tl.float32)[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)
    else:
        tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _w4a16_dequant_kernel(
    qw_ptr,
    s_ptr,
    z_ptr,
    out_ptr,
    N,
    K,
    stride_qn,
    stride_sg,
    stride_zg,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k8 = pid_k * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    qw = tl.load(qw_ptr + offs_n[:, None] * stride_qn + offs_k8[None, :], mask=mask_n[:, None])
    q = _unpack_k(qw, BLOCK_N, BLOCK_K)
    s, z = _load_group(
        s_ptr, z_ptr, (pid_k * BLOCK_K) // GROUP_SIZE, offs_n, mask_n, stride_sg, stride_zg
    )
    w = (q - z[:, None]).to(tl.float16) * s[:, None]
    tl.store(out_ptr + offs_n[:, None] * stride_on + offs_k[None, :], w, mask=mask_n[:, None])


def _check_inputs(
    x: torch.Tensor,
    qweight_t: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> tuple[int, int, int]:
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D [M, K], got shape {tuple(x.shape)}")
    if x.dtype != torch.float16:
        raise TypeError(f"x must be fp16, got {x.dtype}")
    m, k = x.shape
    n = qweight_t.shape[0]
    if qweight_t.shape[1] * 8 != k:
        raise ValueError(f"x has K={k} but qweight_t encodes K={qweight_t.shape[1] * 8}")
    if k % group_size != 0:
        raise ValueError(f"K={k} is not a multiple of group_size={group_size}")
    if tuple(scales.shape) != (k // group_size, n):
        raise ValueError(f"scales shape {tuple(scales.shape)} != {(k // group_size, n)}")
    if tuple(qzeros.shape) != (k // group_size, n // 8):
        raise ValueError(f"qzeros shape {tuple(qzeros.shape)} != {(k // group_size, n // 8)}")
    for name, t in (("qweight_t", qweight_t), ("scales", scales), ("qzeros", qzeros)):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if x.stride(1) != 1:
        raise ValueError("x must be contiguous along K (stride(1) == 1)")
    return m, n, k


def w4a16_gemm(
    x: torch.Tensor,
    qweight_t: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    config: GemmConfig,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x [M, K] fp16`` @ dequant(W)^T (+ bias) -> ``[M, N] fp16``."""
    m, n, k = _check_inputs(x, qweight_t, scales, qzeros, group_size)
    config.validate(group_size)
    if bias is not None and tuple(bias.shape) != (n,):
        raise ValueError(f"bias shape {tuple(bias.shape)} != {(n,)}")

    split_k = min(config.split_k, k // config.block_k)
    grid = (triton.cdiv(n, config.block_n), triton.cdiv(m, config.block_m), split_k)
    if split_k == 1:
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        stride_os = 0
    else:
        out = torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
        stride_os = out.stride(0)
    _w4a16_gemm_kernel[grid](
        x,
        qweight_t,
        scales,
        qzeros,
        bias if bias is not None else scales,  # dummy pointer, never read
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        qweight_t.stride(0),
        scales.stride(0),
        qzeros.stride(0),
        stride_os,
        out.stride(-2),
        GROUP_SIZE=group_size,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        SPLIT_K=split_k,
        HAS_BIAS=bias is not None,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    if split_k == 1:
        return out
    reduced = out.sum(dim=0)
    if bias is not None:
        reduced += bias.to(torch.float32)
    return reduced.to(x.dtype)


def w4a16_dequant(
    qweight_t: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    block_n: int = 64,
    block_k: int = 128,
) -> torch.Tensor:
    """TokenForge layout -> fp16 ``[N, K]``, the ``nn.Linear.weight`` layout."""
    n = qweight_t.shape[0]
    k = qweight_t.shape[1] * 8
    block_k = min(block_k, group_size)
    out = torch.empty((n, k), dtype=torch.float16, device=qweight_t.device)
    grid = (triton.cdiv(n, block_n), k // block_k)
    _w4a16_dequant_kernel[grid](
        qweight_t,
        scales,
        qzeros,
        out,
        n,
        k,
        qweight_t.stride(0),
        scales.stride(0),
        qzeros.stride(0),
        out.stride(0),
        GROUP_SIZE=group_size,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out
