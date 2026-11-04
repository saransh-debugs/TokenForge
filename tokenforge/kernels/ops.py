"""``torch.library`` registration of the TokenForge kernels.

Each op is a ``custom_op`` with a fake (meta) implementation, so torch.compile traces it as a
single opaque node and CUDA graphs capture it like any other kernel launch (ADR D4). The real
implementations never read tensor values on the host and never synchronize.

Ops:

* ``tokenforge::w4a16_linear(x, qweight_t, scales, qzeros, bias, group_size, backend)``
* ``tokenforge::w4a16_dequant(qweight_t, scales, qzeros, group_size)``
* ``tokenforge::rmsnorm_fp8_quant(x, weight, eps, scale)``
"""

from __future__ import annotations

import functools
from typing import Any

import torch
import torch.nn.functional as F

from tokenforge.kernels import autotune, dispatch
from tokenforge.kernels.config import m_bucket
from tokenforge.kernels.device import gpu_identity, triton_runnable
from tokenforge.kernels.pack import W4A16Weights
from tokenforge.kernels.reference import rmsnorm_fp8_reference, w4a16_torch


def cuda_ext_available() -> bool:
    """The compiled CUDA GEMV extension (``tokenforge.kernels._C``) is importable."""
    return _cuda_ext() is not None


@functools.cache
def _cuda_ext() -> Any:
    try:
        from tokenforge.kernels import _C  # type: ignore[attr-defined]
    except ImportError:
        return None
    return _C


@functools.cache
def _gpu_id(device: torch.device) -> str:
    return gpu_identity(device)


@torch.library.custom_op("tokenforge::w4a16_linear", mutates_args=())
def w4a16_linear(
    x: torch.Tensor,
    qweight_t: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    backend: str,
) -> torch.Tensor:
    k = x.shape[-1]
    n = qweight_t.shape[0]
    out_shape = (*x.shape[:-1], n)
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    if m == 0:
        return x.new_empty(out_shape)
    if backend not in ("triton", "cuda"):
        raise ValueError(f"backend must be 'triton' or 'cuda' at op level, got {backend!r}")

    decision = dispatch.select(
        m,
        backend=backend,  # type: ignore[arg-type]
        on_gpu=x.is_cuda,
        triton_ok=triton_runnable(x2),
        cuda_ext_available=cuda_ext_available(),
    )
    dispatch.record(decision)

    if decision.kernel == "torch_reference":
        w = W4A16Weights(qweight_t, scales, qzeros, group_size)
        y = w4a16_torch(x2, w, bias)
    elif decision.kernel == "dequant_gemm":
        from tokenforge.kernels.triton.w4a16 import w4a16_dequant

        y = F.linear(x2, w4a16_dequant(qweight_t, scales, qzeros, group_size), bias)
    elif decision.kernel == "cuda_gemv":
        ext = _cuda_ext()
        y = ext.w4a16_gemv(x2, qweight_t, scales, qzeros, bias, group_size)
    else:
        from tokenforge.kernels.triton.w4a16 import w4a16_gemm

        cfg = autotune.lookup(_gpu_id(x.device), m_bucket(m), n, k, group_size)
        y = w4a16_gemm(x2, qweight_t, scales, qzeros, group_size, cfg, bias)
    return y.reshape(out_shape)


@w4a16_linear.register_fake
def _w4a16_linear_fake(
    x: torch.Tensor,
    qweight_t: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    backend: str,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], qweight_t.shape[0]))


@torch.library.custom_op("tokenforge::w4a16_dequant", mutates_args=())
def w4a16_dequant(
    qweight_t: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int
) -> torch.Tensor:
    if triton_runnable(qweight_t):
        from tokenforge.kernels.triton.w4a16 import w4a16_dequant as _dequant

        return _dequant(qweight_t, scales, qzeros, group_size)
    from tokenforge.kernels.pack import dequantize_tf

    return dequantize_tf(W4A16Weights(qweight_t, scales, qzeros, group_size))


@w4a16_dequant.register_fake
def _w4a16_dequant_fake(
    qweight_t: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int
) -> torch.Tensor:
    return qweight_t.new_empty((qweight_t.shape[0], qweight_t.shape[1] * 8), dtype=torch.float16)


@torch.library.custom_op("tokenforge::rmsnorm_fp8_quant", mutates_args=())
def rmsnorm_fp8_quant(
    x: torch.Tensor, weight: torch.Tensor, eps: float, scale: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    h = x.shape[-1]
    x2 = x.reshape(-1, h)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    if triton_runnable(x2):
        from tokenforge.kernels.triton.rmsnorm_fp8 import rmsnorm_fp8_quant as _kernel

        q, s = _kernel(x2, weight, eps, scale)
    else:
        q, s = rmsnorm_fp8_reference(x2, weight, eps, scale)
    if scale is not None:
        s = s.clone()  # custom ops may not return an alias of an input
    return q.reshape(x.shape), s.reshape((*x.shape[:-1], 1)) if scale is None else s


@rmsnorm_fp8_quant.register_fake
def _rmsnorm_fp8_quant_fake(
    x: torch.Tensor, weight: torch.Tensor, eps: float, scale: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    q = x.new_empty(x.shape, dtype=torch.float8_e4m3fn)
    if scale is None:
        return q, x.new_empty((*x.shape[:-1], 1), dtype=torch.float32)
    return q, torch.empty_like(scale)
