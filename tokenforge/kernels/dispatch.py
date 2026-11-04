"""Pure-Python kernel dispatch.

Two decisions, made at different times:

* :func:`layer_support` runs once per layer at weight-load time. An unsupported layer keeps
  vLLM's own AWQ weights and method (``awq_marlin`` on sm80+, ``awq`` below), and each such
  layer increments ``tokenforge_fallback_total{reason}``.
* :func:`select` runs per call on (M, backend) only. It reads shapes, never tensor values,
  so it is safe inside CUDA-graph capture and opaque to torch.compile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Literal, get_args

from tokenforge.kernels.config import MAX_TRITON_M, m_bucket
from tokenforge.kernels.metrics import FALLBACKS, KERNEL_CALLS
from tokenforge.kernels.pack import PACK_FACTOR, SUPPORTED_GROUP_SIZES

Backend = Literal["triton", "cuda", "marlin"]
Kernel = Literal["triton_gemm", "cuda_gemv", "dequant_gemm", "torch_reference"]
FallbackReason = Literal[
    "kill_switch",
    "unsupported_dtype",
    "unsupported_group_size",
    "unsupported_shape",
    "unsupported_arch",
    "cuda_ext_unavailable",
]

BACKENDS: Final[tuple[str, ...]] = get_args(Backend)
BACKEND_ENV: Final = "TOKENFORGE_BACKEND"
MIN_SM: Final = 75
CUDA_GEMV_MAX_M: Final = 4


def backend_from_env() -> Backend:
    value = os.environ.get(BACKEND_ENV, "triton").strip().lower()
    if value not in BACKENDS:
        raise ValueError(f"{BACKEND_ENV}={value!r} is not one of {BACKENDS}")
    return value  # type: ignore[return-value]


def layer_support(
    n: int, k: int, group_size: int, dtype: str, sm: int | None, backend: Backend
) -> FallbackReason | None:
    """``None`` if TokenForge kernels can serve this layer, else the fallback reason."""
    if backend == "marlin":
        return "kill_switch"
    if dtype not in ("float16", "torch.float16"):
        return "unsupported_dtype"
    if group_size not in SUPPORTED_GROUP_SIZES:
        return "unsupported_group_size"
    if k % group_size != 0 or k % PACK_FACTOR != 0 or n % PACK_FACTOR != 0 or n <= 0:
        return "unsupported_shape"
    if sm is not None and sm < MIN_SM:
        return "unsupported_arch"
    return None


def record_layer_fallback(reason: FallbackReason) -> None:
    FALLBACKS.inc(reason=reason)


@dataclass(frozen=True)
class Decision:
    kernel: Kernel
    bucket: str
    fallback: FallbackReason | None = None


def bucket_label(m: int) -> str:
    return f">{MAX_TRITON_M}" if m > MAX_TRITON_M else str(m_bucket(m))


def select(
    m: int,
    *,
    backend: Backend,
    on_gpu: bool,
    triton_ok: bool,
    cuda_ext_available: bool,
) -> Decision:
    """Kernel for one call with ``m`` rows. The layer has already passed :func:`layer_support`."""
    if backend == "marlin":
        raise ValueError("backend 'marlin' is resolved per layer, not per call")
    bucket = bucket_label(m)
    if not triton_ok:
        return Decision("torch_reference", bucket)
    if m > MAX_TRITON_M:
        return Decision("dequant_gemm", bucket)
    if backend == "cuda" and m <= CUDA_GEMV_MAX_M:
        if on_gpu and cuda_ext_available:
            return Decision("cuda_gemv", bucket)
        return Decision("triton_gemm", bucket, fallback="cuda_ext_unavailable")
    return Decision("triton_gemm", bucket)


def record(decision: Decision) -> None:
    KERNEL_CALLS.inc(kernel=decision.kernel, bucket=decision.bucket)
    if decision.fallback is not None:
        FALLBACKS.inc(reason=decision.fallback)
