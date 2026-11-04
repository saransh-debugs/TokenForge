"""AWQ GEMM-format packing and the TokenForge kernel layout.

AWQ GEMM layout (AutoAWQ ``WQLinear_GEMM``), the on-disk format every engine reads:

* ``qweight``: int32 ``[K, N // 8]``. Eight 4-bit weights per word, packed along N.
* ``qzeros``:  int32 ``[K // G, N // 8]``. Zero-points, packed the same way.
* ``scales``:  fp16  ``[K // G, N]``.

Inside a word, nibble ``i`` (bits ``4i .. 4i+3``) holds column offset ``AWQ_ORDER[i]``.
The interleave puts offsets ``(2j, 2j+1)`` in nibbles ``(j, j+4)``, so a single LOP3 with
the ``0x000F000F`` mask plus the ``0x6400`` magic number yields an fp16x2 of two adjacent
values.

TokenForge layout (what the kernels read):

* ``qweight_t``: int32 ``[N, K // 8]``. Packed along **K**, same nibble interleave. A thread
  that owns output column ``n`` streams contiguous K, so a 128-bit load feeds 32 MACs and the
  CUDA GEMV can reduce over K with warp shuffles.
* ``scales`` and ``qzeros`` are kept in AWQ layout unchanged: both are read once per
  (N tile, K group), and the AWQ layout makes that read coalesced along N.

Dequantization is ``w[k, n] = fp16(q[k, n] - z[k // G, n]) * s[k // G, n]``: the integer
difference is exact in fp16, so every implementation rounds exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

AWQ_ORDER: tuple[int, ...] = (0, 2, 4, 6, 1, 3, 5, 7)
"""Nibble ``i`` of a packed word holds offset ``AWQ_ORDER[i]``."""

AWQ_REVERSE_ORDER: tuple[int, ...] = (0, 4, 1, 5, 2, 6, 3, 7)
"""Offset ``j`` lives in nibble ``AWQ_REVERSE_ORDER[j]``. Equals ``(j % 2) * 4 + j // 2``."""

PACK_FACTOR = 8
SUPPORTED_GROUP_SIZES: tuple[int, ...] = (32, 64, 128)


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack unsigned 4-bit values along the last dim into int32 words, AWQ interleave."""
    if values.shape[-1] % PACK_FACTOR != 0:
        raise ValueError(f"last dim {values.shape[-1]} is not a multiple of {PACK_FACTOR}")
    if values.numel() and (int(values.min()) < 0 or int(values.max()) > 15):
        raise ValueError("int4 values must lie in [0, 15]")
    *lead, n = values.shape
    v = values.to(torch.int64).reshape(*lead, n // PACK_FACTOR, PACK_FACTOR)
    word = torch.zeros(v.shape[:-1], dtype=torch.int64, device=values.device)
    for nibble, offset in enumerate(AWQ_ORDER):
        word |= v[..., offset] << (4 * nibble)
    # int64 -> int32 wraps modulo 2**32, which is exactly the two's-complement bit pattern.
    return word.to(torch.int32)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_int4`. Returns int32 values in [0, 15]."""
    if packed.dtype != torch.int32:
        raise TypeError(f"packed tensor must be int32, got {packed.dtype}")
    shifts = torch.tensor(
        [4 * AWQ_REVERSE_ORDER[j] for j in range(PACK_FACTOR)],
        dtype=torch.int32,
        device=packed.device,
    )
    values = (packed.unsqueeze(-1) >> shifts) & 0xF
    return values.reshape(*packed.shape[:-1], packed.shape[-1] * PACK_FACTOR)


# --------------------------------------------------------------------------- AWQ format


def awq_pack(intweight: torch.Tensor, zeros: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``intweight [K, N]`` and ``zeros [K // G, N]`` (both in [0, 15]) into AWQ format."""
    return pack_int4(intweight), pack_int4(zeros)


def awq_unpack(qweight: torch.Tensor) -> torch.Tensor:
    """AWQ ``qweight [K, N // 8]`` -> int32 ``[K, N]``. Also works for ``qzeros``."""
    return unpack_int4(qweight)


def awq_quantize(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min-max asymmetric int4 quantization of an ``nn.Linear`` weight ``[N, K]``.

    Produces a checkpoint in AWQ GEMM format with the same rounding rules as AutoAWQ's
    ``pseudo_quantize_tensor``. This is a fixture generator for tests and microbenchmarks;
    production runs consume the official AWQ checkpoint (ADR D2).

    Returns ``(qweight [K, N//8] int32, qzeros [K//G, N//8] int32, scales [K//G, N] fp16)``.
    """
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"K={k} is not a multiple of group_size={group_size}")
    w = weight.to(torch.float32).reshape(n, k // group_size, group_size)
    max_val = w.amax(dim=-1, keepdim=True)
    min_val = w.amin(dim=-1, keepdim=True)
    scales = ((max_val - min_val).clamp(min=1e-5) / 15).to(torch.float16).to(torch.float32)
    zeros = (-torch.round(min_val / scales)).clamp(0, 15)
    q = (torch.round(w / scales) + zeros).clamp(0, 15)
    intweight = q.reshape(n, k).t().contiguous().to(torch.int32)  # [K, N]
    zeros_kn = zeros.squeeze(-1).t().contiguous().to(torch.int32)  # [K//G, N]
    scales_kn = scales.squeeze(-1).t().contiguous().to(torch.float16)  # [K//G, N]
    qweight, qzeros = awq_pack(intweight, zeros_kn)
    return qweight, qzeros, scales_kn


def awq_dequantize(
    qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    """AWQ format -> fp16 ``[K, N]`` with the single-rounding rule shared by every kernel."""
    q = awq_unpack(qweight)
    z = awq_unpack(qzeros).repeat_interleave(group_size, dim=0)
    s = scales.repeat_interleave(group_size, dim=0)
    return (q - z).to(torch.float16) * s


# --------------------------------------------------------------------------- kernel layout


@dataclass(frozen=True)
class W4A16Weights:
    """One linear layer's weights in TokenForge layout."""

    qweight_t: torch.Tensor  # int32 [N, K // 8], packed along K
    scales: torch.Tensor  # fp16 [K // G, N]
    qzeros: torch.Tensor  # int32 [K // G, N // 8], AWQ layout
    group_size: int

    @property
    def n(self) -> int:
        return int(self.qweight_t.shape[0])

    @property
    def k(self) -> int:
        return int(self.qweight_t.shape[1]) * PACK_FACTOR


def repack_awq_to_tf(qweight: torch.Tensor) -> torch.Tensor:
    """AWQ ``qweight [K, N // 8]`` -> TokenForge ``qweight_t [N, K // 8]``."""
    return pack_int4(awq_unpack(qweight).t().contiguous())


def repack_tf_to_awq(qweight_t: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`repack_awq_to_tf`."""
    return pack_int4(unpack_int4(qweight_t).t().contiguous())


def validate_awq(
    qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int
) -> None:
    """Raise ``ValueError`` with a precise reason if tensors are not a valid AWQ layer."""
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("qweight and qzeros must be int32")
    if scales.dtype != torch.float16:
        raise ValueError(f"scales must be fp16, got {scales.dtype}")
    if group_size not in SUPPORTED_GROUP_SIZES:
        raise ValueError(f"group_size {group_size} not in {SUPPORTED_GROUP_SIZES}")
    k, n8 = qweight.shape
    n = n8 * PACK_FACTOR
    if k % group_size != 0:
        raise ValueError(f"K={k} is not a multiple of group_size={group_size}")
    if tuple(qzeros.shape) != (k // group_size, n8):
        raise ValueError(f"qzeros shape {tuple(qzeros.shape)} != {(k // group_size, n8)}")
    if tuple(scales.shape) != (k // group_size, n):
        raise ValueError(f"scales shape {tuple(scales.shape)} != {(k // group_size, n)}")


def pack_awq(
    qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int
) -> W4A16Weights:
    """Load-time conversion of one AWQ layer into the TokenForge layout."""
    validate_awq(qweight, qzeros, scales, group_size)
    return W4A16Weights(
        qweight_t=repack_awq_to_tf(qweight),
        scales=scales.contiguous(),
        qzeros=qzeros.contiguous(),
        group_size=group_size,
    )


def unpack_to_awq(weights: W4A16Weights) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack_awq`, for tests: returns ``(qweight, qzeros, scales)``."""
    return repack_tf_to_awq(weights.qweight_t), weights.qzeros, weights.scales


def dequantize_tf(weights: W4A16Weights) -> torch.Tensor:
    """TokenForge layout -> fp16 ``[N, K]`` (the ``nn.Linear.weight`` layout)."""
    q = unpack_int4(weights.qweight_t)  # [N, K]
    z = unpack_int4(weights.qzeros).repeat_interleave(weights.group_size, dim=0).t()
    s = weights.scales.repeat_interleave(weights.group_size, dim=0).t()
    return (q - z).to(torch.float16) * s
