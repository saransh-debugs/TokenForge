"""L1 kernels: W4A16 GEMM/GEMV, fused RMSNorm + FP8 quant, AWQ repacking, dispatch.

Importing :mod:`tokenforge.kernels.ops` registers the ``torch.ops.tokenforge`` namespace.
"""

from tokenforge.kernels.pack import (
    AWQ_ORDER,
    AWQ_REVERSE_ORDER,
    W4A16Weights,
    awq_dequantize,
    awq_quantize,
    dequantize_tf,
    pack_awq,
    unpack_to_awq,
)

__all__ = [
    "AWQ_ORDER",
    "AWQ_REVERSE_ORDER",
    "W4A16Weights",
    "awq_dequantize",
    "awq_quantize",
    "dequantize_tf",
    "pack_awq",
    "unpack_to_awq",
]
