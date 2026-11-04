"""Deterministic test cases shared by the numerics tests and the tolerance calibration.

Calibration covers a superset of the cases the tests run (same generator, same seeds), so
the frozen constant is at least the vendor GEMM's error ratio on every tested input.
"""

from __future__ import annotations

import functools
import zlib

import torch

from tokenforge.kernels.pack import W4A16Weights, awq_quantize, pack_awq

GROUP_SIZE = 128
TEST_SEEDS = (0,)
CALIBRATION_SEEDS = (0, 1, 2, 3, 4)


def _seed(*parts: object) -> int:
    return zlib.crc32(repr(parts).encode())


@functools.lru_cache(maxsize=16)
def layer_weights(n: int, k: int, seed: int, group_size: int = GROUP_SIZE) -> W4A16Weights:
    gen = torch.Generator().manual_seed(_seed("w", n, k, seed, group_size))
    weight = torch.randn((n, k), generator=gen) * 0.02
    return pack_awq(*awq_quantize(weight, group_size), group_size)


def activations(m: int, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(_seed("x", m, k, seed))
    return torch.randn((m, k), generator=gen).to(torch.float16)


def bias(n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(_seed("b", n, seed))
    return (torch.randn((n,), generator=gen) * 0.1).to(torch.float16)
