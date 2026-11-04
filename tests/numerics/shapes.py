"""The shape suite shared by numerics tests, calibration and microbenchmarks."""

from __future__ import annotations

import os
from typing import NamedTuple


class LinearShape(NamedTuple):
    model: str
    layer: str
    n: int
    k: int


# vLLM fuses q/k/v and gate/up at TP=1, so these are the shapes the plugin sees.
QWEN25_05B = [
    LinearShape("qwen2.5-0.5b", "qkv_proj", 896 + 2 * 128, 896),
    LinearShape("qwen2.5-0.5b", "o_proj", 896, 896),
    LinearShape("qwen2.5-0.5b", "gate_up_proj", 2 * 4864, 896),
    LinearShape("qwen2.5-0.5b", "down_proj", 896, 4864),
]
QWEN25_7B = [
    LinearShape("qwen2.5-7b", "qkv_proj", 3584 + 2 * 512, 3584),
    LinearShape("qwen2.5-7b", "o_proj", 3584, 3584),
    LinearShape("qwen2.5-7b", "gate_up_proj", 2 * 18944, 3584),
    LinearShape("qwen2.5-7b", "down_proj", 3584, 18944),
]
# N not divisible by any tile size (still a multiple of 8, as AWQ packing requires).
RAGGED = [
    LinearShape("synthetic", "n_ragged_small", 200, 256),
    LinearShape("synthetic", "n_ragged", 1000, 1024),
]

M_SUITE = (1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 127, 128, 129)
M_SUITE_CPU = (1, 3, 8, 16, 31, 129)


def suite() -> str:
    """``cpu`` (interpreter-sized) unless ``TOKENFORGE_SHAPE_SUITE=full``."""
    return os.environ.get("TOKENFORGE_SHAPE_SUITE", "cpu")


def linear_shapes() -> list[LinearShape]:
    if suite() == "full":
        return QWEN25_05B + QWEN25_7B + RAGGED
    return QWEN25_05B + RAGGED[:1]


def m_values() -> tuple[int, ...]:
    return M_SUITE if suite() == "full" else M_SUITE_CPU
