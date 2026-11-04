"""Derived numerical tolerance for W4A16 kernels.

With fp32 accumulation the error grows like sqrt(K), so

    atol = c * sqrt(K) * eps_fp16 * max|x| * max|w|

``c`` is calibrated once per device class against the vendor fp16 GEMM (cuBLAS on CUDA,
ATen's fp16 matmul on CPU) by ``python -m tests.numerics.calibrate`` and frozen in
``tolerances.json``. A kernel that needs a looser constant than the vendor GEMM fails.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

EPS_FP16 = float(torch.finfo(torch.float16).eps)  # 2**-10
TOLERANCES_PATH = Path(__file__).with_name("tolerances.json")


def error_scale(x: torch.Tensor, w: torch.Tensor) -> float:
    """``sqrt(K) * eps_fp16 * max|x| * max|w|`` over finite entries."""
    k = x.shape[-1]
    xf = x.to(torch.float64)
    wf = w.to(torch.float64)
    xmax = float(xf[torch.isfinite(xf)].abs().max()) if torch.isfinite(xf).any() else 0.0
    wmax = float(wf.abs().max()) if wf.numel() else 0.0
    return math.sqrt(k) * EPS_FP16 * xmax * wmax


def load_constant(device_type: str) -> float:
    data = json.loads(TOLERANCES_PATH.read_text())
    entry = data["w4a16"].get(device_type)
    if entry is None:
        raise LookupError(
            f"no calibrated tolerance for device '{device_type}'; run "
            f"`python -m tests.numerics.calibrate --device {device_type}` once and commit it"
        )
    return float(entry["c"])


def atol(x: torch.Tensor, w: torch.Tensor, device_type: str) -> float:
    return load_constant(device_type) * error_scale(x, w)
