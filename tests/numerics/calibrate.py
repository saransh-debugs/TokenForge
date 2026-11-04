"""Calibrate the tolerance constant ``c`` against the vendor fp16 GEMM, then freeze it.

    python -m tests.numerics.calibrate --device cpu --write    # ATen fp16 matmul
    TOKENFORGE_SHAPE_SUITE=full python -m tests.numerics.calibrate --device cuda --write

``c`` is the worst ``max|y_vendor - y_ref| / error_scale`` over every (shape, M, seed) the
numerics tests use plus extra seeds, with ``y_ref`` an fp64 GEMM on the same fp16 inputs,
rounded up to two significant figures.
"""

from __future__ import annotations

import argparse
import json
import math
import platform

import torch

from tests.numerics import cases
from tests.numerics.shapes import linear_shapes, m_values, suite
from tests.numerics.tolerance import TOLERANCES_PATH, error_scale
from tokenforge.kernels.pack import dequantize_tf


def calibrate(device: str) -> dict[str, object]:
    shapes = linear_shapes()
    worst = 0.0
    worst_case: tuple[object, ...] = ()
    n_cases = 0
    for shape in shapes:
        for seed in cases.CALIBRATION_SEEDS:
            w16 = dequantize_tf(cases.layer_weights(shape.n, shape.k, seed))
            w64 = w16.to(torch.float64)
            w_dev = w16.to(device)
            for m in m_values():
                x16 = cases.activations(m, shape.k, seed)
                ref = x16.to(torch.float64) @ w64.t()
                y = (x16.to(device) @ w_dev.t()).cpu().to(torch.float64)
                ratio = float((y - ref).abs().max()) / error_scale(x16, w16)
                n_cases += 1
                if ratio > worst:
                    worst, worst_case = ratio, (shape.layer, shape.n, shape.k, m, seed)
    exp = math.floor(math.log10(worst))
    c = math.ceil(worst / 10 ** (exp - 1)) * 10 ** (exp - 1)
    return {
        "c": round(c, 6),
        "observed_max": worst,
        "worst_case": list(worst_case),
        "cases": n_cases,
        "suite": suite(),
        "vendor_gemm": "cublas" if device == "cuda" else "aten-cpu-fp16",
        "device_name": torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
        "torch": torch.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--write", action="store_true", help="freeze the result in tolerances.json")
    args = ap.parse_args()
    result = calibrate(args.device)
    print(json.dumps(result, indent=2))
    if args.write:
        data = json.loads(TOLERANCES_PATH.read_text()) if TOLERANCES_PATH.exists() else {}
        data.setdefault("w4a16", {})[args.device] = result
        TOLERANCES_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
