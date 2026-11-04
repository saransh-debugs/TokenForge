"""T1: W4A16 kernel vs an fp64 reference on the fp16-dequantized weights.

Runs on CUDA when present, otherwise under the Triton interpreter on CPU. The CPU tier uses
interpreter-sized shapes; ``TOKENFORGE_SHAPE_SUITE=full`` runs every Qwen2.5 0.5B/7B layer
and every M in the suite (GPU tiers).
"""

from __future__ import annotations

import pytest
import torch

import tokenforge.kernels.ops  # noqa: F401  (registers torch.ops.tokenforge)
from tests.numerics import cases
from tests.numerics.shapes import linear_shapes, m_values
from tests.numerics.tolerance import atol
from tokenforge.kernels.config import GemmConfig
from tokenforge.kernels.metrics import KERNEL_CALLS
from tokenforge.kernels.pack import W4A16Weights, awq_pack, dequantize_tf, pack_awq
from tokenforge.kernels.reference import w4a16_reference
from tokenforge.kernels.triton.w4a16 import w4a16_dequant, w4a16_gemm

G = cases.GROUP_SIZE


def _on(w: W4A16Weights, device: torch.device) -> W4A16Weights:
    return W4A16Weights(
        w.qweight_t.to(device), w.scales.to(device), w.qzeros.to(device), w.group_size
    )


def _linear(x: torch.Tensor, w: W4A16Weights, bias: torch.Tensor | None = None) -> torch.Tensor:
    return torch.ops.tokenforge.w4a16_linear(
        x, w.qweight_t, w.scales, w.qzeros, bias, w.group_size, "triton"
    )


def _assert_close(y: torch.Tensor, x: torch.Tensor, w: W4A16Weights, ref: torch.Tensor) -> None:
    y64 = y.detach().cpu().to(torch.float64)
    tol = atol(x.cpu(), dequantize_tf(_on(w, torch.device("cpu"))), y.device.type)
    err = float((y64 - ref).abs().max())
    assert err <= tol, f"max abs err {err:.3e} > derived tol {tol:.3e}"


SHAPES = linear_shapes()


@pytest.mark.parametrize("m", m_values())
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{s.model}/{s.layer}" for s in SHAPES])
def test_linear_shape_suite(shape, m: int, device: torch.device) -> None:  # type: ignore[no-untyped-def]
    for seed in cases.TEST_SEEDS:
        w = cases.layer_weights(shape.n, shape.k, seed)
        x = cases.activations(m, shape.k, seed)
        y = _linear(x.to(device), _on(w, device))
        assert y.shape == (m, shape.n)
        assert y.dtype == torch.float16
        _assert_close(y, x, w, w4a16_reference(x, w))


PATH_CONFIGS = [
    GemmConfig(1, 64, 128, 1, 4, 2),
    GemmConfig(1, 32, 64, 4, 2, 2),
    GemmConfig(2, 64, 64, 1, 4, 2),
    GemmConfig(8, 32, 32, 2, 4, 2),
    GemmConfig(16, 32, 64, 1, 4, 2),
    GemmConfig(32, 64, 32, 3, 4, 3),
]


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("cfg", PATH_CONFIGS, ids=lambda c: f"{c.path}-bm{c.block_m}-sk{c.split_k}")
def test_every_code_path(cfg: GemmConfig, with_bias: bool, device: torch.device) -> None:
    n, k = 200, 512  # N not a multiple of any block_n
    w = cases.layer_weights(n, k, 0)
    b = cases.bias(n, 0) if with_bias else None
    m_list = (1,) if cfg.block_m == 1 else (cfg.block_m - 1, cfg.block_m + 3)
    for m in m_list:
        x = cases.activations(m, k, 0)
        wd = _on(w, device)
        y = w4a16_gemm(
            x.to(device),
            wd.qweight_t,
            wd.scales,
            wd.qzeros,
            G,
            cfg,
            None if b is None else b.to(device),
        )
        _assert_close(y, x, w, w4a16_reference(x, w, b))


def test_dequant_kernel_is_exact(device: torch.device) -> None:
    w = cases.layer_weights(200, 384, 0)
    wd = _on(w, device)
    out = w4a16_dequant(wd.qweight_t, wd.scales, wd.qzeros, G)
    assert torch.equal(out.cpu(), dequantize_tf(w))


def test_large_m_routes_to_dequant_gemm(device: torch.device) -> None:
    KERNEL_CALLS.reset()
    w = cases.layer_weights(128, 256, 0)
    x = cases.activations(300, 256, 0)
    y = _linear(x.to(device), _on(w, device))
    _assert_close(y, x, w, w4a16_reference(x, w))
    assert KERNEL_CALLS.value(kernel="dequant_gemm", bucket=">128") == 1


def test_leading_dims_and_bias_through_op(device: torch.device) -> None:
    w = cases.layer_weights(64, 256, 0)
    b = cases.bias(64, 0)
    x = cases.activations(6, 256, 0)
    y = _linear(x.reshape(2, 3, 256).to(device), _on(w, device), b.to(device))
    assert y.shape == (2, 3, 64)
    _assert_close(y.reshape(6, 64), x, w, w4a16_reference(x, w, b))


def test_zero_rows(device: torch.device) -> None:
    w = _on(cases.layer_weights(64, 256, 0), device)
    y = _linear(torch.empty((0, 256), dtype=torch.float16, device=device), w)
    assert y.shape == (0, 64)


def test_non_contiguous_and_misaligned_inputs_are_handled(device: torch.device) -> None:
    w = cases.layer_weights(64, 256, 0)
    wd = _on(w, device)
    x = cases.activations(4, 256, 0)
    ref = w4a16_reference(x, w)
    # Column-major view: the op makes it contiguous along K.
    x_cm = x.t().contiguous().t().to(device)
    assert x_cm.stride(1) != 1
    _assert_close(_linear(x_cm, wd), x, w, ref)
    # Storage offset of one fp16 element: base pointer is not 16-byte aligned.
    buf = torch.empty(4 * 256 + 1, dtype=torch.float16, device=device)
    x_mis = buf[1:].view(4, 256)
    x_mis.copy_(x.to(device))
    assert x_mis.data_ptr() % 16 != 0
    _assert_close(_linear(x_mis, wd), x, w, ref)


def test_rejects_bad_weight_layouts(device: torch.device) -> None:
    w = _on(cases.layer_weights(64, 256, 0), device)
    x = cases.activations(1, 256, 0).to(device)
    cfg = GemmConfig(1, 64, 128)
    with pytest.raises(ValueError, match="contiguous"):
        w4a16_gemm(x, w.qweight_t.t().contiguous().t(), w.scales, w.qzeros, G, cfg)
    with pytest.raises(ValueError, match="encodes K"):
        w4a16_gemm(x[:, :128], w.qweight_t, w.scales, w.qzeros, G, cfg)
    with pytest.raises(TypeError, match="fp16"):
        w4a16_gemm(x.float(), w.qweight_t, w.scales, w.qzeros, G, cfg)
    with pytest.raises(ValueError, match="divide"):
        w4a16_gemm(x, w.qweight_t, w.scales, w.qzeros, G, GemmConfig(1, 64, 256))


# ----------------------------------------------------------------------------- F1 and edges


def _weights_from_ints(q: torch.Tensor, z: torch.Tensor, s: torch.Tensor) -> W4A16Weights:
    """q [K, N] ints, z [K//G, N] ints, s [K//G, N] fp16 -> TokenForge weights."""
    qweight, qzeros = awq_pack(q.to(torch.int32), z.to(torch.int32))
    return pack_awq(qweight, qzeros, s.to(torch.float16), G)


def test_f1_outliers_cancel_without_fp16_overflow(device: torch.device) -> None:
    """Products of 6e4 x 1.6 overflow fp16; fp32 products + accumulation must cancel them."""
    k, n = 256, 64
    gen = torch.Generator().manual_seed(0)
    q = torch.randint(0, 16, (k, n), generator=gen)
    q[1] = q[0] = 15  # rows 0 and 1 share group 0 -> identical dequantized weights
    z = torch.full((k // G, n), 7)
    s = torch.full((k // G, n), 0.2)
    w = _weights_from_ints(q, z, s)
    x = (torch.randn((2, k), generator=gen) * 0.5).to(torch.float16)
    x[:, 0], x[:, 1] = 6e4, -6e4
    for cfg in (GemmConfig(1, 64, 128), GemmConfig(2, 32, 64), GemmConfig(16, 64, 64)):
        wd = _on(w, device)
        y = w4a16_gemm(
            x[: max(1, min(2, cfg.block_m))].to(device), wd.qweight_t, wd.scales, wd.qzeros, G, cfg
        )
        assert bool(torch.isfinite(y).all()), f"spurious Inf/NaN with {cfg}"
        ref = w4a16_reference(x[: y.shape[0]], w)
        assert bool(torch.isfinite(ref).all())
        _assert_close(y, x[: y.shape[0]], w, ref)


def test_f1_real_inf_and_nan_propagate(device: torch.device) -> None:
    w = cases.layer_weights(64, 256, 0)
    x = cases.activations(16, 256, 0)
    x[0, 5] = float("inf")
    x[1, 7] = float("-inf")
    x[2, 9] = float("nan")
    ref16 = w4a16_reference(x, w).to(torch.float16)
    for cfg in (GemmConfig(1, 64, 128), GemmConfig(4, 32, 64, 2), GemmConfig(16, 64, 64)):
        wd = _on(w, device)
        rows = x[: max(cfg.block_m, 3)]
        y = w4a16_gemm(rows.to(device), wd.qweight_t, wd.scales, wd.qzeros, G, cfg).cpu()
        r = ref16[: rows.shape[0]]
        assert torch.equal(torch.isnan(y), torch.isnan(r)), cfg
        assert torch.equal(torch.isposinf(y), torch.isposinf(r)), cfg
        assert torch.equal(torch.isneginf(y), torch.isneginf(r)), cfg
        assert bool(torch.isfinite(y[3:]).all()), "Inf leaked into clean rows"


@pytest.mark.parametrize("scale_value", [0.0, 2.0**-24, 6e-6])  # zero, min denormal, denormal
def test_zero_and_denormal_scales(scale_value: float, device: torch.device) -> None:
    k, n = 256, 64
    gen = torch.Generator().manual_seed(1)
    q = torch.randint(0, 16, (k, n), generator=gen)
    z = torch.randint(0, 16, (k // G, n), generator=gen)
    s = torch.rand((k // G, n), generator=gen) * 0.01 + 1e-3
    s[0, :] = scale_value  # group 0 of every column
    w = _weights_from_ints(q, z, s)
    x = cases.activations(3, k, 0)
    wd = _on(w, device)
    for cfg in (GemmConfig(1, 64, 128), GemmConfig(4, 64, 64)):
        y = w4a16_gemm(x[: cfg.block_m].to(device), wd.qweight_t, wd.scales, wd.qzeros, G, cfg)
        assert bool(torch.isfinite(y).all())
        _assert_close(y, x[: cfg.block_m], w, w4a16_reference(x[: cfg.block_m], w))


def test_all_zero_weights_are_exact(device: torch.device) -> None:
    k, n = 128, 16
    w = _weights_from_ints(torch.full((k, n), 9), torch.full((1, n), 9), torch.full((1, n), 0.5))
    y = _linear(cases.activations(2, k, 0).to(device), _on(w, device))
    assert bool((y == 0).all())
