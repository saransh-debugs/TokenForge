"""T1/T2: fused RMSNorm + e4m3 quantization vs the PyTorch reference.

On CUDA, Triton supports ``float8e4nv`` only on sm >= 89, so these run on the L4 (GPU-full)
and under the interpreter on CPU.
"""

from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

import tokenforge.kernels.ops  # noqa: F401
from tokenforge.kernels.reference import FP8_E4M3_MAX, rmsnorm, rmsnorm_fp8_reference
from tokenforge.kernels.triton.rmsnorm_fp8 import _round_to_e4m3_grid, rmsnorm_fp8_quant

pytestmark = pytest.mark.sm89
EPS = 1e-6


@triton.jit
def _cast_e4m3(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, _round_to_e4m3_grid(v).to(tl.float8e4nv), mask=m)


def _inputs(t: int, h: int, seed: int, scale: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn((t, h), generator=gen) * scale).to(torch.float16)
    w = (torch.rand((h,), generator=gen) + 0.5).to(torch.float16)
    return x, w


def _codes(q: torch.Tensor) -> torch.Tensor:
    return q.cpu().view(torch.uint8).to(torch.int16)


def _assert_within_one_code(q: torch.Tensor, ref: torch.Tensor) -> None:
    """Same sign, magnitudes at most one e4m3 code apart (ordering of the low 7 bits)."""
    a, b = _codes(q), _codes(ref)
    mag_a, mag_b = a & 0x7F, b & 0x7F
    sign_ok = (
        ((a & 0x80) == (b & 0x80)) | ((mag_a == 0) & (mag_b <= 1)) | ((mag_b == 0) & (mag_a <= 1))
    )
    assert bool(sign_ok.all()), "sign mismatch"
    diff = (mag_a - mag_b).abs()
    assert int(diff.max()) <= 1, f"max code distance {int(diff.max())}"
    assert float((diff > 0).float().mean()) < 0.05, "too many one-code differences"


@pytest.mark.parametrize("h", [896, 3584, 1000])
@pytest.mark.parametrize("t", [1, 7, 33])
def test_dynamic_per_token(t: int, h: int, device: torch.device) -> None:
    x, w = _inputs(t, h, seed=t * h)
    q, s = rmsnorm_fp8_quant(x.to(device), w.to(device), EPS)
    q_ref, s_ref = rmsnorm_fp8_reference(x, w, EPS)
    assert q.dtype == torch.float8_e4m3fn
    assert s.shape == (t, 1)
    assert float(((s.cpu() - s_ref).abs() / s_ref).max()) <= 1e-3
    _assert_within_one_code(q, q_ref)
    # Per-token scaling: every row hits the e4m3 maximum exactly once at its absmax.
    assert bool((q.cpu().float().abs().amax(dim=-1) == FP8_E4M3_MAX).all())
    # Dequantized values track the fp16 RMSNorm output to e4m3 precision (3 mantissa bits).
    y = rmsnorm(x, w, EPS).float()
    deq = q.cpu().float() * s.cpu()
    rel = (deq - y).abs() / y.abs().clamp(min=float(s_ref.max()))
    assert float(rel.max()) <= 2.0**-3


def test_static_scale_saturates_exactly(device: torch.device) -> None:
    x, w = _inputs(16, 896, seed=11, scale=3.0)
    scale = torch.tensor([2e-3], dtype=torch.float32)
    q, s_out = rmsnorm_fp8_quant(x.to(device), w.to(device), EPS, scale.to(device))
    q_ref, _ = rmsnorm_fp8_reference(x, w, EPS, scale)
    assert torch.equal(s_out.cpu(), scale)
    y = rmsnorm(x, w, EPS).float() / scale
    saturated = y.abs() >= FP8_E4M3_MAX
    assert int(saturated.sum()) > 100, "test must exercise saturation"
    qf = q.cpu().float()
    assert torch.equal(qf[saturated], q_ref.float()[saturated])
    assert bool((qf[saturated].abs() == FP8_E4M3_MAX).all())
    assert not bool(torch.isnan(qf).any()), "saturating cast must never emit NaN"
    _assert_within_one_code(q, q_ref)


def test_e4m3_rounding_is_bit_exact_at_every_midpoint(device: torch.device) -> None:
    """Every e4m3 value, every midpoint between neighbors, and the fp32 values either side
    of each midpoint round exactly like PyTorch's ``to(float8_e4m3fn)``, signs included."""
    values = torch.arange(0, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    mids = (values[1:] + values[:-1]) / 2
    up, down = torch.tensor(1e9), torch.tensor(-1e9)
    probe = torch.cat([values, mids, mids.nextafter(up), mids.nextafter(down)])
    probe = torch.cat([probe, -probe, torch.tensor([0.0, -0.0])])
    out = torch.empty(probe.numel(), dtype=torch.float8_e4m3fn, device=device)
    _cast_e4m3[(triton.cdiv(probe.numel(), 1024),)](
        probe.to(device), out, probe.numel(), BLOCK=1024
    )
    assert torch.equal(_codes(out), _codes(probe.to(torch.float8_e4m3fn)))


def test_zero_row_uses_min_scale(device: torch.device) -> None:
    x = torch.zeros((2, 256), dtype=torch.float16)
    w = torch.ones(256, dtype=torch.float16)
    q, s = rmsnorm_fp8_quant(x.to(device), w.to(device), EPS)
    assert bool((q.cpu().float() == 0).all())
    assert bool((s.cpu() > 0).all())


def test_op_matches_kernel_and_keeps_leading_dims(device: torch.device) -> None:
    x, w = _inputs(6, 896, seed=3)
    q, s = torch.ops.tokenforge.rmsnorm_fp8_quant(
        x.reshape(2, 3, 896).to(device), w.to(device), EPS, None
    )
    assert q.shape == (2, 3, 896)
    assert s.shape == (2, 3, 1)
    qk, sk = rmsnorm_fp8_quant(x.to(device), w.to(device), EPS)
    assert torch.equal(_codes(q.reshape(6, 896)), _codes(qk))
    assert torch.equal(s.reshape(6, 1).cpu(), sk.cpu())


def test_rejects_bad_inputs(device: torch.device) -> None:
    x, w = _inputs(2, 128, seed=0)
    with pytest.raises(ValueError, match="2-D"):
        rmsnorm_fp8_quant(x.reshape(-1).to(device), w.to(device), EPS)
    with pytest.raises(ValueError, match="weight shape"):
        rmsnorm_fp8_quant(x.to(device), w[:64].to(device), EPS)
    with pytest.raises(ValueError, match="single fp32"):
        rmsnorm_fp8_quant(x.to(device), w.to(device), EPS, torch.ones(2, device=device))
