"""T0: AWQ packing, the TokenForge layout, and dequantization."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tokenforge.kernels.pack import (
    AWQ_ORDER,
    AWQ_REVERSE_ORDER,
    awq_dequantize,
    awq_pack,
    awq_quantize,
    awq_unpack,
    dequantize_tf,
    pack_awq,
    pack_int4,
    repack_awq_to_tf,
    repack_tf_to_awq,
    unpack_int4,
    unpack_to_awq,
    validate_awq,
)

G = 128


def _autoawq_pack_scalar(intweight: torch.Tensor) -> torch.Tensor:
    """Literal transcription of AutoAWQ's WQLinear_GEMM packing loop (independent oracle)."""
    k, n = intweight.shape
    out = torch.zeros((k, n // 8), dtype=torch.int64)
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    for col in range(n // 8):
        for i in range(8):
            out[:, col] |= intweight[:, col * 8 + order_map[i]].to(torch.int64) << (i * 4)
    return out.to(torch.int32)


def test_orders_are_inverse() -> None:
    assert [AWQ_ORDER[AWQ_REVERSE_ORDER[j]] for j in range(8)] == list(range(8))
    assert list(AWQ_REVERSE_ORDER) == [(j % 2) * 4 + j // 2 for j in range(8)]


@settings(max_examples=60, deadline=None)
@given(
    rows=st.integers(1, 9),
    words=st.integers(1, 6),
    seed=st.integers(0, 2**31 - 1),
)
def test_pack_roundtrip(rows: int, words: int, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    values = torch.randint(0, 16, (rows, words * 8), generator=gen, dtype=torch.int32)
    values[0, 0], values[-1, -1] = 0, 15  # both extremes always present
    packed = pack_int4(values)
    assert packed.dtype == torch.int32
    assert packed.shape == (rows, words)
    assert torch.equal(unpack_int4(packed), values)


@settings(max_examples=25, deadline=None)
@given(k=st.integers(1, 6), n_words=st.integers(1, 5), seed=st.integers(0, 2**31 - 1))
def test_awq_pack_matches_autoawq_loop(k: int, n_words: int, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    intweight = torch.randint(0, 16, (k, n_words * 8), generator=gen, dtype=torch.int32)
    zeros = torch.randint(0, 16, (1, n_words * 8), generator=gen, dtype=torch.int32)
    qweight, _ = awq_pack(intweight, zeros)
    assert torch.equal(qweight, _autoawq_pack_scalar(intweight))
    assert torch.equal(awq_unpack(qweight), intweight)


def test_high_nibble_sign_bit() -> None:
    """15 in nibble 7 sets bit 31; the pattern must survive int32 wrap-around."""
    values = torch.zeros((1, 8), dtype=torch.int32)
    values[0, AWQ_ORDER[7]] = 15
    packed = pack_int4(values)
    assert int(packed) == -(2**28)  # 0xF0000000 as int32
    assert torch.equal(unpack_int4(packed), values)


def test_pack_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        pack_int4(torch.full((1, 8), 16, dtype=torch.int32))
    with pytest.raises(ValueError, match="multiple of 8"):
        pack_int4(torch.zeros((1, 7), dtype=torch.int32))


@settings(max_examples=20, deadline=None)
@given(k_groups=st.integers(1, 3), n_words=st.integers(1, 4), seed=st.integers(0, 2**31 - 1))
def test_repack_is_invertible(k_groups: int, n_words: int, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    k, n = k_groups * G, n_words * 8
    qweight = torch.randint(-(2**31), 2**31 - 1, (k, n // 8), generator=gen, dtype=torch.int32)
    qweight_t = repack_awq_to_tf(qweight)
    assert qweight_t.shape == (n, k // 8)
    assert torch.equal(repack_tf_to_awq(qweight_t), qweight)
    # Element (k, n) of the logical matrix is the same in both layouts.
    assert torch.equal(unpack_int4(qweight_t).t(), awq_unpack(qweight))


def _random_awq(n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    intweight = torch.randint(0, 16, (k, n), generator=gen, dtype=torch.int32)
    zeros = torch.randint(0, 16, (k // G, n), generator=gen, dtype=torch.int32)
    zeros[0, 0], zeros[-1, -1] = 0, 15
    scales = (torch.rand((k // G, n), generator=gen) * 0.01 + 1e-3).to(torch.float16)
    qweight, qzeros = awq_pack(intweight, zeros)
    return qweight, qzeros, scales


def test_pack_awq_roundtrip_and_dequant_agree() -> None:
    qweight, qzeros, scales = _random_awq(n=64, k=3 * G, seed=7)
    w = pack_awq(qweight, qzeros, scales, G)
    assert (w.n, w.k) == (64, 3 * G)
    back = unpack_to_awq(w)
    assert torch.equal(back[0], qweight)
    assert torch.equal(back[1], qzeros)
    assert torch.equal(back[2], scales)
    assert torch.equal(dequantize_tf(w), awq_dequantize(qweight, qzeros, scales, G).t())


def test_dequant_at_group_boundaries() -> None:
    qweight, qzeros, scales = _random_awq(n=16, k=2 * G, seed=3)
    q = awq_unpack(qweight)
    z = awq_unpack(qzeros)
    deq = awq_dequantize(qweight, qzeros, scales, G)
    for kk in (0, G - 1, G, 2 * G - 1):
        g = kk // G
        for nn in (0, 7, 8, 15):
            expected = (
                torch.tensor(float(q[kk, nn] - z[g, nn]), dtype=torch.float16) * scales[g, nn]
            )
            assert deq[kk, nn] == expected, (kk, nn)


def test_zero_points_at_extremes() -> None:
    k, n = G, 8
    intweight = torch.tensor([[0, 15, 0, 15, 7, 8, 1, 14]] * k, dtype=torch.int32)
    for zp in (0, 15):
        zeros = torch.full((1, n), zp, dtype=torch.int32)
        qweight, qzeros = awq_pack(intweight, zeros)
        scales = torch.ones((1, n), dtype=torch.float16)
        deq = awq_dequantize(qweight, qzeros, scales, G)
        assert torch.equal(deq[0], (intweight[0] - zp).to(torch.float16))


def test_awq_quantize_error_bound() -> None:
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn((32, 2 * G), generator=gen) * 0.05
    qweight, qzeros, scales = awq_quantize(weight, G)
    validate_awq(qweight, qzeros, scales, G)
    deq = awq_dequantize(qweight, qzeros, scales, G).t().float()  # [N, K]
    step = scales.float().repeat_interleave(G, dim=0).t()
    assert bool(((deq - weight).abs() <= 0.5 * step + 1e-3 * step + 1e-6).all())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda q, z, s: (q.float(), z, s), "int32"),
        (lambda q, z, s: (q, z, s.float()), "fp16"),
        (lambda q, z, s: (q, z[:1], s), "qzeros shape"),
        (lambda q, z, s: (q, z, s[:, :8]), "scales shape"),
    ],
)
def test_validate_awq_rejects(mutate, match) -> None:  # type: ignore[no-untyped-def]
    qweight, qzeros, scales = _random_awq(n=16, k=2 * G, seed=1)
    with pytest.raises(ValueError, match=match):
        validate_awq(*mutate(qweight, qzeros, scales), G)


def test_validate_awq_rejects_group_and_k() -> None:
    qweight, qzeros, scales = _random_awq(n=16, k=2 * G, seed=1)
    with pytest.raises(ValueError, match="group_size"):
        validate_awq(qweight, qzeros, scales, 256)
    with pytest.raises(ValueError, match="multiple of group_size"):
        validate_awq(qweight[:-8], qzeros, scales, G)
