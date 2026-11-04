"""T0: dispatcher decisions, counters, and launch-config validity."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tokenforge.kernels import dispatch
from tokenforge.kernels.config import (
    M_BUCKETS,
    GemmConfig,
    candidate_configs,
    default_config,
    m_bucket,
)
from tokenforge.kernels.metrics import FALLBACKS, KERNEL_CALLS


@pytest.fixture(autouse=True)
def _reset_counters() -> None:
    KERNEL_CALLS.reset()
    FALLBACKS.reset()


@given(st.integers(1, 128))
def test_m_bucket_is_tight(m: int) -> None:
    b = m_bucket(m)
    assert b in M_BUCKETS
    assert b >= m
    assert b == 1 or b // 2 < m


def test_m_bucket_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        m_bucket(0)
    with pytest.raises(ValueError, match="exceeds"):
        m_bucket(129)


@pytest.mark.parametrize(
    ("n", "k", "group", "dtype", "sm", "backend", "expected"),
    [
        (1152, 896, 128, "float16", 89, "triton", None),
        (1152, 896, 128, "torch.float16", None, "cuda", None),
        (1152, 896 + 64, 128, "float16", 89, "triton", "unsupported_shape"),  # F2
        (1150, 896, 128, "float16", 89, "triton", "unsupported_shape"),
        (1152, 896, 256, "float16", 89, "triton", "unsupported_group_size"),
        (1152, 896, 128, "bfloat16", 89, "triton", "unsupported_dtype"),
        (1152, 896, 128, "float16", 70, "triton", "unsupported_arch"),
        (1152, 896, 128, "float16", 89, "marlin", "kill_switch"),
    ],
)
def test_layer_support(n, k, group, dtype, sm, backend, expected) -> None:  # type: ignore[no-untyped-def]
    assert dispatch.layer_support(n, k, group, dtype, sm, backend) == expected


@pytest.mark.parametrize(
    ("m", "backend", "ext", "kernel", "bucket", "fallback"),
    [
        (1, "triton", False, "triton_gemm", "1", None),
        (3, "triton", True, "triton_gemm", "4", None),
        (128, "triton", False, "triton_gemm", "128", None),
        (129, "triton", False, "dequant_gemm", ">128", None),
        (4096, "cuda", True, "dequant_gemm", ">128", None),
        (1, "cuda", True, "cuda_gemv", "1", None),
        (4, "cuda", True, "cuda_gemv", "4", None),
        (5, "cuda", True, "triton_gemm", "8", None),
        (2, "cuda", False, "triton_gemm", "2", "cuda_ext_unavailable"),
    ],
)
def test_select(m, backend, ext, kernel, bucket, fallback) -> None:  # type: ignore[no-untyped-def]
    d = dispatch.select(m, backend=backend, on_gpu=True, triton_ok=True, cuda_ext_available=ext)
    assert (d.kernel, d.bucket, d.fallback) == (kernel, bucket, fallback)


def test_select_without_triton_uses_torch_reference() -> None:
    d = dispatch.select(
        8, backend="triton", on_gpu=False, triton_ok=False, cuda_ext_available=False
    )
    assert d.kernel == "torch_reference"


def test_select_rejects_marlin_per_call() -> None:
    with pytest.raises(ValueError, match="per layer"):
        dispatch.select(1, backend="marlin", on_gpu=True, triton_ok=True, cuda_ext_available=True)


def test_record_counts_calls_and_fallbacks() -> None:
    d = dispatch.select(2, backend="cuda", on_gpu=True, triton_ok=True, cuda_ext_available=False)
    dispatch.record(d)
    dispatch.record(d)
    assert KERNEL_CALLS.value(kernel="triton_gemm", bucket="2") == 2
    assert FALLBACKS.value(reason="cuda_ext_unavailable") == 2
    dispatch.record_layer_fallback("unsupported_shape")
    assert FALLBACKS.value(reason="unsupported_shape") == 1


def test_backend_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(dispatch.BACKEND_ENV, raising=False)
    assert dispatch.backend_from_env() == "triton"
    monkeypatch.setenv(dispatch.BACKEND_ENV, " Marlin ")
    assert dispatch.backend_from_env() == "marlin"
    monkeypatch.setenv(dispatch.BACKEND_ENV, "cublas")
    with pytest.raises(ValueError, match="TOKENFORGE_BACKEND"):
        dispatch.backend_from_env()


@pytest.mark.parametrize("bucket", M_BUCKETS)
@pytest.mark.parametrize(("n", "k"), [(1152, 896), (896, 4864), (37888, 3584), (200, 256)])
def test_default_config_is_valid(bucket: int, n: int, k: int) -> None:
    cfg = default_config(bucket, n, k, 128)
    cfg.validate(128)
    assert cfg.block_m <= max(bucket, 16)
    assert cfg.split_k <= k // cfg.block_k


@pytest.mark.parametrize("bucket", M_BUCKETS)
def test_candidates_are_valid(bucket: int) -> None:
    cands = candidate_configs(bucket, 128)
    assert cands
    for cfg in cands:
        cfg.validate(128)
    assert all(c.block_m == 1 for c in cands) == (bucket == 1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"block_m": 3, "block_n": 64, "block_k": 64}, "block_m"),
        ({"block_m": 1, "block_n": 48, "block_k": 64}, "block_n"),
        ({"block_m": 1, "block_n": 64, "block_k": 256}, "divide"),
        ({"block_m": 1, "block_n": 64, "block_k": 64, "split_k": 0}, "split_k"),
        ({"block_m": 1, "block_n": 64, "block_k": 64, "num_warps": 3}, "num_warps"),
        ({"block_m": 1, "block_n": 64, "block_k": 64, "num_stages": 9}, "num_stages"),
    ],
)
def test_config_validation(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GemmConfig(**kwargs).validate(128)


def test_config_from_dict_strict() -> None:
    good = GemmConfig(1, 64, 128).to_dict()
    assert GemmConfig.from_dict(good) == GemmConfig(1, 64, 128)
    with pytest.raises(ValueError, match="unknown"):
        GemmConfig.from_dict({**good, "block_q": 1})
    with pytest.raises(TypeError, match="integers"):
        GemmConfig.from_dict({**good, "split_k": True})
    with pytest.raises(TypeError, match="object"):
        GemmConfig.from_dict([1, 2])
