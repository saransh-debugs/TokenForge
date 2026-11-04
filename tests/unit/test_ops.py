"""T0/T4 groundwork: op registration is torch.compile- and fake-tensor-safe."""

from __future__ import annotations

import pytest
import torch
from torch.library import opcheck

import tokenforge.kernels.ops  # noqa: F401
from tests.numerics import cases
from tokenforge.kernels.metrics import KERNEL_CALLS

G = cases.GROUP_SIZE
OPCHECK_TESTS = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")


def _layer(n: int = 64, k: int = 256):  # type: ignore[no-untyped-def]
    w = cases.layer_weights(n, k, 0)
    return w.qweight_t, w.scales, w.qzeros


@pytest.mark.parametrize("with_bias", [False, True])
def test_opcheck_w4a16_linear(with_bias: bool) -> None:
    qw, s, z = _layer()
    x = cases.activations(3, 256, 0)
    b = cases.bias(64, 0) if with_bias else None
    opcheck(
        torch.ops.tokenforge.w4a16_linear.default,
        (x, qw, s, z, b, G, "triton"),
        test_utils=OPCHECK_TESTS,
    )


def test_opcheck_dequant_and_rmsnorm() -> None:
    qw, s, z = _layer()
    opcheck(torch.ops.tokenforge.w4a16_dequant.default, (qw, s, z, G), test_utils=OPCHECK_TESTS)
    x = cases.activations(4, 256, 1)
    w = torch.ones(256, dtype=torch.float16)
    opcheck(
        torch.ops.tokenforge.rmsnorm_fp8_quant.default,
        (x, w, 1e-6, None),
        test_utils=("test_schema", "test_faketensor"),
    )


class _QLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        qw, s, z = _layer()
        self.register_buffer("qw", qw)
        self.register_buffer("s", s)
        self.register_buffer("z", z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.ops.tokenforge.w4a16_linear(x, self.qw, self.s, self.z, None, G, "triton")
        return torch.relu(h) * 2


def test_torch_compile_fullgraph_no_graph_breaks() -> None:
    mod = _QLinear()
    x = cases.activations(5, 256, 2)
    eager = mod(x)
    compiled = torch.compile(mod, backend="aot_eager", fullgraph=True, dynamic=True)
    assert torch.equal(compiled(x), eager)
    # A new M must reuse the dynamic graph and still dispatch by its own bucket.
    KERNEL_CALLS.reset()
    x2 = cases.activations(17, 256, 3)
    assert torch.equal(compiled(x2), mod(x2))
    assert KERNEL_CALLS.value(kernel="triton_gemm", bucket="32") == 2


def test_fake_tensor_shapes() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    qw, s, z = _layer()
    with FakeTensorMode() as mode:
        x = mode.from_tensor(cases.activations(2, 256, 0)).reshape(1, 2, 256)
        out = torch.ops.tokenforge.w4a16_linear(
            x, mode.from_tensor(qw), mode.from_tensor(s), mode.from_tensor(z), None, G, "triton"
        )
        assert out.shape == (1, 2, 64)
        assert out.dtype == torch.float16


def test_rejects_marlin_at_op_level() -> None:
    qw, s, z = _layer()
    with pytest.raises(ValueError, match="op level"):
        torch.ops.tokenforge.w4a16_linear(cases.activations(1, 256, 0), qw, s, z, None, G, "marlin")
