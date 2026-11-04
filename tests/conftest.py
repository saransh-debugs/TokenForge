"""Shared fixtures. Without a GPU, Triton kernels run in the interpreter on CPU tensors.

``TRITON_INTERPRET`` is read when ``@triton.jit`` decorates a function, so it is set here,
before any test module imports a kernel.
"""

from __future__ import annotations

import os

import pytest

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:  # serving-only environment
    torch = None  # type: ignore[assignment]
    _HAS_CUDA = False

if not _HAS_CUDA:
    os.environ.setdefault("TRITON_INTERPRET", "1")


@pytest.fixture(scope="session")
def device() -> torch.device:
    if torch is None:
        pytest.skip("torch not installed")
    return torch.device("cuda" if _HAS_CUDA else "cpu")


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Every test gets its own autotune cache dir; nothing leaks into ~/.cache."""
    monkeypatch.setenv("TOKENFORGE_CACHE_DIR", str(tmp_path_factory.mktemp("tf-cache")))
    try:
        from tokenforge.kernels import autotune
    except ImportError:
        yield
        return
    autotune.reset_caches()
    yield
    autotune.reset_caches()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    sm = None
    if _HAS_CUDA:
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
    skip_gpu = pytest.mark.skip(reason="needs a CUDA device")
    skip_sm89 = pytest.mark.skip(reason="needs compute capability >= 8.9")
    for item in items:
        if "gpu" in item.keywords and not _HAS_CUDA:
            item.add_marker(skip_gpu)
        if "sm89" in item.keywords and sm is not None and sm < 89:
            item.add_marker(skip_sm89)
