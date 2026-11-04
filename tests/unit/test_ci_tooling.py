"""The GPU-report gate and the microbenchmark regression check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenforge.bench import micro
from tokenforge.bench.env import code_tree_hash, fingerprint
from tokenforge.ci import gpu_report


@pytest.fixture
def report_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(gpu_report, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(gpu_report, "ROOT", tmp_path.parent)
    return tmp_path


def _write(report_dir: Path, passed: bool) -> Path:
    tree = code_tree_hash()
    path = report_dir / f"dev-{tree[:16]}.json"
    steps = [{"name": "numerics", "passed": passed}]
    path.write_text(json.dumps({"env": {"code_tree": tree}, "steps": steps, "passed": passed}))
    return path


def test_code_tree_hash_is_stable_and_hex() -> None:
    a, b = code_tree_hash(), code_tree_hash()
    assert a == b
    assert len(a) == 64
    int(a, 16)


def test_check_requires_report(report_dir: Path) -> None:
    assert gpu_report.check("HEAD", "dev") == 1


def test_check_accepts_passing_report(report_dir: Path) -> None:
    _write(report_dir, passed=True)
    assert gpu_report.check("HEAD", "dev") == 0


def test_check_rejects_failing_report(report_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(report_dir, passed=False)
    assert gpu_report.check("HEAD", "dev") == 1
    assert "numerics" in capsys.readouterr().err


def test_fingerprint_has_reproducibility_fields() -> None:
    fp = fingerprint()
    for key in ("git_sha", "code_tree", "versions_lock_sha256", "torch", "container_digest"):
        assert key in fp


def test_microbench_regression_gate() -> None:
    rows = [
        {"impl": "tokenforge-triton", "layer": "qwen2.5-7b/o_proj", "m": 1, "ms": 0.104},
        {"impl": "tokenforge-triton", "layer": "qwen2.5-7b/o_proj", "m": 8, "ms": 0.110},
    ]
    baseline = {
        "tokenforge-triton|qwen2.5-7b/o_proj|1": 0.100,
        "tokenforge-triton|qwen2.5-7b/o_proj|8": 0.100,
    }
    failures = micro.check(rows, baseline)
    assert len(failures) == 1
    assert "|8" in failures[0]


def test_microbench_bytes_moved_counts_quant_metadata() -> None:
    c = micro.Case("x", m=1, n=4096, k=4096)
    weights = 4096 * 4096 // 2
    assert micro.bytes_moved(c) > weights
    assert micro.bytes_moved(c) < weights * 1.1
