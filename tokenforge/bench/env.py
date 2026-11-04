"""Environment fingerprint attached to every benchmark row and CI report."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VERSIONS_LOCK = ROOT / "infra" / "versions.lock"
# Paths whose content a GPU report vouches for (ADR 0008).
CODE_PATHS = ("tokenforge/kernels", "vllm_plugin", "tests/numerics", "tests/unit")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def git_sha() -> str:
    return _git("rev-parse", "HEAD") or "unknown"


def git_dirty(paths: tuple[str, ...] = ()) -> bool:
    return bool(_git("status", "--porcelain", "--", *paths))


def code_tree_hash(rev: str = "HEAD") -> str:
    """Hash of the git trees under :data:`CODE_PATHS` at ``rev``.

    Unlike the commit SHA it does not change when a report file is committed, so a report
    produced on the PR head still matches the PR head after it is committed.
    """
    lines = []
    for path in CODE_PATHS:
        tree = _git("rev-parse", f"{rev}:{path}")
        lines.append(f"{path}:{tree or 'absent'}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "absent"


def fingerprint() -> dict[str, Any]:
    fp: dict[str, Any] = {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "code_tree": code_tree_hash(),
        "versions_lock_sha256": file_sha256(VERSIONS_LOCK),
        "python": platform.python_version(),
        "host": platform.node(),
        "container_digest": os.environ.get("TOKENFORGE_CONTAINER_DIGEST", "none"),
    }
    try:
        import torch

        fp["torch"] = torch.__version__
        fp["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            fp["gpu"] = props.name
            fp["sm"] = f"{props.major}{props.minor}"
            fp["gpu_uuid"] = str(getattr(props, "uuid", "unknown"))
    except ImportError:
        pass
    for mod in ("triton", "vllm", "tensorrt_llm"):
        try:
            fp[mod] = __import__(mod).__version__
        except ImportError:
            continue
    try:
        import pynvml

        pynvml.nvmlInit()
        fp["driver"] = pynvml.nvmlSystemGetDriverVersion()
    except Exception:  # NVML absent off-GPU
        pass
    return fp
