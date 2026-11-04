"""Device identity helpers. None of these read tensor values or sync with the device."""

from __future__ import annotations

import os
import re

import torch


def interpreter_active() -> bool:
    """True when Triton kernels execute in the CPU interpreter (``TRITON_INTERPRET=1``)."""
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def device_sm(device: torch.device) -> int | None:
    """Compute capability as ``major * 10 + minor`` (89 for L4, 75 for T4), None off-GPU."""
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def triton_runnable(t: torch.Tensor) -> bool:
    """Whether a Triton kernel can run on this tensor in the current process."""
    return t.is_cuda or (interpreter_active() and t.device.type == "cpu")


def gpu_identity(device: torch.device) -> str:
    """Stable, filesystem-safe id of the device, e.g. ``nvidia-l4-sm89``."""
    if device.type != "cuda":
        return "cpu-interpreter" if interpreter_active() else "cpu"
    name = torch.cuda.get_device_name(device)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{slug}-sm{device_sm(device)}"
