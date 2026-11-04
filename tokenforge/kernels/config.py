"""Launch configurations and M buckets for the W4A16 Triton kernel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

M_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
MAX_TRITON_M = M_BUCKETS[-1]
_POW2 = frozenset(1 << i for i in range(12))


def m_bucket(m: int) -> int:
    """Smallest bucket >= m. Callers route m > 128 elsewhere."""
    if m < 1:
        raise ValueError(f"M must be >= 1, got {m}")
    for b in M_BUCKETS:
        if m <= b:
            return b
    raise ValueError(f"M={m} exceeds the largest Triton bucket ({MAX_TRITON_M})")


@dataclass(frozen=True)
class GemmConfig:
    block_m: int
    block_n: int
    block_k: int
    split_k: int = 1
    num_warps: int = 4
    num_stages: int = 2

    @property
    def path(self) -> str:
        if self.block_m == 1:
            return "gemv"
        return "fma" if self.block_m < 16 else "dot"

    def validate(self, group_size: int) -> None:
        """Raise ``ValueError`` if this config cannot run for ``group_size``."""
        if self.block_m not in M_BUCKETS:
            raise ValueError(f"block_m={self.block_m} not in {M_BUCKETS}")
        for name in ("block_n", "block_k"):
            v = getattr(self, name)
            if v not in _POW2 or v < 16:
                raise ValueError(f"{name}={v} must be a power of two >= 16")
        if self.block_k > group_size or group_size % self.block_k != 0:
            raise ValueError(f"block_k={self.block_k} must divide group_size={group_size}")
        if not 1 <= self.split_k <= 16:
            raise ValueError(f"split_k={self.split_k} must be in [1, 16]")
        if self.num_warps not in (1, 2, 4, 8, 16):
            raise ValueError(f"num_warps={self.num_warps} must be a power of two <= 16")
        if not 1 <= self.num_stages <= 6:
            raise ValueError(f"num_stages={self.num_stages} must be in [1, 6]")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> GemmConfig:
        if not isinstance(d, dict):
            raise TypeError(f"config must be an object, got {type(d).__name__}")
        fields = set(cls.__dataclass_fields__)
        unknown = set(d) - fields
        if unknown:
            raise ValueError(f"unknown config keys {sorted(unknown)}")
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in d.values()):
            raise TypeError("config values must be integers")
        return cls(**d)


def default_config(bucket: int, n: int, k: int, group_size: int) -> GemmConfig:
    """Heuristic config used before tuning and in the Triton interpreter."""
    block_k = min(128, group_size)
    if bucket == 1:
        cfg = GemmConfig(block_m=1, block_n=64, block_k=block_k, num_warps=4, num_stages=3)
    elif bucket < 16:
        cfg = GemmConfig(block_m=bucket, block_n=64, block_k=min(64, block_k), num_warps=4)
    else:
        cfg = GemmConfig(
            block_m=min(bucket, 64),
            block_n=64,
            block_k=min(64, block_k),
            num_warps=4,
            num_stages=3,
        )
    # Split K when the N tiling alone cannot fill the device (decode-sized N).
    n_tiles = -(-n // cfg.block_n)
    k_blocks = k // cfg.block_k
    split = 1
    while n_tiles * split < 64 and split * 2 <= min(8, k_blocks):
        split *= 2
    return GemmConfig(**{**cfg.to_dict(), "split_k": split})


def candidate_configs(bucket: int, group_size: int) -> list[GemmConfig]:
    """Search space for the autotuner, pruned for register pressure."""
    out: list[GemmConfig] = []
    block_ms = [1] if bucket == 1 else ([bucket] if bucket < 16 else [16, 32, 64])
    for bm in block_ms:
        if bm > bucket:
            continue
        for bn in (32, 64, 128, 256):
            for bk in (32, 64, 128):
                if bk > group_size or group_size % bk:
                    continue
                for warps in (2, 4, 8):
                    threads = 32 * warps
                    # fp32 live elements per thread in the widest intermediate.
                    live = (bn * bk if bm == 1 else bm * bn * (bk if bm < 16 else 1)) / threads
                    if live > 128 or (bm >= 16 and bm * bn / threads > 64):
                        continue
                    out.extend(
                        GemmConfig(bm, bn, bk, split, warps, stages)
                        for split in (1, 2, 4, 8)
                        for stages in (2, 3, 4)
                    )
    return out
