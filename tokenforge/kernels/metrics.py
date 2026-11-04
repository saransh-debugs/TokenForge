"""Kernel-path counters.

``tokenforge_kernel_calls_total{kernel,bucket}`` proves the custom path is running and
``tokenforge_fallback_total{reason}`` makes every fallback visible (a silent fallback would
invalidate a benchmark). Counters are host-side: they count dispatch decisions. Under CUDA
graph replay the host code does not run, so captured calls are counted once, at capture.

When ``prometheus_client`` is importable the counters are also exported through it; in vLLM
that lands in the multiprocess registry the API server scrapes.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


class LabeledCounter:
    def __init__(self, name: str, documentation: str, labelnames: tuple[str, ...]) -> None:
        self.name = name
        self.labelnames = labelnames
        self._values: defaultdict[tuple[str, ...], float] = defaultdict(float)
        self._lock = threading.Lock()
        self._prom: Any = None
        try:
            import prometheus_client

            self._prom = prometheus_client.Counter(name, documentation, labelnames)
        except ImportError:
            pass
        except ValueError:  # already registered (module re-import in tests)
            import prometheus_client

            self._prom = prometheus_client.REGISTRY._names_to_collectors.get(name)

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        if set(labels) != set(self.labelnames):
            raise ValueError(f"{self.name} expects labels {self.labelnames}, got {sorted(labels)}")
        return tuple(str(labels[k]) for k in self.labelnames)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] += amount
        if self._prom is not None:
            self._prom.labels(*key).inc(amount)

    def value(self, **labels: str) -> float:
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def total(self) -> float:
        with self._lock:
            return sum(self._values.values())

    def snapshot(self) -> dict[tuple[str, ...], float]:
        with self._lock:
            return dict(self._values)

    def reset(self) -> None:
        """Clear the in-process view (tests only; Prometheus counters are monotonic)."""
        with self._lock:
            self._values.clear()


KERNEL_CALLS = LabeledCounter(
    "tokenforge_kernel_calls",
    "TokenForge kernel dispatches by kernel and M bucket",
    ("kernel", "bucket"),
)
FALLBACKS = LabeledCounter(
    "tokenforge_fallback",
    "TokenForge fallbacks away from the requested kernel path, by reason",
    ("reason",),
)
