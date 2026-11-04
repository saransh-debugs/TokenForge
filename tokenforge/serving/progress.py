"""Progress-watchdog liveness probe (F12).

A hung engine still answers ``/health``; what it stops doing is producing tokens. Run as a
Kubernetes exec liveness probe next to any engine that exports Prometheus metrics:

    python -m tokenforge.serving.progress --metrics-url http://127.0.0.1:8000/metrics

Exit 0 while the engine is idle or generating. Exit 1 when requests are pending and the
generated-token counter has not moved for ``--stall-seconds`` (default 30), or when the
metrics endpoint does not answer. State between probe runs lives in a small JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families


@dataclass(frozen=True)
class ProbeConfig:
    stall_s: float = 30.0
    pending: tuple[str, ...] = ("vllm:num_requests_running", "vllm:num_requests_waiting")
    progress: str = "vllm:generation_tokens_total"


def read_metrics(text: str, names: set[str]) -> dict[str, float]:
    out = dict.fromkeys(names, 0.0)
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            if s.name in names:
                out[s.name] += s.value
    return out


def evaluate(text: str, state_path: Path, now: float, cfg: ProbeConfig) -> tuple[bool, str]:
    values = read_metrics(text, {*cfg.pending, cfg.progress})
    pending = sum(values[n] for n in cfg.pending)
    tokens = values[cfg.progress]
    try:
        state = json.loads(state_path.read_text())
        last_tokens, last_t = float(state["tokens"]), float(state["t"])
    except (OSError, ValueError, KeyError, TypeError):
        last_tokens, last_t = -1.0, now
    if tokens != last_tokens or pending == 0:
        last_tokens, last_t = tokens, now
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"tokens": last_tokens, "t": last_t}))
    os.replace(tmp, state_path)
    stalled_for = now - last_t
    if pending > 0 and stalled_for >= cfg.stall_s:
        return False, f"{pending:.0f} requests pending, no tokens for {stalled_for:.0f}s"
    return True, f"ok (pending={pending:.0f}, tokens={tokens:.0f})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    ap.add_argument("--stall-seconds", type=float, default=30.0)
    ap.add_argument("--state", default="/tmp/tokenforge-progress.json")
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args(argv)
    try:
        with urllib.request.urlopen(args.metrics_url, timeout=args.timeout) as resp:
            text = resp.read().decode()
    except OSError as exc:
        print(f"metrics unreachable: {exc}", file=sys.stderr)
        return 1
    ok, msg = evaluate(text, Path(args.state), time.time(), ProbeConfig(stall_s=args.stall_seconds))
    print(msg, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
