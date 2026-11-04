"""GPU CI tiers and the report that CPU CI verifies (ADR 0008).

    python -m tokenforge.ci.gpu_report run --tier dev     # Kaggle T4 (GPU-dev)
    python -m tokenforge.ci.gpu_report run --tier full    # rented L4 (GPU-full, `make gpu-ci`)
    python -m tokenforge.ci.gpu_report check              # CPU CI: a passing report exists

GitHub's free runners have no GPU, so the GPU tiers run on Kaggle or the L4 and write
``bench/results/ci/<tier>-<code_tree[:16]>.json`` with the environment fingerprint and a
pass/fail per step. The report is keyed by the hash of the git trees it vouches for, so
committing it does not invalidate it. ``check`` fails a PR whose kernel or plugin code has
no passing report for its exact content.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from tokenforge.bench.env import CODE_PATHS, ROOT, code_tree_hash, fingerprint, git_dirty

REPORT_DIR = ROOT / "bench" / "results" / "ci"
SANITIZER_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")


def _pytest(
    name: str, args: list[str], env: dict[str, str], prefix: list[str] | None = None
) -> dict[str, Any]:
    junit = REPORT_DIR / f".{name}.xml"
    cmd = [*(prefix or []), sys.executable, "-m", "pytest", *args, f"--junitxml={junit}", "-q"]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, check=False)
    step: dict[str, Any] = {
        "name": name,
        "cmd": " ".join(cmd),
        "exit_code": proc.returncode,
        "seconds": round(time.monotonic() - t0, 1),
    }
    if junit.exists():
        suite = ET.parse(junit).getroot()
        suite = suite if suite.tag == "testsuite" else suite[0]
        step.update({k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")})
        junit.unlink()
    step["passed"] = proc.returncode == 0
    return step


def _command(name: str, cmd: list[str]) -> dict[str, Any]:
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    return {
        "name": name,
        "cmd": " ".join(cmd),
        "exit_code": proc.returncode,
        "seconds": round(time.monotonic() - t0, 1),
        "passed": proc.returncode == 0,
    }


def steps_for(tier: str) -> list[dict[str, Any]]:
    full = {"TOKENFORGE_SHAPE_SUITE": "full"}
    out = [
        _pytest("unit", ["tests/unit"], {}),
        _pytest("numerics", ["tests/numerics"], full),
    ]
    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None:
        out.append(
            {"name": "compute-sanitizer", "passed": False, "error": "compute-sanitizer not on PATH"}
        )
    else:
        prefix = [sanitizer, "--error-exitcode=99", "--target-processes=all"]
        out.extend(
            _pytest(
                f"sanitizer-{tool}",
                ["tests/numerics/test_w4a16.py", "-k", "shape_suite"],
                full,
                prefix=[*prefix, f"--tool={tool}"],
            )
            for tool in SANITIZER_TOOLS
        )
    bench_suite = "t4" if tier == "dev" else "l4"
    out.append(
        _command(
            "microbench",
            [sys.executable, "-m", "tokenforge.bench.micro", "--suite", bench_suite, "--check"],
        )
    )
    extra = ("tests/model", "tests/integration") if tier == "full" else ()
    out.extend(
        _pytest(path.split("/")[-1], [path], full)
        for path in extra
        if any((ROOT / path).glob("test_*.py"))
    )
    return out


def run(tier: str) -> Path:
    if git_dirty(CODE_PATHS):
        raise SystemExit("refusing: uncommitted changes under " + ", ".join(CODE_PATHS))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    steps = steps_for(tier)
    env = fingerprint()
    report: dict[str, Any] = {
        "tier": tier,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": env,
        "steps": steps,
        "passed": all(s["passed"] for s in steps),
    }
    path = REPORT_DIR / f"{tier}-{env['code_tree'][:16]}.json"
    path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {path.relative_to(ROOT)}")
    return path


def check(rev: str, required_tier: str) -> int:
    tree = code_tree_hash(rev)
    path = REPORT_DIR / f"{required_tier}-{tree[:16]}.json"
    if not path.exists():
        print(
            f"missing GPU report {path.relative_to(ROOT)} for code tree {tree[:16]}; run "
            f"`python -m tokenforge.ci.gpu_report run --tier {required_tier}` on a GPU box "
            "and commit the report",
            file=sys.stderr,
        )
        return 1
    report = json.loads(path.read_text())
    if report["env"]["code_tree"] != tree:
        print(f"{path.name}: code tree mismatch", file=sys.stderr)
        return 1
    if not report["passed"]:
        failed = [s["name"] for s in report["steps"] if not s["passed"]]
        print(f"{path.name}: failing steps {failed}", file=sys.stderr)
        return 1
    print(f"ok: {path.relative_to(ROOT)} ({report['env'].get('gpu', '?')})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tier", choices=["dev", "full"], required=True)
    c = sub.add_parser("check")
    c.add_argument("--rev", default="HEAD")
    c.add_argument("--tier", choices=["dev", "full"], default="dev")
    args = ap.parse_args()
    if args.cmd == "run":
        return 0 if json.loads(run(args.tier).read_text())["passed"] else 1
    return check(args.rev, args.tier)


if __name__ == "__main__":
    raise SystemExit(main())
