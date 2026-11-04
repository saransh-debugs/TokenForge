"""GPU spend ledger and the $75 hard stop.

    python -m tokenforge.ledger start --provider runpod --gpu L4 --rate 0.44
    python -m tokenforge.ledger stop [--notes "..."]
    python -m tokenforge.ledger check          # exit 2 at or above the provisioning stop
    python -m tokenforge.ledger report

Every rented session is a row in ``infra/ledger.csv``. An open session (started, not yet
stopped) counts toward the total at its accrued cost, so ``check`` cannot be dodged by
forgetting to stop. Provisioning stops at $70; the cap is $75.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CAP_USD = 75.0
PROVISION_STOP_USD = 70.0
FIELDS = (
    "session_id",
    "provider",
    "gpu",
    "start_utc",
    "end_utc",
    "hours",
    "usd_per_hour",
    "cost_usd",
    "notes",
)
ROOT = Path(__file__).resolve().parents[1]


def ledger_path() -> Path:
    return Path(os.environ.get("TOKENFORGE_LEDGER", ROOT / "infra" / "ledger.csv"))


def session_path() -> Path:
    return ledger_path().with_name(".session.json")


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


@dataclass(frozen=True)
class Totals:
    closed_usd: float
    open_usd: float

    @property
    def total(self) -> float:
        return self.closed_usd + self.open_usd


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: header {reader.fieldnames} != {list(FIELDS)}")
        return list(reader)


def open_session() -> dict[str, str] | None:
    try:
        return dict(json.loads(session_path().read_text()))
    except FileNotFoundError:
        return None


def totals(now: datetime | None = None) -> Totals:
    closed = sum(float(r["cost_usd"]) for r in read_rows(ledger_path()))
    sess = open_session()
    accrued = 0.0
    if sess is not None:
        hours = ((now or _now()) - _parse(sess["start_utc"])).total_seconds() / 3600
        accrued = max(0.0, hours) * float(sess["usd_per_hour"])
    return Totals(closed, accrued)


def append_row(row: dict[str, str]) -> None:
    path = ledger_path()
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def start(provider: str, gpu: str, rate: float, now: datetime | None = None) -> str:
    if open_session() is not None:
        raise RuntimeError("a session is already open; run `stop` first")
    t = totals(now)
    if t.total >= PROVISION_STOP_USD:
        raise BudgetExceededError(t.total)
    sid = uuid.uuid4().hex[:12]
    sess = {
        "session_id": sid,
        "provider": provider,
        "gpu": gpu,
        "start_utc": _iso(now or _now()),
        "usd_per_hour": f"{rate:.4f}",
    }
    session_path().write_text(json.dumps(sess))
    return sid


def stop(notes: str = "", now: datetime | None = None) -> dict[str, str]:
    sess = open_session()
    if sess is None:
        raise RuntimeError("no open session")
    end = now or _now()
    hours = max(0.0, (end - _parse(sess["start_utc"])).total_seconds() / 3600)
    row = {
        **sess,
        "end_utc": _iso(end),
        "hours": f"{hours:.4f}",
        "cost_usd": f"{hours * float(sess['usd_per_hour']):.4f}",
        "notes": notes,
    }
    append_row(row)
    session_path().unlink()
    return row


class BudgetExceededError(RuntimeError):
    def __init__(self, total: float) -> None:
        super().__init__(
            f"spend ${total:.2f} >= provisioning stop ${PROVISION_STOP_USD:.2f} "
            f"(cap ${CAP_USD:.2f})"
        )
        self.total = total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--provider", required=True)
    s.add_argument("--gpu", required=True)
    s.add_argument("--rate", type=float, required=True, help="USD per hour")
    p = sub.add_parser("stop")
    p.add_argument("--notes", default="")
    sub.add_parser("check")
    sub.add_parser("report")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "start":
            print(start(args.provider, args.gpu, args.rate))
        elif args.cmd == "stop":
            row = stop(args.notes)
            print(f"session {row['session_id']}: {row['hours']} h, ${row['cost_usd']}")
        else:
            t = totals()
            print(
                json.dumps(
                    {
                        "closed_usd": round(t.closed_usd, 2),
                        "open_usd": round(t.open_usd, 2),
                        "total_usd": round(t.total, 2),
                        "provision_stop_usd": PROVISION_STOP_USD,
                        "cap_usd": CAP_USD,
                        "remaining_usd": round(CAP_USD - t.total, 2),
                    }
                )
            )
            if args.cmd == "check" and t.total >= PROVISION_STOP_USD:
                return 2
    except BudgetExceededError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
