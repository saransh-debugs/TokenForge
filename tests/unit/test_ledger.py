"""The budget guard: spend is tracked per session and provisioning stops at $70."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tokenforge import ledger

T0 = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def ledger_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "ledger.csv"
    monkeypatch.setenv("TOKENFORGE_LEDGER", str(path))
    return path


def test_session_roundtrip_appends_row() -> None:
    ledger.start("runpod", "L4", 0.5, now=T0)
    row = ledger.stop("m1 bench", now=T0 + timedelta(hours=2))
    assert row["hours"] == "2.0000"
    assert row["cost_usd"] == "1.0000"
    rows = ledger.read_rows(ledger.ledger_path())
    assert [r["notes"] for r in rows] == ["m1 bench"]
    assert ledger.totals(T0).total == pytest.approx(1.0)


def test_open_session_counts_toward_total() -> None:
    ledger.start("runpod", "L4", 0.8, now=T0)
    assert ledger.totals(T0 + timedelta(hours=10)).open_usd == pytest.approx(8.0)


def test_refuses_to_provision_at_stop_threshold() -> None:
    ledger.start("runpod", "L4", 10.0, now=T0)
    ledger.stop(now=T0 + timedelta(hours=7))  # $70
    with pytest.raises(ledger.BudgetExceededError, match=r"\$70.00"):
        ledger.start("runpod", "L4", 0.4, now=T0 + timedelta(hours=8))
    assert ledger.main(["check"]) == 2


def test_cannot_double_start() -> None:
    ledger.start("a", "L4", 0.4, now=T0)
    with pytest.raises(RuntimeError, match="already open"):
        ledger.start("a", "L4", 0.4, now=T0)


def test_header_is_validated(ledger_file: Path) -> None:
    ledger_file.write_text("date,cost\n2026-11-01,3\n")
    with pytest.raises(ValueError, match="header"):
        ledger.totals()


def test_committed_ledger_parses() -> None:
    committed = Path(__file__).resolve().parents[2] / "infra" / "ledger.csv"
    rows = ledger.read_rows(committed)
    assert sum(float(r["cost_usd"]) for r in rows) < ledger.PROVISION_STOP_USD
