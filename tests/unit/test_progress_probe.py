"""F12 detection logic: the exec liveness probe fails only on stalled pending work."""

from __future__ import annotations

from pathlib import Path

from tokenforge.serving.progress import ProbeConfig, evaluate

CFG = ProbeConfig(stall_s=30)


def _metrics(running: int, waiting: int, tokens: int) -> str:
    return (
        f'vllm:num_requests_running{{model_name="m"}} {running}\n'
        f'vllm:num_requests_waiting{{model_name="m"}} {waiting}\n'
        "# TYPE vllm:generation_tokens counter\n"
        f'vllm:generation_tokens_total{{model_name="m"}} {tokens}\n'
    )


def test_idle_engine_is_alive_forever(tmp_path: Path) -> None:
    state = tmp_path / "s.json"
    for t in range(0, 600, 60):
        assert evaluate(_metrics(0, 0, 100), state, float(t), CFG)[0]


def test_progressing_engine_is_alive(tmp_path: Path) -> None:
    state = tmp_path / "s.json"
    for i, t in enumerate(range(0, 300, 10)):
        assert evaluate(_metrics(4, 2, 100 + i), state, float(t), CFG)[0]


def test_stalled_engine_fails_after_threshold(tmp_path: Path) -> None:
    state = tmp_path / "s.json"
    assert evaluate(_metrics(4, 10, 500), state, 0.0, CFG)[0]
    assert evaluate(_metrics(4, 10, 500), state, 20.0, CFG)[0]
    ok, reason = evaluate(_metrics(4, 10, 500), state, 31.0, CFG)
    assert not ok
    assert "no tokens" in reason
    # Recovery: tokens move again (e.g. after restart) -> alive.
    assert evaluate(_metrics(4, 10, 501), state, 32.0, CFG)[0]


def test_corrupt_state_resets(tmp_path: Path) -> None:
    state = tmp_path / "s.json"
    state.write_text("{garbage")
    assert evaluate(_metrics(1, 0, 5), state, 100.0, CFG)[0]
