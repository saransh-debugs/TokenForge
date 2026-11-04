"""T5 serving contract (mock engine) plus the gateway-level failure modes F11-F19."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from typing import Any

import httpx
import pytest

from tests.stack import MODEL, StackFactory
from tokenforge.serving.mock_engine.engine import EngineConfig
from tokenforge.serving.tokenizer import load_tokenizer, render_chat

TOK = load_tokenizer("simple")


def _sse(resp: httpx.Response) -> list[Any]:
    """All SSE data payloads; JSON-decoded except the literal [DONE]."""
    out: list[Any] = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            data = line[6:]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


def _completion(**kw: Any) -> dict[str, Any]:
    return {"model": MODEL, "prompt": "Explain the KV cache in one line.", "max_tokens": 8, **kw}


# ------------------------------------------------------------------------- happy path


def test_completion_schema_and_usage(make_stack: StackFactory) -> None:
    s = make_stack()
    r = httpx.post(f"{s.gateway_url}/v1/completions", json=_completion(max_tokens=5))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"] == MODEL
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["choices"][0]["text"].strip()
    assert body["usage"] == {
        "prompt_tokens": len(TOK.encode(_completion()["prompt"])),
        "completion_tokens": 5,
        "total_tokens": len(TOK.encode(_completion()["prompt"])) + 5,
    }
    assert r.headers["x-request-id"]


def test_streaming_framing_and_usage(make_stack: StackFactory) -> None:
    s = make_stack()
    with httpx.stream(
        "POST",
        f"{s.gateway_url}/v1/completions",
        json=_completion(stream=True, max_tokens=6, stream_options={"include_usage": True}),
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _sse(r)
    assert events[-1] == "[DONE]"
    usage = events[-2]
    assert usage["choices"] == []
    assert usage["usage"]["completion_tokens"] == 6
    chunks = events[:-2]
    assert len(chunks) == 6
    assert all(c["object"] == "text_completion" for c in chunks)
    assert [c["choices"][0]["finish_reason"] for c in chunks][-1] == "length"
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])


def test_chat_stream_role_first_and_prompt_count(make_stack: StackFactory) -> None:
    s = make_stack()
    messages = [{"role": "user", "content": "Name three GPUs with 24 GB of memory."}]
    req = {
        "model": MODEL,
        "messages": messages,
        "max_completion_tokens": 4,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with httpx.stream("POST", f"{s.gateway_url}/v1/chat/completions", json=req) as r:
        events = _sse(r)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert all("role" not in e["choices"][0]["delta"] for e in events[1:-2])
    assert events[-2]["usage"]["prompt_tokens"] == len(TOK.encode(render_chat(TOK, messages)))
    assert s.gateway_metric("gateway_usage_mismatch_total") == 0


def test_stop_sequences_are_honored(make_stack: StackFactory) -> None:
    s = make_stack()
    stops = ["the", "of", "and", "to"]
    r = httpx.post(f"{s.gateway_url}/v1/completions", json=_completion(max_tokens=300, stop=stops))
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert not set(choice["text"].split()) & set(stops)


def test_request_id_and_traceparent_propagation(make_stack: StackFactory) -> None:
    s = make_stack()
    r = httpx.post(
        f"{s.gateway_url}/v1/completions", json=_completion(), headers={"x-request-id": "abc-123"}
    )
    assert r.headers["x-request-id"] == "abc-123"
    r = httpx.post(
        f"{s.gateway_url}/v1/completions",
        json=_completion(),
        headers={"x-request-id": "bad id with spaces!"},
    )
    assert r.headers["x-request-id"] != "bad id with spaces!"


# ------------------------------------------------------------------------- F16 validation


@pytest.mark.parametrize(
    ("body", "status", "needle"),
    [
        (_completion(max_tokens=0), 400, "max_tokens"),
        (_completion(max_tokens=10**9), 400, "exceeds the limit"),
        (_completion(prompt="word " * 9000), 400, "context"),
        (_completion(prompt=""), 400, "prompt"),
        (_completion(n=2), 400, "n"),
        (_completion(logprobs=5), 400, "logprobs"),
        (_completion(temperature=-1), 400, "temperature"),
        (_completion(stop=["a", "b", "c", "d", "e"]), 400, "stop"),
        (_completion(model="gpt-4"), 404, "does not exist"),
        ({"model": MODEL, "messages": []}, 400, "messages"),
    ],
)
def test_f16_invalid_requests_are_4xx_and_never_reach_engine(
    make_stack: StackFactory, body: dict[str, Any], status: int, needle: str
) -> None:
    s = make_stack()
    route = "chat/completions" if "messages" in body else "completions"
    r = httpx.post(f"{s.gateway_url}/v1/{route}", json=body)
    assert r.status_code == status, r.text
    assert needle in r.json()["error"]["message"]
    assert s.engine_metric("vllm:prompt_tokens_total") == 0


@pytest.mark.parametrize(
    "raw",
    [b"{not json", b"\xff\xfe\xfd", b"[]", b'{"model": "' + MODEL.encode() + b'", "prompt": 3}'],
)
def test_f16_malformed_bodies(make_stack: StackFactory, raw: bytes) -> None:
    s = make_stack()
    r = httpx.post(
        f"{s.gateway_url}/v1/completions", content=raw, headers={"content-type": "application/json"}
    )
    assert r.status_code == 400, r.text
    assert s.engine_metric("vllm:prompt_tokens_total") == 0


def test_f16_body_limit(make_stack: StackFactory) -> None:
    s = make_stack(max_body_bytes=1024)
    r = httpx.post(f"{s.gateway_url}/v1/completions", json=_completion(prompt="x" * 4096))
    assert r.status_code == 413


# ------------------------------------------------------------------------- retries (D7)


def test_retry_before_first_token_succeeds(make_stack: StackFactory) -> None:
    s = make_stack()
    s.faults(fail_next=1, fail_status=503)
    r = httpx.post(f"{s.gateway_url}/v1/completions", json=_completion())
    assert r.status_code == 200
    assert s.gateway_metric("gateway_retries_total") == 1


def test_retries_exhausted_is_retriable_503(make_stack: StackFactory) -> None:
    s = make_stack()
    s.faults(fail_next=2, fail_status=500)
    r = httpx.post(f"{s.gateway_url}/v1/completions", json=_completion(stream=True))
    assert r.status_code == 503
    assert r.headers["retry-after"] == "2"


def test_engine_4xx_is_not_retried(make_stack: StackFactory) -> None:
    s = make_stack()
    s.faults(fail_next=1, fail_status=400)
    r = httpx.post(f"{s.gateway_url}/v1/completions", json=_completion())
    assert r.status_code == 400
    assert s.gateway_metric("gateway_retries_total") == 0


# ------------------------------------------------------------------------- F13 crash


def test_f13_engine_death_mid_stream_ends_with_error_event(make_stack: StackFactory) -> None:
    s = make_stack()
    s.faults(crash_after_tokens=3)
    with httpx.stream(
        "POST", f"{s.gateway_url}/v1/completions", json=_completion(stream=True, max_tokens=50)
    ) as r:
        assert r.status_code == 200
        events = _sse(r)
    assert events[-1] != "[DONE]"
    assert events[-1]["error"]["type"] == "engine_error"
    assert any("choices" in e for e in events[:-1]), "tokens were streamed before the crash"
    assert s.gateway_metric("gateway_stream_terminations_total", type="engine_error") == 1


def test_f13_engine_death_non_stream_is_502(make_stack: StackFactory) -> None:
    s = make_stack()
    s.faults(crash_after_tokens=3)
    r = httpx.post(f"{s.gateway_url}/v1/completions", json=_completion(max_tokens=50))
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "engine_error"


# ------------------------------------------------------------------------- F14 disconnects


def _wait_for(pred: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_f14_client_disconnect_aborts_engine_request(make_stack: StackFactory) -> None:
    s = make_stack(engine=EngineConfig(tpot_base_s=0.02))
    clients = []
    for _ in range(8):
        c = httpx.Client(timeout=10)
        resp = c.send(
            c.build_request(
                "POST",
                f"{s.gateway_url}/v1/completions",
                json=_completion(stream=True, max_tokens=2000),
            ),
            stream=True,
        )
        it = resp.iter_lines()
        next(it)
        clients.append((c, resp, it))  # keep the iterator alive: dropping it closes the stream
    assert s.engine_metric("vllm:num_requests_running") == 8
    for c, resp, _ in clients:
        resp.close()
        c.close()
    assert _wait_for(lambda: s.engine_metric("vllm:num_requests_running") == 0), "leaked requests"
    assert _wait_for(lambda: s.engine_metric("vllm:kv_cache_usage_perc") == 0)
    assert _wait_for(
        lambda: s.gateway_metric("gateway_stream_terminations_total", type="client_disconnect") == 8
    )
    assert s.gateway_metric("gateway_inflight") == 0


# ------------------------------------------------------------------------- F15 slow reader


def test_f15_slow_reader_is_cut_off(make_stack: StackFactory) -> None:
    s = make_stack(
        engine=EngineConfig(tpot_base_s=0.0005, tpot_per_seq_s=0.0),
        write_timeout_s=0.3,
        max_tokens_limit=8000,
        gateway_sndbuf=4096,
    )
    host, port = s.gateway_url.removeprefix("http://").split(":")
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
    sock.connect((host, int(port)))
    body = json.dumps(_completion(stream=True, max_tokens=7000)).encode()
    sock.sendall(
        b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )
    try:
        # A healthy request completes normally while the slow reader is stalled.
        r = httpx.post(f"{s.gateway_url}/v1/completions", json=_completion(max_tokens=20))
        assert r.status_code == 200
        assert _wait_for(
            lambda: s.gateway_metric("gateway_stream_terminations_total", type="slow_reader") == 1,
            timeout=20,
        )
        assert _wait_for(lambda: s.engine_metric("vllm:num_requests_running") == 0)
    finally:
        sock.close()


# ------------------------------------------------------------------------- F11/F17 admission


async def test_f11_engine_queue_threshold_sheds_with_429(make_stack: StackFactory) -> None:
    s = make_stack(engine=EngineConfig(max_num_seqs=1, tpot_base_s=0.05), waiting_threshold=2)
    async with httpx.AsyncClient(timeout=30) as client:

        async def hold() -> None:
            async with client.stream(
                "POST",
                f"{s.gateway_url}/v1/completions",
                json=_completion(stream=True, max_tokens=500),
            ) as r:
                async for _ in r.aiter_lines():
                    pass

        held = [asyncio.create_task(hold()) for _ in range(4)]
        try:
            for _ in range(250):
                if s.engine_metric("vllm:num_requests_waiting") >= 2:
                    break
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.2)  # let the gateway's poller observe it
            r = await client.post(f"{s.gateway_url}/v1/completions", json=_completion())
            assert r.status_code == 429
            assert r.headers["retry-after"] == "2"
            assert s.gateway_metric("gateway_rejections_total", reason="engine_queue") >= 1
        finally:
            for t in held:
                t.cancel()
            await asyncio.gather(*held, return_exceptions=True)


def test_f11_gateway_queue_full_sheds_with_429(make_stack: StackFactory) -> None:
    s = make_stack(engine=EngineConfig(tpot_base_s=0.05), max_inflight=1, max_queue=0)
    c = httpx.Client(timeout=30)
    resp = c.send(
        c.build_request(
            "POST", f"{s.gateway_url}/v1/completions", json=_completion(stream=True, max_tokens=500)
        ),
        stream=True,
    )
    try:
        r = httpx.post(f"{s.gateway_url}/v1/completions", json=_completion())
        assert r.status_code == 429
        assert "gateway_queue" in r.json()["error"]["message"]
    finally:
        resp.close()
        c.close()


def test_f17_cold_engine_holds_then_serves(make_stack: StackFactory) -> None:
    s = make_stack(mock={"startup_delay_s": 1.0}, queue_timeout_s=5.0)
    t0 = time.monotonic()
    r = httpx.post(f"{s.gateway_url}/v1/completions", json=_completion(), timeout=10)
    assert r.status_code == 200
    assert time.monotonic() - t0 >= 0.5, "request should have waited for the engine"


def test_f17_cold_engine_times_out_with_503(make_stack: StackFactory) -> None:
    s = make_stack(mock={"startup_delay_s": 30.0}, queue_timeout_s=0.3)
    assert httpx.get(f"{s.gateway_url}/health/ready").status_code == 503
    r = httpx.post(f"{s.gateway_url}/v1/completions", json=_completion())
    assert r.status_code == 503
    assert "engine_not_ready" in r.json()["error"]["message"]
    assert r.headers["retry-after"] == "2"


# ------------------------------------------------------------------------- F19 tokenizer


def test_f19_tokenizer_mismatch_is_detected(make_stack: StackFactory) -> None:
    s = make_stack(tokenizer="simple:v2")
    prompt = "Batch 1024 requests at 3584 hidden size with 18944 intermediate."
    r = httpx.post(f"{s.gateway_url}/v1/completions", json=_completion(prompt=prompt))
    assert r.status_code == 200
    gateway_count = len(load_tokenizer("simple:v2").encode(prompt))
    assert r.json()["usage"]["prompt_tokens"] != gateway_count
    assert s.gateway_metric("gateway_usage_mismatch_total") == 1


# ------------------------------------------------------------------------- probes


def test_health_and_metrics_endpoints(make_stack: StackFactory) -> None:
    s = make_stack()
    assert httpx.get(f"{s.gateway_url}/health/live").status_code == 200
    assert httpx.get(f"{s.gateway_url}/health/ready").status_code == 200
    httpx.post(f"{s.gateway_url}/v1/completions", json=_completion())
    text = httpx.get(f"{s.gateway_url}/metrics").text
    for name in (
        "gateway_requests_total",
        "gateway_ttft_seconds",
        "gateway_queue_depth",
        "gateway_engine_waiting",
        "gateway_stream_terminations_total",
    ):
        assert name in text
    assert httpx.get(f"{s.gateway_url}/v1/models").json()["data"][0]["id"] == MODEL
