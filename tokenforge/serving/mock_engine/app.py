"""OpenAI-compatible HTTP surface for :class:`MockEngine` (the shape of vLLM's server)."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from prometheus_client import generate_latest

from tokenforge.serving import sse
from tokenforge.serving.mock_engine.engine import (
    EngineConfig,
    EngineCrashedError,
    EngineMetrics,
    MockEngine,
)
from tokenforge.serving.tokenizer import Tokenizer, load_tokenizer, render_chat


@dataclass
class MockSettings:
    model: str = "qwen2.5-7b-instruct"
    tokenizer: str = "simple"
    startup_delay_s: float = 0.0
    enable_faults: bool = False
    stall_s: float = 30.0
    engine: EngineConfig | None = None

    @classmethod
    def from_env(cls) -> MockSettings:
        e = os.environ.get
        cfg = EngineConfig(
            max_num_seqs=int(e("MOCK_MAX_NUM_SEQS", "64")),
            kv_blocks=int(e("MOCK_KV_BLOCKS", "4096")),
            block_size=int(e("MOCK_BLOCK_SIZE", "16")),
            max_model_len=int(e("MOCK_MAX_MODEL_LEN", "8192")),
            prefill_s_per_token=float(e("MOCK_PREFILL_S_PER_TOKEN", "2e-5")),
            tpot_base_s=float(e("MOCK_TPOT_BASE_S", "0.010")),
            tpot_per_seq_s=float(e("MOCK_TPOT_PER_SEQ_S", "2e-4")),
        )
        return cls(
            model=e("MOCK_MODEL", "qwen2.5-7b-instruct"),
            tokenizer=e("MOCK_TOKENIZER", "simple"),
            startup_delay_s=float(e("MOCK_STARTUP_DELAY_S", "0")),
            enable_faults=e("MOCK_ENABLE_FAULTS", "0") == "1",
            stall_s=float(e("MOCK_STALL_S", "30")),
            engine=cfg,
        )


def _err(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": kind, "code": status}}, status_code=status
    )


def create_app(settings: MockSettings | None = None) -> FastAPI:
    st = settings or MockSettings.from_env()
    metrics = EngineMetrics(st.model)
    tok: Tokenizer = load_tokenizer(st.tokenizer)
    state: dict[str, Any] = {"ready_at": time.monotonic() + st.startup_delay_s}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        engine = MockEngine(st.engine or EngineConfig(), metrics)
        engine.start()
        state["engine"] = engine
        yield
        await engine.stop()

    app = FastAPI(title="tokenforge-mock-engine", lifespan=lifespan)

    def engine() -> MockEngine:
        return state["engine"]  # type: ignore[no-any-return]

    def ready() -> bool:
        return time.monotonic() >= state["ready_at"] and not engine().crashed

    @app.get("/health")
    async def health() -> Response:
        if not ready():
            return PlainTextResponse("starting", status_code=503)
        if engine().progress_stalled(st.stall_s):
            return PlainTextResponse("not progressing", status_code=503)
        return PlainTextResponse("ok")

    @app.get("/metrics")
    async def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": st.model, "object": "model"}]}

    @app.post("/admin/faults")
    async def faults(body: dict[str, Any]) -> dict[str, Any]:
        if not st.enable_faults:
            raise HTTPException(404)
        f = engine().faults
        for key, value in body.items():
            if not hasattr(f, key):
                raise HTTPException(400, f"unknown fault {key!r}")
            setattr(f, key, value)
        return {"faults": vars(f)}

    async def _generate(request: Request, chat: bool) -> Response:
        if not ready():
            return _err(503, "engine is starting", "server_error")
        body = await request.json()
        if chat:
            prompt = render_chat(tok, body.get("messages", []))
            limit = body.get("max_completion_tokens") or body.get("max_tokens") or 16
        else:
            prompt = body.get("prompt", "")
            limit = body.get("max_tokens") or 16
        n_prompt = len(tok.encode(prompt))
        max_len = engine().cfg.max_model_len
        if n_prompt + int(limit) > max_len:
            return _err(400, f"prompt ({n_prompt}) + max_tokens ({limit}) > {max_len}")
        f = engine().faults
        if f.fail_next > 0:
            f.fail_next -= 1
            return _err(f.fail_status, "injected failure", "server_error")
        stop = body.get("stop") or ()
        stop = (stop,) if isinstance(stop, str) else tuple(stop)
        req = engine().submit(n_prompt, int(limit), stop)
        created = int(time.time())
        obj = "chat.completion.chunk" if chat else "text_completion"

        def chunk(text: str | None, finish: str | None, first: bool) -> dict[str, Any]:
            if chat:
                delta: dict[str, Any] = {"role": "assistant"} if first else {}
                if text:
                    delta["content"] = text
                choice: dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text or "", "logprobs": None, "finish_reason": finish}
            return {
                "id": req.rid,
                "object": obj,
                "created": created,
                "model": st.model,
                "choices": [choice],
            }

        def usage() -> dict[str, int]:
            return {
                "prompt_tokens": n_prompt,
                "completion_tokens": req.generated,
                "total_tokens": n_prompt + req.generated,
            }

        async def events() -> AsyncIterator[tuple[str, str | None]]:
            while True:
                item = await req.out.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item[1] is not None:
                    return

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

            async def stream() -> AsyncIterator[bytes]:
                first = True
                try:
                    async for text, finish in events():
                        yield sse.event(chunk(text, finish, first))
                        first = False
                    if include_usage:
                        yield sse.event(
                            {
                                "id": req.rid,
                                "object": obj,
                                "created": created,
                                "model": st.model,
                                "choices": [],
                                "usage": usage(),
                            }
                        )
                    yield sse.event(sse.DONE)
                except EngineCrashedError:
                    raise  # connection drops mid-stream, like a killed process
                finally:
                    engine().abort(req)

            return StreamingResponse(stream(), media_type="text/event-stream")

        try:
            parts, finish = [], None
            async for text, fin in events():
                parts.append(text)
                finish = fin
        except asyncio.CancelledError:
            engine().abort(req)
            raise
        except EngineCrashedError:
            return _err(500, "engine died", "server_error")
        text = "".join(parts)
        choice = (
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
            if chat
            else {"index": 0, "text": text, "logprobs": None, "finish_reason": finish}
        )
        return JSONResponse(
            {
                "id": req.rid,
                "object": "chat.completion" if chat else "text_completion",
                "created": created,
                "model": st.model,
                "choices": [choice],
                "usage": usage(),
            }
        )

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _generate(request, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _generate(request, chat=True)

    app.state.engine_state = state
    return app
