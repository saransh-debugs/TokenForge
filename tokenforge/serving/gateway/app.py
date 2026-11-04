"""The TokenForge gateway: an OpenAI-compatible front door for the inference engines.

Request lifecycle:

1. **Validate** (400): schema, model name, ``max_tokens`` bounds, prompt + max_tokens vs the
   context window, counted with the engine's tokenizer. Invalid requests never reach the
   engine.
2. **Admit** (429 / 503): see :mod:`tokenforge.serving.gateway.admission`.
3. **Start**: stream from the engine until the first token. Failures here are retried
   (ADR D7: only before the first token) and otherwise mapped to an HTTP status, because the
   response has not started.
4. **Relay**: forward tokens one chunk at a time with a write timeout. A client disconnect
   or a too-slow reader closes the upstream stream, which aborts the engine request. An
   engine failure mid-stream ends the stream with a terminal SSE ``error`` event, never a
   silent close and never ``[DONE]``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families
from pydantic import ValidationError

from tokenforge.serving import sse
from tokenforge.serving.gateway.admission import (
    AdmissionController,
    OverloadedError,
    QueueTimeoutError,
    Ticket,
)
from tokenforge.serving.gateway.engine_client import Delta, EngineError, Event, OpenAIEngine, Usage
from tokenforge.serving.gateway.observability import (
    GatewayMetrics,
    Stopwatch,
    child_traceparent,
    request_id_from,
    request_id_var,
    trace_id_var,
)
from tokenforge.serving.gateway.schemas import ChatCompletionRequest, CompletionRequest
from tokenforge.serving.gateway.settings import GatewaySettings
from tokenforge.serving.tokenizer import Tokenizer, load_tokenizer, render_chat

log = logging.getLogger("tokenforge.gateway")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


def error_body(status: int, message: str, kind: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": kind, "code": status}}


def error_response(
    status: int, message: str, kind: str, retry_after: int | None = None
) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    return JSONResponse(error_body(status, message, kind), status_code=status, headers=headers)


def parse_metric(text: str, name: str) -> float:
    """Sum of all samples of ``name`` (any labels) in Prometheus text exposition."""
    total = 0.0
    found = False
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == name:
                total += sample.value
                found = True
    if not found:
        raise KeyError(name)
    return total


class BodyLimitMiddleware:
    """Reject request bodies above ``limit`` bytes with 413, streamed or not."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]], limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.limit:
            await error_response(413, f"body exceeds {self.limit} bytes", "invalid_request_error")(
                scope, receive, send
            )
            return
        seen = 0

        async def limited() -> Message:
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.limit:
                    raise _BodyTooLargeError
            return msg

        try:
            await self.app(scope, limited, send)
        except _BodyTooLargeError:
            await error_response(413, f"body exceeds {self.limit} bytes", "invalid_request_error")(
                scope, receive, send
            )


class _BodyTooLargeError(Exception):
    pass


class RequestContextMiddleware:
    """Pure ASGI (no response buffering): request id + trace id into contextvars and headers."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        hdrs = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers") or []}
        rid = request_id_from(hdrs.get("x-request-id"))
        trace_id, _ = child_traceparent(hdrs.get("traceparent"))
        t1 = request_id_var.set(rid)
        t2 = trace_id_var.set(trace_id)

        async def send_with_id(msg: Message) -> None:
            if msg["type"] == "http.response.start":
                headers = list(msg.get("headers") or [])
                if not any(k.lower() == b"x-request-id" for k, _ in headers):
                    headers.append((b"x-request-id", rid.encode()))
                msg["headers"] = headers
            await send(msg)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(t1)
            trace_id_var.reset(t2)


class Gateway:
    def __init__(self, settings: GatewaySettings, tokenizer: Tokenizer) -> None:
        self.s = settings
        self.tok = tokenizer
        self.metrics = GatewayMetrics()
        self.admission = AdmissionController(
            settings.max_inflight,
            settings.max_queue,
            settings.waiting_threshold,
            settings.queue_timeout_s,
        )
        self.client: httpx.AsyncClient | None = None
        self.engine: OpenAIEngine | None = None

    # ------------------------------------------------------------ engine state
    async def poll_engine(self) -> None:
        assert self.client is not None
        ready = False
        waiting: float | None = None
        try:
            r = await self.client.get(
                self.s.engine_url.rstrip("/") + self.s.engine_health_path, timeout=2.0
            )
            ready = r.status_code == 200
        except httpx.HTTPError:
            ready = False
        if ready:
            values = []
            for url in self.s.metrics_urls:
                try:
                    text = (await self.client.get(url, timeout=2.0)).text
                    values.append(parse_metric(text, self.s.engine_waiting_metric))
                except (httpx.HTTPError, KeyError, ValueError):
                    continue
            waiting = sum(values) if values else None
        self.admission.set_engine_state(ready=ready, waiting=waiting)
        self.metrics.engine_ready.set(1.0 if ready else 0.0)
        if waiting is not None:
            self.metrics.engine_waiting.set(waiting)

    async def poll_forever(self) -> None:
        while True:
            try:
                await self.poll_engine()
            except Exception:
                log.exception("engine poll failed")
            await asyncio.sleep(self.s.poll_interval_s)

    # ------------------------------------------------------------ request path
    async def handle(self, request: Request, chat: bool) -> Response:
        route = "chat" if chat else "completions"
        sw = Stopwatch()
        resp = await self._handle(request, chat, route, sw)
        if not isinstance(resp, RelayResponse):
            self.metrics.requests.labels(route, str(resp.status_code)).inc()
            self.metrics.duration.labels(route).observe(sw.elapsed())
            log.info(
                "request",
                extra={
                    "fields": {
                        "route": route,
                        "status": resp.status_code,
                        "duration_ms": round(sw.elapsed() * 1e3, 1),
                    }
                },
            )
        return resp

    async def _handle(self, request: Request, chat: bool, route: str, sw: Stopwatch) -> Response:
        try:
            body = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return error_response(400, f"invalid JSON: {exc}", "invalid_request_error")
        try:
            req: CompletionRequest | ChatCompletionRequest = (
                ChatCompletionRequest if chat else CompletionRequest
            ).model_validate(body)
            stops = req.stop_list()
        except ValidationError as exc:
            errs = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
            )
            return error_response(400, errs or "invalid request", "invalid_request_error")
        except ValueError as exc:
            return error_response(400, str(exc), "invalid_request_error")
        if req.model != self.s.served_model:
            return error_response(404, f"model {req.model!r} does not exist", "model_not_found")

        if isinstance(req, ChatCompletionRequest):
            prompt_text = render_chat(self.tok, [m.model_dump() for m in req.messages])
        else:
            prompt_text = req.prompt
        n_prompt = len(self.tok.encode(prompt_text))
        requested = req.requested_max_tokens()
        if requested is not None and requested > self.s.max_tokens_limit:
            return error_response(
                400,
                f"max_tokens={requested} exceeds the limit {self.s.max_tokens_limit}",
                "invalid_request_error",
            )
        max_tokens = requested or min(self.s.default_max_tokens, self.s.max_context - n_prompt)
        if max_tokens < 1 or n_prompt + max_tokens > self.s.max_context:
            return error_response(
                400,
                f"prompt is {n_prompt} tokens; with max_tokens={max_tokens} it exceeds the "
                f"{self.s.max_context}-token context",
                "invalid_request_error",
            )

        payload = req.model_dump(
            exclude_none=True, exclude={"stream_options", "max_completion_tokens", "stop"}
        )
        payload.update(
            model=self.s.engine_model,
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
        )
        if stops:
            payload["stop"] = stops
        _, traceparent = child_traceparent(request.headers.get("traceparent"))
        headers = {
            "x-request-id": request_id_var.get(),
            "traceparent": traceparent,
            **self.s.extra_headers,
        }

        self.metrics.queue_depth.set(self.admission.queued + 1)
        try:
            ticket = await self.admission.acquire()
        except OverloadedError as exc:
            self.metrics.rejections.labels(exc.reason).inc()
            return error_response(
                429, f"overloaded ({exc.reason})", "rate_limit_error", self.s.retry_after_s
            )
        except QueueTimeoutError as exc:
            self.metrics.rejections.labels(exc.reason).inc()
            return error_response(
                503, f"no capacity ({exc.reason})", "server_error", self.s.retry_after_s
            )
        finally:
            self.metrics.queue_depth.set(self.admission.queued)
        self.metrics.inflight.set(self.admission.inflight)

        try:
            prefix, it = await self._start(chat, payload, headers)
        except EngineError as exc:
            ticket.release()
            self.metrics.inflight.set(self.admission.inflight)
            self.metrics.terminations.labels("engine_error_before_first_token").inc()
            if exc.status is not None and 400 <= exc.status < 500 and exc.status not in (408, 429):
                return error_response(exc.status, str(exc), "invalid_request_error")
            return error_response(503, str(exc), "server_error", self.s.retry_after_s)
        return RelayResponse(
            gw=self,
            route=route,
            chat=chat,
            stream=req.stream,
            include_usage=bool(req.stream_options and req.stream_options.include_usage),
            prefix=prefix,
            it=it,
            ticket=ticket,
            n_prompt=n_prompt,
            sw=sw,
        )

    async def _start(
        self, chat: bool, payload: dict[str, Any], headers: dict[str, str]
    ) -> tuple[list[Event], AsyncIterator[Event]]:
        """Open the engine stream and read up to the first token, retrying failures."""
        assert self.engine is not None
        attempt = 0
        while True:
            it = self.engine.stream(chat, payload, headers).__aiter__()
            buffered: list[Event] = []
            try:
                while True:
                    ev = await it.__anext__()
                    buffered.append(ev)
                    if isinstance(ev, Delta) and (ev.text or ev.finish_reason):
                        return buffered, it
            except StopAsyncIteration:
                return buffered, it
            except EngineError as exc:
                await it.aclose()  # type: ignore[attr-defined]
                if not exc.retriable or attempt >= self.s.retries_before_first_token:
                    raise
                attempt += 1
                self.metrics.retries.inc()
                await asyncio.sleep(self.s.retry_backoff_s)


class RelayResponse(Response):
    """ASGI response that relays engine events to the client (stream) or aggregates them."""

    def __init__(
        self,
        *,
        gw: Gateway,
        route: str,
        chat: bool,
        stream: bool,
        include_usage: bool,
        prefix: list[Event],
        it: AsyncIterator[Event],
        ticket: Ticket,
        n_prompt: int,
        sw: Stopwatch,
    ) -> None:
        super().__init__(status_code=200)
        self.gw, self.route, self.chat, self.stream = gw, route, chat, stream
        self.include_usage = include_usage
        self.prefix, self.it, self.ticket = prefix, it, ticket
        self.n_prompt, self.sw = n_prompt, sw
        self.rid = request_id_var.get()
        self.created = int(time.time())

    def _chunk(self, ev: Delta, first: bool) -> dict[str, Any]:
        model = self.gw.s.served_model
        if self.chat:
            delta: dict[str, Any] = {"role": "assistant"} if first else {}
            if ev.text:
                delta["content"] = ev.text
            choice: dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": ev.finish_reason}
            obj = "chat.completion.chunk"
        else:
            choice = {
                "index": 0,
                "text": ev.text,
                "logprobs": None,
                "finish_reason": ev.finish_reason,
            }
            obj = "text_completion"
        return {
            "id": self.rid,
            "object": obj,
            "created": self.created,
            "model": model,
            "choices": [choice],
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        m = self.gw.metrics
        disconnected = asyncio.Event()

        async def watch() -> None:
            while True:
                msg = await receive()
                if msg["type"] == "http.disconnect":
                    disconnected.set()
                    return

        watcher = asyncio.create_task(watch())
        termination = "completed"
        status = 200
        usage: Usage | None = None
        parts: list[str] = []
        finish: str | None = None
        first_sent = False
        headers = [(b"x-request-id", self.rid.encode())]

        async def write(data: bytes) -> bool:
            nonlocal termination
            if disconnected.is_set():
                termination = "client_disconnect"
                return False
            try:
                await asyncio.wait_for(
                    send({"type": "http.response.body", "body": data, "more_body": True}),
                    self.gw.s.write_timeout_s,
                )
            except TimeoutError:
                termination = "slow_reader"
                return False
            except OSError:
                termination = "client_disconnect"
                return False
            return True

        try:
            if self.stream:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"content-type", b"text/event-stream"),
                            (b"cache-control", b"no-cache"),
                            *headers,
                        ],
                    }
                )
            pending = list(self.prefix)
            while True:
                if pending:
                    ev: Event = pending.pop(0)
                else:
                    nxt = asyncio.ensure_future(self.it.__anext__())
                    disc = asyncio.ensure_future(disconnected.wait())
                    done, _ = await asyncio.wait({nxt, disc}, return_when=asyncio.FIRST_COMPLETED)
                    disc.cancel()
                    if nxt not in done:
                        nxt.cancel()
                        with contextlib.suppress(BaseException):
                            await nxt
                        termination = "client_disconnect"
                        break
                    try:
                        ev = nxt.result()
                    except StopAsyncIteration:
                        break
                    except EngineError as exc:
                        termination = "engine_error"
                        if self.stream:
                            await write(sse.event(error_body(502, str(exc), "engine_error")))
                        break
                if isinstance(ev, Usage):
                    usage = ev
                    continue
                if not ev.text and ev.finish_reason is None:
                    continue
                if ev.finish_reason is not None:
                    finish = ev.finish_reason
                if self.stream:
                    if not await write(sse.event(self._chunk(ev, first=not first_sent))):
                        break
                else:
                    parts.append(ev.text)
                if not first_sent:
                    first_sent = True
                    m.ttft.labels(self.route).observe(self.sw.elapsed())

            if self.stream:
                if termination == "completed":
                    if self.include_usage and usage is not None:
                        await write(
                            sse.event(
                                {
                                    "id": self.rid,
                                    "object": "chat.completion.chunk"
                                    if self.chat
                                    else "text_completion",
                                    "created": self.created,
                                    "model": self.gw.s.served_model,
                                    "choices": [],
                                    "usage": _usage_dict(usage),
                                }
                            )
                        )
                    await write(sse.event(sse.DONE))
                if termination in ("completed", "engine_error"):
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
            else:
                if termination == "completed":
                    body = self._final_body("".join(parts), finish, usage)
                    payload = json.dumps(body).encode()
                else:
                    status = 502
                    payload = json.dumps(
                        error_body(502, "engine failed mid-generation", "engine_error")
                    ).encode()
                if termination != "client_disconnect":
                    await send(
                        {
                            "type": "http.response.start",
                            "status": status,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(payload)).encode()),
                                *headers,
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": payload, "more_body": False})
        finally:
            watcher.cancel()
            with contextlib.suppress(BaseException):
                await self.it.aclose()  # type: ignore[attr-defined]
            self.ticket.release()
            m.inflight.set(self.gw.admission.inflight)
            m.terminations.labels(termination).inc()
            m.requests.labels(self.route, str(status)).inc()
            m.duration.labels(self.route).observe(self.sw.elapsed())
            fields: dict[str, Any] = {
                "route": self.route,
                "status": status,
                "termination": termination,
                "stream": self.stream,
                "prompt_tokens": self.n_prompt,
                "duration_ms": round(self.sw.elapsed() * 1e3, 1),
            }
            if usage is not None:
                fields["completion_tokens"] = usage.completion_tokens
                fields["engine_prompt_tokens"] = usage.prompt_tokens
                m.prompt_tokens.inc(usage.prompt_tokens)
                m.completion_tokens.inc(usage.completion_tokens)
                if usage.prompt_tokens != self.n_prompt:
                    m.usage_mismatch.inc()
                    log.warning("usage mismatch", extra={"fields": fields})
            log.info("request", extra={"fields": fields})
        if termination == "slow_reader":
            raise SlowReaderError  # makes the server drop the connection

    def _final_body(self, text: str, finish: str | None, usage: Usage | None) -> dict[str, Any]:
        model = self.gw.s.served_model
        if self.chat:
            choice: dict[str, Any] = {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": text},
            }
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish}
            obj = "text_completion"
        body: dict[str, Any] = {
            "id": self.rid,
            "object": obj,
            "created": self.created,
            "model": model,
            "choices": [choice],
        }
        if usage is not None:
            body["usage"] = _usage_dict(usage)
        return body


class SlowReaderError(Exception):
    """Raised after cleanup so the ASGI server closes a connection that stopped reading."""


def _usage_dict(u: Usage) -> dict[str, int]:
    return {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
    }


def create_app(settings: GatewaySettings | None = None) -> FastAPI:
    s = settings or GatewaySettings.from_env()
    gw = Gateway(s, load_tokenizer(s.tokenizer))

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(
            connect=s.connect_timeout_s,
            read=s.read_timeout_s,
            write=s.write_timeout_s,
            pool=s.queue_timeout_s,
        )
        limits = httpx.Limits(
            max_connections=s.max_inflight + 16, max_keepalive_connections=min(64, s.max_inflight)
        )
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            gw.client = client
            gw.engine = OpenAIEngine(s.engine_url, client)
            await gw.poll_engine()
            poller = asyncio.create_task(gw.poll_forever())
            try:
                yield
            finally:
                poller.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poller

    app = FastAPI(title="tokenforge-gateway", lifespan=lifespan)
    app.state.gateway = gw

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await gw.handle(request, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await gw.handle(request, chat=True)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": s.served_model, "object": "model", "owned_by": "tokenforge"}],
        }

    @app.get("/health/live")
    async def live() -> Response:
        return PlainTextResponse("ok")

    @app.get("/health/ready")
    async def ready() -> Response:
        if gw.admission.engine_ready.is_set():
            return PlainTextResponse("ok")
        return PlainTextResponse("engine not ready", status_code=503)

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            generate_latest(gw.metrics.registry), media_type="text/plain; version=0.0.4"
        )

    app.add_middleware(BodyLimitMiddleware, limit=s.max_body_bytes)
    app.add_middleware(RequestContextMiddleware)
    return app
