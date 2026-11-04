"""Gateway metrics (RED + queue + stream terminations), JSON logs and trace context.

Logs carry the request id and trace id on every line and never the prompt or completion
text, only lengths and ids.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import secrets
import time
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")

_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def request_id_from(header: str | None) -> str:
    if header and _REQUEST_ID.match(header):
        return header
    return secrets.token_hex(12)


def child_traceparent(header: str | None) -> tuple[str, str]:
    """Return ``(trace_id, traceparent for the engine call)`` per W3C Trace Context."""
    m = _TRACEPARENT.match(header or "")
    trace_id = m.group(1) if m and m.group(1) != "0" * 32 else secrets.token_hex(16)
    flags = m.group(3) if m else "01"
    return trace_id, f"00-{trace_id}-{secrets.token_hex(8)}-{flags}"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "trace_id": trace_id_var.get(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            doc.update(fields)
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, separators=(",", ":"))


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("tokenforge")
    root.handlers = [handler]
    root.setLevel(level)
    root.propagate = False


LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)


class GatewayMetrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.requests = Counter(
            "gateway_requests", "Requests by route and status code", ["route", "code"], registry=r
        )
        self.duration = Histogram(
            "gateway_request_duration_seconds",
            "End-to-end duration",
            ["route"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.ttft = Histogram(
            "gateway_ttft_seconds",
            "Arrival to first streamed token",
            ["route"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.queue_depth = Gauge("gateway_queue_depth", "Requests waiting for a slot", registry=r)
        self.inflight = Gauge("gateway_inflight", "Requests holding a slot", registry=r)
        self.engine_waiting = Gauge(
            "gateway_engine_waiting", "Engine waiting requests, last scrape", registry=r
        )
        self.engine_ready = Gauge("gateway_engine_ready", "1 if the engine is ready", registry=r)
        self.rejections = Counter(
            "gateway_rejections", "Shed requests by reason", ["reason"], registry=r
        )
        self.retries = Counter("gateway_retries", "Retries before the first token", registry=r)
        self.terminations = Counter(
            "gateway_stream_terminations", "How streams ended", ["type"], registry=r
        )
        self.usage_mismatch = Counter(
            "gateway_usage_mismatch", "Engine prompt_tokens != gateway count", registry=r
        )
        self.prompt_tokens = Counter("gateway_prompt_tokens", "Prompt tokens", registry=r)
        self.completion_tokens = Counter(
            "gateway_completion_tokens", "Completion tokens", registry=r
        )


class Stopwatch:
    def __init__(self) -> None:
        self.t0 = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.t0
