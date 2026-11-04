"""Gateway configuration, read once from the environment (``TF_GW_*``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(f"TF_GW_{name}", default)


@dataclass(frozen=True)
class GatewaySettings:
    engine_url: str = "http://127.0.0.1:8000"
    engine_metrics_urls: tuple[str, ...] = ()
    engine_health_path: str = "/health"
    engine_waiting_metric: str = "vllm:num_requests_waiting"
    served_model: str = "qwen2.5-7b-instruct"
    engine_model: str = "qwen2.5-7b-instruct"
    tokenizer: str = "simple"
    max_context: int = 8192
    max_tokens_limit: int = 4096
    default_max_tokens: int = 256
    max_body_bytes: int = 4 * 1024 * 1024
    max_inflight: int = 256
    max_queue: int = 512
    waiting_threshold: float = 64.0
    queue_timeout_s: float = 30.0
    retry_after_s: int = 2
    retries_before_first_token: int = 1
    retry_backoff_s: float = 0.05
    connect_timeout_s: float = 2.0
    read_timeout_s: float = 30.0
    write_timeout_s: float = 10.0
    poll_interval_s: float = 0.5
    log_prompts: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def metrics_urls(self) -> tuple[str, ...]:
        return self.engine_metrics_urls or (self.engine_url.rstrip("/") + "/metrics",)

    @classmethod
    def from_env(cls) -> GatewaySettings:
        d = cls()
        urls = tuple(u for u in _env("ENGINE_METRICS_URLS", "").split(",") if u)
        return cls(
            engine_url=_env("ENGINE_URL", d.engine_url),
            engine_metrics_urls=urls,
            engine_health_path=_env("ENGINE_HEALTH_PATH", d.engine_health_path),
            engine_waiting_metric=_env("ENGINE_WAITING_METRIC", d.engine_waiting_metric),
            served_model=_env("SERVED_MODEL", d.served_model),
            engine_model=_env("ENGINE_MODEL", d.engine_model),
            tokenizer=_env("TOKENIZER", d.tokenizer),
            max_context=int(_env("MAX_CONTEXT", str(d.max_context))),
            max_tokens_limit=int(_env("MAX_TOKENS_LIMIT", str(d.max_tokens_limit))),
            default_max_tokens=int(_env("DEFAULT_MAX_TOKENS", str(d.default_max_tokens))),
            max_body_bytes=int(_env("MAX_BODY_BYTES", str(d.max_body_bytes))),
            max_inflight=int(_env("MAX_INFLIGHT", str(d.max_inflight))),
            max_queue=int(_env("MAX_QUEUE", str(d.max_queue))),
            waiting_threshold=float(_env("WAITING_THRESHOLD", str(d.waiting_threshold))),
            queue_timeout_s=float(_env("QUEUE_TIMEOUT_S", str(d.queue_timeout_s))),
            retry_after_s=int(_env("RETRY_AFTER_S", str(d.retry_after_s))),
            retries_before_first_token=int(_env("RETRIES", str(d.retries_before_first_token))),
            retry_backoff_s=float(_env("RETRY_BACKOFF_S", str(d.retry_backoff_s))),
            connect_timeout_s=float(_env("CONNECT_TIMEOUT_S", str(d.connect_timeout_s))),
            read_timeout_s=float(_env("READ_TIMEOUT_S", str(d.read_timeout_s))),
            write_timeout_s=float(_env("WRITE_TIMEOUT_S", str(d.write_timeout_s))),
            poll_interval_s=float(_env("POLL_INTERVAL_S", str(d.poll_interval_s))),
            log_prompts=_env("LOG_PROMPTS", "0") == "1",
        )
