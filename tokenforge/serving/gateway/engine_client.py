"""Streaming client for OpenAI-compatible engines (vLLM, Triton's OpenAI frontend, mock).

The gateway always streams from the engine, even for non-streaming clients, so there is one
code path for retries, timeouts and aborts. Closing the upstream response is the abort
signal: vLLM and Triton cancel a request whose connection goes away.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from tokenforge.serving import sse


@dataclass(frozen=True)
class Delta:
    text: str
    finish_reason: str | None = None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


Event = Delta | Usage


class EngineError(Exception):
    def __init__(self, message: str, *, status: int | None, retriable: bool) -> None:
        super().__init__(message)
        self.status = status
        self.retriable = retriable


def _retriable_status(status: int) -> bool:
    return status in (408, 429) or status >= 500


class OpenAIEngine:
    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client

    async def stream(
        self, chat: bool, payload: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[Event]:
        path = "/v1/chat/completions" if chat else "/v1/completions"
        try:
            async with self.client.stream(
                "POST", self.base_url + path, json=payload, headers=headers
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread())[:2048].decode("utf-8", "replace")
                    raise EngineError(
                        f"engine returned {resp.status_code}: {body}",
                        status=resp.status_code,
                        retriable=_retriable_status(resp.status_code),
                    )
                async for data in sse.iter_data(resp.aiter_lines()):
                    if data == sse.DONE:
                        return
                    obj = json.loads(data)
                    if "error" in obj:
                        raise EngineError(str(obj["error"]), status=None, retriable=False)
                    for choice in obj.get("choices") or ():
                        text = choice.get("text")
                        if text is None:
                            text = (choice.get("delta") or {}).get("content") or ""
                        yield Delta(text, choice.get("finish_reason"))
                    usage = obj.get("usage")
                    if usage:
                        yield Usage(int(usage["prompt_tokens"]), int(usage["completion_tokens"]))
                raise EngineError("engine stream ended without [DONE]", status=None, retriable=True)
        except httpx.TimeoutException as exc:
            raise EngineError(
                f"engine timeout: {type(exc).__name__}", status=None, retriable=True
            ) from exc
        except httpx.TransportError as exc:
            raise EngineError(
                f"engine connection error: {type(exc).__name__}", status=None, retriable=True
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:  # JSONDecodeError is a ValueError
            raise EngineError(
                f"malformed engine stream: {exc}", status=None, retriable=False
            ) from exc
