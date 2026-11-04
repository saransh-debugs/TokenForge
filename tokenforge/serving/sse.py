"""Server-sent events framing for OpenAI-style streams."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

DONE = "[DONE]"


def event(payload: dict[str, Any] | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n".encode()


async def iter_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield the ``data`` field of each SSE event (multi-line data joined with newlines)."""
    buf: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if buf:
                yield "\n".join(buf)
                buf = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            buf.append(value[1:] if value.startswith(" ") else value)
    if buf:
        yield "\n".join(buf)
