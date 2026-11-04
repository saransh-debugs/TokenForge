"""Request schemas: the OpenAI subset the gateway accepts. Unknown fields are a 400."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

StopItem = Annotated[str, StringConstraints(min_length=1, max_length=128)]


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = False


class _Common(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=256)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: StopItem | list[StopItem] | None = Field(default=None)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    user: str | None = Field(default=None, max_length=256)

    def stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        stops = [self.stop] if isinstance(self.stop, str) else list(self.stop)
        if len(stops) > 4:
            raise ValueError("at most 4 stop sequences")
        return stops

    def requested_max_tokens(self) -> int | None:
        return self.max_tokens


class CompletionRequest(_Common):
    prompt: str = Field(min_length=1)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(_Common):
    messages: list[ChatMessage] = Field(min_length=1, max_length=512)
    max_completion_tokens: int | None = Field(default=None, ge=1)

    def requested_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens
