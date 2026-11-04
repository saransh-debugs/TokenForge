"""Tokenizers shared by the gateway (validation, usage checks) and the mock engine.

The gateway must count tokens exactly like the engine: a tokenizer revision mismatch shows
up as a ``usage`` disagreement (F19). Chat prompts are rendered with ChatML, Qwen2.5's
template, including its default system prompt.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class Tokenizer(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


def render_chatml(messages: Sequence[dict[str, str]], default_system: str | None) -> str:
    """ChatML rendering with ``add_generation_prompt=True``."""
    parts: list[str] = []
    if default_system is not None and (not messages or messages[0]["role"] != "system"):
        parts.append(f"<|im_start|>system\n{default_system}<|im_end|>\n")
    parts.extend(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


class SimpleTokenizer:
    """Deterministic word/punctuation tokenizer for the mock engine and CPU tests.

    Token counts depend only on the text and ``revision``; any revision other than ``v1``
    also splits digit runs, so two revisions disagree on counts the way real ones do (F19).
    Ids are assigned per instance on first sight and are only meaningful to that instance.
    """

    _PATTERN = re.compile(r"<\|[a-z_]+\|>|\w+|[^\w\s]|\s+")

    def __init__(self, revision: str = "v1") -> None:
        self.name = f"simple-{revision}"
        self._split_digits = revision != "v1"  # a different revision tokenizes differently
        self._vocab: dict[str, int] = {}
        self._inv: list[str] = []

    def _pieces(self, text: str) -> list[str]:
        pieces = [p for p in self._PATTERN.findall(text) if not p.isspace()]
        if self._split_digits:
            pieces = [c for p in pieces for c in (list(p) if p.isdigit() else [p])]
        return pieces

    def encode(self, text: str) -> list[int]:
        ids = []
        for p in self._pieces(text):
            idx = self._vocab.get(p)
            if idx is None:
                idx = len(self._inv)
                self._vocab[p] = idx
                self._inv.append(p)
            ids.append(idx)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._inv[i] for i in ids)


class HFTokenizer:
    """A Hugging Face tokenizer pinned to a local path or ``repo@revision``."""

    def __init__(self, spec: str) -> None:
        from transformers import AutoTokenizer  # deferred: heavy, only real-engine gateways

        repo, _, revision = spec.partition("@")
        self._tok = AutoTokenizer.from_pretrained(repo, revision=revision or None)
        self.name = f"hf:{spec}"

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return str(self._tok.decode(list(ids)))

    def render_chat(self, messages: Sequence[dict[str, str]]) -> str:
        return str(
            self._tok.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        )


def load_tokenizer(spec: str) -> Tokenizer:
    """``simple`` / ``simple:<revision>`` / ``hf:<repo-or-path>[@revision]``."""
    if spec == "simple" or spec.startswith("simple:"):
        return SimpleTokenizer(spec.partition(":")[2] or "v1")
    if spec.startswith("hf:"):
        return HFTokenizer(spec[3:])
    raise ValueError(f"unknown tokenizer spec {spec!r}")


def render_chat(tok: Tokenizer, messages: Sequence[dict[str, str]]) -> str:
    if isinstance(tok, HFTokenizer):
        return tok.render_chat(messages)
    return render_chatml(messages, QWEN_DEFAULT_SYSTEM)
