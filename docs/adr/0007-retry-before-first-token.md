# ADR 0007: Gateway retries only before the first token

Status: accepted · Date: 2026-11-04 · Decision log: D7

## Context
Once a token has been streamed to the client, a retry would duplicate output.

## Decision
The gateway reads the engine stream up to the first token before sending response headers.
Failures in that window are retried (default once) and otherwise mapped to an HTTP status
(retriable 503 with Retry-After; engine 4xx passed through). After the first token, an
engine failure ends the stream with a terminal SSE `error` event and no `[DONE]`.

## Consequences
Zero silent truncations: every stream ends in `[DONE]` or an explicit error event, which
the load generator verifies (F13). TTFT seen by the client includes retries.

## Alternatives considered
Transparent retries mid-stream: corrupts streams.
