# ADR 0003: Out-of-tree vLLM plugin, not a fork

Status: accepted · Date: 2026-11-04 · Decision log: D3

## Context
vLLM internals change monthly. A fork turns every upgrade into a rebase.

## Decision
A separate package, `tokenforge-vllm`, registered through the `vllm.general_plugins` entry
point. Its surface is `QuantizationConfig` + `LinearMethodBase` only, guarded by a supported
version range that fails loudly at startup (F6).

## Consequences
Upgrades are a version bump plus a GPU-full run. Reviewers read the plugin alone.

## Alternatives considered
Forking vLLM: rebase cost, and the diff is hard to review in isolation.
