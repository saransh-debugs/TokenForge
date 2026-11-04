# ADR 0005: Single L4, plus a one-day 2x L4 autoscaling test

Status: accepted · Date: 2026-11-04 · Decision log: D5

## Context
Budget cap $75. Autoscaling needs at least two GPUs to be real.

## Decision
All development and benchmarking on one rented L4. Scale logic (KEDA on
`num_requests_waiting`) is proven on CPU in kind with the in-repo mock engine, then
validated once on 2x L4 for a day (~$8).

## Consequences
The mock engine must reproduce the engine signals that drive scaling and admission:
queueing behind `max_num_seqs`, KV pressure, preemption, aborts, and vLLM's metric names.

## Alternatives considered
A multi-GPU node for the whole project: 3-4x the cost.
