# ADR 0006: Open-loop load generation

Status: accepted · Date: 2026-11-04 · Decision log: D6

## Context
Closed-loop clients wait for a response before sending the next request, so they slow down
exactly when the server saturates and hide queueing collapse.

## Decision
The load generator is open-loop: Poisson arrivals and a bursty gamma process (CV = 3).
Goodput counts only requests that meet both the TTFT and TPOT SLOs.

## Consequences
Reported goodput falls off a cliff at saturation instead of plateauing; that cliff is the
number the benchmark exists to find.

## Alternatives considered
Fixed-concurrency closed loop only: overstates goodput.
