# ADR 0008: GPU CI reports keyed by code-tree hash

Status: accepted · Date: 2026-11-04

## Context
GPU tiers cannot run on free GitHub runners. The plan requires CI to fail when a kernel or
plugin PR's GPU report does not match the PR head, but committing the report changes the
head SHA, so "report SHA == head SHA" can never hold for a committed report.

## Decision
`python -m tokenforge.ci.gpu_report run --tier dev|full` writes
`bench/results/ci/<tier>-<hash[:16]>.json`, where the hash covers the git trees of
`tokenforge/kernels`, `vllm_plugin`, `tests/numerics` and `tests/unit`. It refuses to run
with uncommitted changes in those paths. CPU CI recomputes the hash at the PR head and
requires a passing report with that exact hash whenever kernel or plugin code changed.

## Consequences
The report can be committed in the same PR without invalidating itself, and any later code
change to the vouched paths invalidates it.

## Alternatives considered
Matching the commit SHA (impossible for a committed report); a status check posted from
the GPU box (needs a token on rented hardware).
