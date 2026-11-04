# ADR 0004: torch.library custom ops with fake implementations

Status: accepted · Date: 2026-11-04 · Decision log: D4

## Context
vLLM runs models under torch.compile and CUDA graphs. Raw pybind calls cause graph breaks
and can hide host syncs.

## Decision
Every kernel is a `torch.library.custom_op` with a `register_fake` meta implementation.
Implementations never allocate inside kernels, never sync with the host, and never read
tensor values on the host. Dispatch reads only shapes.

## Consequences
`opcheck` (schema, fake tensor, AOT dispatch) and a `fullgraph=True, dynamic=True` compile
run in the CPU tier. Split-K reduces in a fixed order, so CUDA-graph replay and eager are
bit-identical (F5).

## Alternatives considered
Raw pybind / `torch.ops` without fakes: graph breaks under torch.compile.
