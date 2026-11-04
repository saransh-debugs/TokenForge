# ADR 0009: Python code lives under the `tokenforge` package

Status: accepted · Date: 2026-11-04

## Context
The plan's layout puts kernels, serving and bench code in top-level `kernels/`, `serving/`
and `bench/` directories. As import names, `kernels` collides with Hugging Face's `kernels`
package that vLLM's dependency tree can bring in, and `bench`/`serving` are generic.

## Decision
All importable Python lives under `tokenforge/` (`tokenforge.kernels`,
`tokenforge.serving.gateway`, `tokenforge.serving.mock_engine`, `tokenforge.bench`,
`tokenforge.ci`). Top-level `serving/`, `bench/`, `chaos/`, `infra/`, `docs/` hold the
non-Python assets from the plan's layout (Helm charts, results, manifests, scripts).
`vllm_plugin/` keeps its own pyproject as planned.

## Consequences
No import shadowing, a single wheel, and mypy/pytest resolve modules without path hacks.

## Alternatives considered
Top-level packages named after the directories (import collisions); per-directory
distributions (one more nesting level everywhere).
