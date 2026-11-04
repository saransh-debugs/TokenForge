# TokenForge: one entry point. `make help` lists targets.
SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run
CHARTS := serving/helm/gateway serving/helm/mock-engine
K8S_VERSION := 1.36.0

.PHONY: help
help: ## List targets
	@grep -hE '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

# ------------------------------------------------------------------ environment
.PHONY: setup lock lock-check
setup: ## Create the dev env from uv.lock (CPU torch + Triton interpreter)
	$(UV) sync --frozen --extra kernels --extra serving --extra bench
	$(RUN) pre-commit install

lock: ## Re-resolve uv.lock and re-export the serving image requirements
	$(UV) lock
	$(UV) export --frozen --extra serving --no-dev --no-emit-project --no-hashes --no-header \
	  -o serving/docker/requirements.serving.txt

lock-check: ## Fail while infra/versions.lock still has values to resolve on the L4
	@! grep -n '"resolve-on-l4"' infra/versions.lock || { echo "unresolved pins above"; exit 1; }

# ------------------------------------------------------------------ quality
.PHONY: fmt lint typecheck helm-lint
fmt: ## Format Python
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

lint: helm-lint ## ruff, shellcheck, hadolint, helm lint, kubeconform
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) --with shellcheck-py shellcheck infra/*.sh
	@if command -v hadolint >/dev/null; then hadolint serving/docker/Dockerfile; \
	  else echo "hadolint not installed; skipped locally (CI runs it)"; fi

typecheck: ## mypy --strict (Triton kernel bodies relaxed)
	$(RUN) mypy

helm-lint: ## helm lint --strict + kubeconform for every chart and values file
	@for c in $(CHARTS); do helm lint --strict $$c; done
	@helm template t serving/helm/gateway -f serving/helm/values/mock-gateway.yaml > /tmp/tf-gw.yaml
	@helm template t serving/helm/mock-engine -f serving/helm/values/mock-engine.yaml \
	  --set autoscaling.enabled=true > /tmp/tf-engine.yaml
	kubeconform -strict -summary -kubernetes-version $(K8S_VERSION) -ignore-missing-schemas \
	  /tmp/tf-gw.yaml /tmp/tf-engine.yaml

# ------------------------------------------------------------------ tests
.PHONY: test test-cpu test-gpu calibrate-cpu calibrate-cuda
test: test-cpu ## Alias for test-cpu

test-cpu: ## CPU tier: unit, numerics (Triton interpreter), contract (mock engine)
	TRITON_INTERPRET=1 $(RUN) pytest -m "not gpu and not slow"

test-gpu: ## Full shape suite on the local GPU
	TOKENFORGE_SHAPE_SUITE=full $(RUN) pytest -m "not slow" tests/unit tests/numerics

calibrate-cpu: ## Re-derive the CPU tolerance constant (commit the diff with a reason)
	TRITON_INTERPRET=1 $(RUN) python -m tests.numerics.calibrate --device cpu --write

calibrate-cuda: ## Derive the CUDA tolerance constant against cuBLAS on this GPU
	TOKENFORGE_SHAPE_SUITE=full $(RUN) python -m tests.numerics.calibrate --device cuda --write

# ------------------------------------------------------------------ GPU tiers
.PHONY: gpu-dev gpu-ci gpu-report-check bench-micro
gpu-dev: ## GPU-dev tier (Kaggle T4): numerics, sanitizers, microbench, report JSON
	python -m tokenforge.ci.gpu_report run --tier dev

gpu-ci: budget-check ## GPU-full tier (rented L4): everything + FP8, report JSON
	python -m tokenforge.ci.gpu_report run --tier full

gpu-report-check: ## Verify a passing GPU report exists for HEAD's kernel/plugin code
	python -m tokenforge.ci.gpu_report check --tier dev

bench-micro: ## Kernel microbenchmarks with the 5% regression gate
	python -m tokenforge.bench.micro --suite l4 --check

# ------------------------------------------------------------------ stack
.PHONY: up down smoke
up: ## Mock stack in kind (gateway + mock engine), smoke-tested and timed
	infra/up.sh

down: ## Delete the kind cluster
	infra/down.sh

smoke: ## Smoke-test a running stack on localhost:8080
	python3 infra/smoke.py --url http://127.0.0.1:8080

# ------------------------------------------------------------------ budget
.PHONY: budget-check budget-report provision
budget-check: ## Exit non-zero once spend >= $$70
	python3 -m tokenforge.ledger check

budget-report: ## Spend so far vs the $$75 cap
	python3 -m tokenforge.ledger report

provision: budget-check ## Prepare a rented GPU box (TF_PROVIDER, TF_RATE required)
	infra/provision.sh

# ------------------------------------------------------------------ release
.PHONY: changelog
changelog: ## Regenerate CHANGELOG.md from conventional commits
	$(UV) tool run git-cliff --config cliff.toml -o CHANGELOG.md
