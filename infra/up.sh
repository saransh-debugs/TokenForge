#!/usr/bin/env bash
# `make up` for the mock stack: kind cluster + gateway + mock engine, smoke-tested and timed.
# The duration is appended to bench/results/kind/make_up.jsonl (regression metric).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLUSTER="${TF_KIND_CLUSTER:-tokenforge}"
REPO="tokenforge/serving"
start=$(date +%s)

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "error: '$1' not found on PATH" >&2; exit 1; }
}
for tool in docker kind kubectl helm python3; do need "$tool"; done

if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER" --config serving/kind/cluster.yaml --wait 180s
fi
kubectl config use-context "kind-$CLUSTER" >/dev/null

# TF_BUILD_CA: CA bundle for builds behind a TLS-intercepting proxy (optional).
docker build -q ${TF_BUILD_CA:+--secret "id=ca,src=$TF_BUILD_CA"} \
  -f serving/docker/Dockerfile -t "$REPO:build" . >/dev/null
# Tag by content so `helm upgrade` rolls pods exactly when the image changed.
TAG="dev-$(docker image inspect -f '{{.Id}}' "$REPO:build" | cut -d: -f2 | cut -c1-12)"
docker tag "$REPO:build" "$REPO:$TAG"
kind load docker-image "$REPO:$TAG" --name "$CLUSTER" >/dev/null

helm upgrade --install engine serving/helm/mock-engine \
  -f serving/helm/values/mock-engine.yaml --set image.tag="$TAG" --wait --timeout 5m
helm upgrade --install gateway serving/helm/gateway \
  -f serving/helm/values/mock-gateway.yaml --set image.tag="$TAG" --wait --timeout 5m

python3 infra/smoke.py --url "http://127.0.0.1:8080" --timeout 120

elapsed=$(( $(date +%s) - start ))
mkdir -p bench/results/kind
printf '{"ts":"%s","git_sha":"%s","target":"mock","image_tag":"%s","seconds":%d}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD 2>/dev/null || echo unknown)" \
  "$TAG" "$elapsed" >> bench/results/kind/make_up.jsonl
echo "make up: mock stack ready in ${elapsed}s (gateway at http://127.0.0.1:8080)"
