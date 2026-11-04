#!/usr/bin/env bash
set -euo pipefail
kind delete cluster --name "${TF_KIND_CLUSTER:-tokenforge}"
