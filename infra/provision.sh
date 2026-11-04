#!/usr/bin/env bash
# Prepare a fresh rented L4 (or T4) box: budget gate, ledger session, GPU Python env pinned
# by versions.lock, sanity checks, and the idle watchdog. Refuses once spend >= $70.
#
#   TF_PROVIDER=runpod TF_RATE=0.44 infra/provision.sh [--serving]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROVIDER="${TF_PROVIDER:?set TF_PROVIDER (e.g. runpod, jarvislabs)}"
RATE="${TF_RATE:?set TF_RATE (USD per hour)}"
SERVING=0
if [[ "${1:-}" == "--serving" ]]; then SERVING=1; fi

lock() { python3 -c "import tomllib,sys;d=tomllib.load(open('infra/versions.lock','rb'));print(d[sys.argv[1]][sys.argv[2]])" "$1" "$2"; }

command -v nvidia-smi >/dev/null || { echo "error: no NVIDIA driver on this box" >&2; exit 1; }
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
HOST_RAM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
if (( SERVING == 1 && HOST_RAM_GB < 60 )); then
  echo "error: ${HOST_RAM_GB} GB host RAM; TensorRT-LLM 7B engine builds need >= 64 GB" >&2
  exit 1
fi

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Budget gate before anything that spends time on the meter.
python3 -m tokenforge.ledger check
python3 -m tokenforge.ledger start --provider "$PROVIDER" --gpu "$GPU_NAME" --rate "$RATE"

# GPU env: vLLM brings its pinned torch/triton; TokenForge installs on top of it.
uv venv --python "$(lock python version)" .venv-gpu
# shellcheck disable=SC1091
source .venv-gpu/bin/activate
uv pip install "vllm==$(lock engine_vllm vllm)"
uv pip install -e ".[serving,bench]" pytest pytest-asyncio pytest-timeout hypothesis
python - <<'PY'
import torch, triton, vllm
assert torch.cuda.is_available(), "CUDA not visible to torch"
print("torch", torch.__version__, "triton", triton.__version__, "vllm", vllm.__version__,
      "gpu", torch.cuda.get_device_name(), "sm", torch.cuda.get_device_capability())
PY

if (( SERVING == 1 )); then
  curl -sfL https://get.k3s.io \
    | INSTALL_K3S_VERSION="$(lock cluster k3s)" sh -s - --write-kubeconfig-mode 644
fi

nohup infra/watchdog.sh >"$ROOT/infra/watchdog.log" 2>&1 &
echo "provisioned: $GPU_NAME, watchdog pid $!, ledger session open at \$${RATE}/h"
