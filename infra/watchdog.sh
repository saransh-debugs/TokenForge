#!/usr/bin/env bash
# Power the rented box off when GPU utilization stays under a threshold, or when total
# spend reaches the provisioning stop. Closes the ledger session first so the cost is
# recorded.
#
#   TF_IDLE_UTIL=5 TF_IDLE_MINUTES=20 TF_POWEROFF="shutdown -h now" infra/watchdog.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IDLE_UTIL="${TF_IDLE_UTIL:-5}"
IDLE_MINUTES="${TF_IDLE_MINUTES:-20}"
POLL_SECONDS="${TF_POLL_SECONDS:-60}"
PY="${TF_PYTHON:-python3}"
read -r -a POWEROFF <<< "${TF_POWEROFF:-sudo shutdown -h now}"

idle_samples=0
needed=$(( IDLE_MINUTES * 60 / POLL_SECONDS ))

poweroff() {
  echo "watchdog: $1; closing ledger session and powering off" >&2
  (cd "$ROOT" && "$PY" -m tokenforge.ledger stop --notes "watchdog: $1") || true
  sync
  "${POWEROFF[@]}"
  exit 0
}

while true; do
  # Highest utilization across GPUs: any busy GPU keeps the box alive.
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
    | sort -n | tail -1 | tr -d ' ')
  if [[ -z "$util" ]]; then
    echo "watchdog: nvidia-smi returned nothing" >&2
  elif (( util < IDLE_UTIL )); then
    idle_samples=$(( idle_samples + 1 ))
  else
    idle_samples=0
  fi
  if (( idle_samples >= needed )); then
    poweroff "GPU < ${IDLE_UTIL}% for ${IDLE_MINUTES} min"
  fi
  if ! (cd "$ROOT" && "$PY" -m tokenforge.ledger check >/dev/null); then
    poweroff "spend reached the provisioning stop"
  fi
  sleep "$POLL_SECONDS"
done
