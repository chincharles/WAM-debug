#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:$PWD/navsim:${PYTHONPATH:-}"
case "${NPROC_PER_NODE:-1}" in 1|2|4|8) ;; *) echo 'NPROC_PER_NODE must be 1/2/4/8' >&2; exit 2;; esac
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == --dry-run ]]; then exec python scripts/kf/train.py "$@"; fi
done
if [[ "${NPROC_PER_NODE:-1}" == 1 ]]; then
  exec python scripts/kf/train.py "$@"
else
  exec python -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" scripts/kf/train.py "$@"
fi
