#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Use the dedicated PPU environment when it has been installed; never activate VLA.
if [[ -f .venv-kf-ppu/bin/activate ]]; then
  source .venv-kf-ppu/bin/activate
  export SIMWAM_RUNTIME=ppu
fi
if [[ -f scripts/kf/env.local.sh ]]; then
  source scripts/kf/env.local.sh
fi
if [[ "${SIMWAM_RUNTIME:-}" == ppu ]] || command -v ppu-smi >/dev/null 2>&1; then
  source scripts/kf/env.ppu.sh
fi
exec python scripts/kf/debug_server.py "$@"
