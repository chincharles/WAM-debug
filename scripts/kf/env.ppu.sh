#!/usr/bin/env bash
# Source after env.local.sh and activating the environment created by install_ppu.py.
export SIMWAM_RUNTIME=ppu
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/configs/server/compat:$PWD/src:$PWD/navsim"
export NAVSIM_DEVKIT_ROOT="$PWD/navsim"
export SIMWAM_KF_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Preserve the image's LD_LIBRARY_PATH and SDK variables; never source the VLA venv.
