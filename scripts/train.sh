#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/wav2vec2_frozen_1h.yaml}"

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PWD/.cache/huggingface/transformers}"
export TOKENIZERS_PARALLELISM=false

python -B -m src.train --config "$CONFIG"
