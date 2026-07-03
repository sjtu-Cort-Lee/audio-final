#!/usr/bin/env bash
set -euo pipefail

echo "== machine =="
hostname
pwd

echo "== gpu =="
nvidia-smi || true

echo "== python =="
which python || true
python --version || true
which python3 || true
python3 --version || true

echo "== conda =="
which conda || true
conda --version || true

echo "== pip =="
which pip || true
pip --version || true

echo "== disk =="
df -h .
