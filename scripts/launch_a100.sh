#!/usr/bin/env bash
# Launch the full Mamba-2-Lite pretraining run on A100 80GB.
# Usage: bash scripts/launch_a100.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. Data — prepare the Chinchilla-optimal 8.0B-token dataset
python data/prepare_data.py --stage pretrain --output-dir data/pretrain_chinchilla --max-tokens 8000000000 --shard-size-tokens 50000000

# 2. Microbench — measure peak VRAM
python scripts/microbench_a100.py

# 3. Step-time benchmark — validate MFU
python scripts/step_time_a100.py --steps 20 --warmup 5

# 4. Full pretraining run (~12-15 h on A100 80GB)
python training/pretrain.py --config configs/pretrain_a100_400m.yaml