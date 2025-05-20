# !bin/bash
export CUDA_VISIBLE_DEVICES=7
uv run scripts/compute_norm_stats.py --config-name pi0_mshab
export CUDA_VISIBLE_DEVICES=4,5,6,7
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_mshab --exp-name=openpi_0 --overwrite