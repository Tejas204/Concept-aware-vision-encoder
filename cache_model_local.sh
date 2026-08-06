#!/bin/bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mlu

cd "$HOME/Concept-aware-vision-encoder"

# Set HF_TOKEN before running if the model requires authentication:
# export HF_TOKEN="your_token"

echo "Caching model on: $(hostname)"
echo "Started: $(date)"

python -u utils/cache_model_local.py

echo "Finished: $(date)"