#!/bin/bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mlu

cd "$HOME/Concept-aware-vision-encoder/scripts"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
PY

python -u execute_finetuning.py

echo "=========================================="
echo "Finished finetuning: $(date)"
echo "=========================================="