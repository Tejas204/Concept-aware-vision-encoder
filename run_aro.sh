#!/bin/bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mlu

cd "$HOME/Concept-aware-vision-encoder/experiments"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
PY

python evaluate_aro_llava.py \
  --model-id /scratch/common_models/llava-onevision-qwen2-0.5b-si-hf \
  --output-dir /nethome/tadhopavkar/Concept-aware-vision-encoder/results/llava-onevision-qwen2-0.5b-si-hf/aro_baseline \
  --seeds 42 123 17

echo "=========================================="
echo "Finished Running aro evaluation: $(date)"
echo "=========================================="