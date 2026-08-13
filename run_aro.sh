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
  --vision-checkpoint /nethome/tadhopavkar/Concept-aware-vision-encoder/checkpoints/best_finetuned_siglip_concept_0.0.pt \
  --output-dir /nethome/tadhopavkar/Concept-aware-vision-encoder/results/aro_finetuning/llava-onevision-qwen2-0.5b-si-hf/best_finetuned_siglip_concept_0.0 \
  --generate-visualization \
  --plot-output-dir /nethome/tadhopavkar/Concept-aware-vision-encoder/visualizations/aro_finetuning/llava-onevision-qwen2-0.5b-si-hf/best_finetuned_siglip_concept_0.0 \
  --seeds 42 123 17

echo "=========================================="
echo "Finished Running aro evaluation: $(date)"
echo "=========================================="
