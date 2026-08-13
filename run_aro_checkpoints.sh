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

checkpoints=(
  "best_finetuned_siglip_concept_0.0.pt"
  "best_finetuned_siglip_concept_0.05.pt"
  "best_finetuned_siglip_concept_0.1.pt"
  "best_finetuned_siglip_concept_0.3.pt"
  "best_finetuned_siglip_concept_0.5.pt"
  "best_finetuned_siglip_concept_1.0.pt"
  "best_finetuned_siglip.pt"
)

model_id="/scratch/common_models/llava-onevision-qwen2-0.5b-si-hf"
model_name="llava-onevision-qwen2-0.5b-si-hf"

for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name="$(basename "$checkpoint" .pt)"

  echo "Evaluating: $checkpoint_name"

  python evaluate_aro_llava.py \
    --model-id "$model_id" \
    --vision-checkpoint "/nethome/tadhopavkar/Concept-aware-vision-encoder/checkpoints/$checkpoint" \
    --output-dir "/nethome/tadhopavkar/Concept-aware-vision-encoder/results/aro_finetuning/$model_name/$checkpoint_name" \
    --generate-visualization \
    --plot-output-dir "/nethome/tadhopavkar/Concept-aware-vision-encoder/visualizations/aro_finetuning/$model_name/$checkpoint_name" \
    --seeds 42 123 17
done

echo "=========================================="
echo "Finished Running aro evaluation: $(date)"
echo "=========================================="
