# Concept-Aware Vision Encoder

This project studies whether concept supervision improves the spatial and compositional reasoning of a LLaVA-OneVision SigLIP vision encoder. It includes data preparation, dataset analysis, linear probing, full/LoRA vision-tower fine-tuning, and evaluation on the spatial subset of ARO Visual Relation.

## Setup

The supplied environment targets Python 3.11, CUDA 12.6, and the `mlu` Conda environment.

```bash
bash setup.sh
conda activate mlu
```

Alternatively, install the dependencies directly:

```bash
pip install -r requirements.txt
```

Before running experiments, update the dataset, model, checkpoint, result, and visualization paths in `config/`, `scripts/`, and the relevant shell launchers. Several paths are currently specific to the original `/nethome` and `/scratch` environment.

## Main workflows

Run commands from the repository root unless noted otherwise.

```bash
# Build processed metadata and concept/object/predicate vocabularies
python scripts/execute_datapipe.py

# Generate dataset-frequency visualizations
python scripts/execute_visualizations.py

# Train concept, object, and predicate linear probes
python scripts/execute_probe.py

# Fine-tune the vision tower and optional bottleneck/LoRA adapters
python scripts/execute_finetuning.py
```

The corresponding cluster launchers are `run_probe.sh`, `run_finetuning.sh`, and the `submit_*.sub` files.

## ARO evaluation

Baseline evaluation:

```bash
python experiments/evaluate_aro_llava.py \
  --model-id <model-path-or-id> \
  --output-dir results/aro_baseline \
  --seeds 42 123 17
```

Use `--vision-checkpoint <checkpoint.pt>` for full fine-tuning or `--vision-adapter <adapter-directory>` for LoRA. Each seed directory receives `predictions.jsonl` and `summary.json`.

Qualitative comparisons can be generated without rerunning model evaluation:

```bash
# Baseline errors corrected by fine-tuning
python experiments/evaluate_aro_llava.py --plot-examples

# Baseline successes made wrong by fine-tuning
python experiments/evaluate_aro_llava.py --plot-wrong-examples
```

These commands use the fixed seed-17 result paths defined in `experiments/evaluate_aro_llava.py` and save figures under `results/qualitative_results`.

## Repository layout

- `config/`: fine-tuning and probing configuration.
- `pipeline/`: metadata processing and PyTorch datasets.
- `models/`: SigLIP vision-tower fine-tuning.
- `experiments/`: linear probing and ARO evaluation.
- `scripts/`: Python workflow entry points.
- `utils/`: dataset visualization and model caching utilities.
- `run_*.sh`, `submit_*.sub`: cluster execution and submission scripts.
