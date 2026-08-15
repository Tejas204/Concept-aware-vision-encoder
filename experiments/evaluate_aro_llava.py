#!/usr/bin/env python3
"""Evaluate LLaVA-OneVision on spatial ARO relations.

Examples
--------
# Quick smoke test
python evaluate_aro_llava.py \
  --model-id llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  --max-samples 100

# Baseline experiment
python evaluate_aro_llava.py \
  --model-id llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --output-dir results/aro_baseline

# Same evaluation with a locally saved PEFT adapter for the vision encoder
python evaluate_aro_llava.py \
  --model-id llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --vision-adapter checkpoints/vision_adapter \
  --output-dir results/aro_finetuned

---------------------------------------------------------
# Original model
python experiments/evaluate_aro_llava.py --model-id MODEL

# Standard fine-tuned vision encoder
python experiments/evaluate_aro_llava.py \
  --model-id lava-hf/llava-onevision-qwen2-7b-ov-hf \
  --vision-checkpoint checkpoints/best_finetuned_siglip.pt

# LoRA vision adapter
python evaluate_aro_llava.py \
  --model-id llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --vision-adapter checkpoints/best_finetuned_siglip_vision_adapter

# QLoRA adapter with quantized base
python evaluate_aro_llava.py \
  llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --vision-adapter checkpoints/best_finetuned_siglip_vision_adapter \
  --use-qlora

python evaluate_aro_llava.py \
  --model-id /scratch/common_models/llava-onevision-qwen2-7b-si-hf \
  --output-dir /nethome/tadhopavkar/Concept-aware-vision-encoder/results/aro_baseline/llava-onevision-qwen2-7b-si-hf \
  --generate-visualization \
  --plot-output-dir /nethome/tadhopavkar/Concept-aware-vision-encoder/visualizations/aro_baseline/llava-onevision-qwen2-7b-si-hf \
  --seeds 42 123 17

The output directory receives predictions.jsonl and summary.json.  The latter
contains micro accuracy, macro (per-relation) accuracy, and relation counts.
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration


# This is our explicit spatial subset. ARO does not publish it as a separate split.
# Run with --list-relations to inspect the predicate names in the dataset.
DEFAULT_SPATIAL_RELATIONS = {
    "above",
    "behind",
    "in",
    "in front of",
    "next to",
    "on",
    "to the left of",
    "to the right of",
    "under",
}

BASELINE_PREDICTIONS_DIR = Path(
    "/nethome/tadhopavkar/Concept-aware-vision-encoder/results/aro_baseline/"
    "llava-onevision-qwen2-0.5b-si-hf/seed_17"
)
FINETUNING_PREDICTIONS_DIR = Path(
    "/nethome/tadhopavkar/Concept-aware-vision-encoder/results/aro_finetuning/"
    "llava-onevision-qwen2-0.5b-si-hf/"
    "best_finetuned_siglip_concept_0.0/seed_17"
)
LORA_PREDICTIONS_DIR = Path(
    "/nethome/tadhopavkar/Concept-aware-vision-encoder/results/aro_finetuning_lora/"
    "llava-onevision-qwen2-0.5b-si-hf/"
    "best_finetuned_siglip_lora_concept_0.0_vision_adapter/seed_17"
)
QUALITATIVE_RESULTS_DIR = Path(
    "/nethome/tadhopavkar/Concept-aware-vision-encoder/results/qualitative_results"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-id", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    p.add_argument(
        "--vision-adapter",
        type=Path,
        default=None,
        help=(
            "Optional local PEFT adapter directory saved from a wrapped SigLIP "
            "vision encoder. Omit this flag for the untouched LLaVA baseline."
        ),
    )
    p.add_argument("--vision-checkpoint", type=Path, default=None)
    p.add_argument("--use-qlora", action="store_true")
    p.add_argument("--dataset-id", default="mteb/ARO-Visual-Relation")
    p.add_argument("--split", default="test")
    p.add_argument("--list-relations", action="store_true",
                   help="Print relation counts and exit before loading LLaVA.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("results/aro_llava"))
    p.add_argument(
        "--plot-examples",
        action="store_true",
        help=(
            "Only generate baseline-vs-finetuning qualitative comparisons from "
            "the fixed seed-17 prediction files; do not load or evaluate a model."
        ),
    )
    p.add_argument(
        "--plot-wrong-examples",
        action="store_true",
        help=(
            "Only plot examples that the baseline got right but finetuning got "
            "wrong; do not load or evaluate a model."
        ),
    )
    p.add_argument("--generate-visualization", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--plot-output-dir", type=Path, default=Path("visualizations/aro_examples"))
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def normalize_relation(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def get_field(example: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        if key in example:
            return example[key]
    raise KeyError(f"None of {candidates} found. Available columns: {sorted(example)}")


def relation_of(example: dict[str, Any]) -> str:
    return normalize_relation(get_field(example, ("relation_name", "relation", "predicate")))


def calculate_spatial_frequency(ds):
    counts: dict[str, int] = {relation: 0 for relation in DEFAULT_SPATIAL_RELATIONS}
    for row in ds:
        relation = relation_of(row)
        if relation in counts:
            counts[relation] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def sample_dataset_rows(ds, min_freq):
    # Based on min_freq, for each spatial relation, randomly sample min_freq rows
    # Create the randomly sampled dataset and return it.
    sampled_datasets = []
    for relation in sorted(DEFAULT_SPATIAL_RELATIONS):
        relation_ds = ds.filter(lambda row, relation=relation: relation_of(row) == relation)
        if len(relation_ds) < min_freq:
            raise ValueError(
                f"Requested {min_freq} rows for relation {relation!r}, but only {len(relation_ds)} are available."
            )
        sampled_datasets.append(relation_ds.select(random.sample(range(len(relation_ds)), min_freq)))

    return concatenate_datasets(sampled_datasets)

def prepare_dataset(args: argparse.Namespace) -> Dataset:
    ds = load_dataset(args.dataset_id, split=args.split)
    if args.list_relations:
        counts: dict[str, int] = defaultdict(int)
        for row in ds:
            counts[relation_of(row)] += 1
        for relation, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"{relation:35s} {count:7d}")
        raise SystemExit(0)
    ds = ds.filter(lambda row: relation_of(row) in DEFAULT_SPATIAL_RELATIONS)

    # Print the total number of samples with the above given spatial relation
    print(f"Total samples in filtered benchmark is: {len(ds)}")

    # Find the frequency of each spatial relation
    frequencies = calculate_spatial_frequency(ds=ds)

    # Choose the minimum of the frequencies (min_freq), and sample randomly min_freq times for each relation
    min_freq = min(frequencies.values())
    ds = sample_dataset_rows(ds=ds, min_freq=min_freq)

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    if len(ds) == 0:
        raise ValueError("No examples remain after filtering.")
    return ds


def collate(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Keep PIL images and variable metadata intact; processor batching happens later.
    return examples


def make_prompt(processor: AutoProcessor, caption_a: str, caption_b: str) -> str:
    question = (
        "Which caption correctly describes the image?\n"
        f"A. {caption_a}\n"
        f"B. {caption_b}\n"
        "Answer with only A or B."
    )
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def answer_token_id(processor: AutoProcessor, answer: str) -> int:
    # The chat prompt ends immediately after the assistant marker.  For Qwen2,
    # A/B are one token each. Validate this because multi-token labels require
    # sequence-level scoring, not the one-forward-pass method used here.
    ids = processor.tokenizer.encode(answer, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Answer {answer!r} tokenizes as {ids}; expected exactly one token.")
    return ids[0]


def image_of(row: dict[str, Any]) -> Image.Image:
    image = get_field(row, ("image", "images"))
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected a PIL image, received {type(image).__name__}")
    return image.convert("RGB")


def captions_of(row: dict[str, Any]) -> tuple[str, str]:
    true_caption = str(get_field(row, ("true_caption", "caption", "positive_caption")))
    false_caption = str(get_field(row, ("false_caption", "negative_caption")))
    return true_caption, false_caption


def load_prediction_records(predictions_dir: Path) -> list[dict[str, Any]]:
    predictions_path = predictions_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"{predictions_path} was not found.")
    return [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _plot_prediction_transitions(
    ds: Dataset,
    output_dir: Path,
    *,
    baseline_correct: bool,
    candidate_correct: bool,
    filename_prefix: str,
    title: str,
) -> None:
    """Plot matched samples whose correctness changes between two runs."""
    baseline_records = load_prediction_records(BASELINE_PREDICTIONS_DIR)
    comparison_runs = {
        "finetuning": load_prediction_records(FINETUNING_PREDICTIONS_DIR),
        "finetuning_lora": load_prediction_records(LORA_PREDICTIONS_DIR),
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_by_index = {record["index"]: record for record in baseline_records}
    for run_name, candidate_records in comparison_runs.items():
        candidate_by_index = {record["index"]: record for record in candidate_records}
        run_output_dir = output_dir / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        for relation in sorted(DEFAULT_SPATIAL_RELATIONS):
            selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for index, baseline in baseline_by_index.items():
                candidate = candidate_by_index.get(index)
                if candidate is None or normalize_relation(baseline["relation"]) != relation:
                    continue
                identity_fields = ("relation", "true_caption", "false_caption", "gold")
                if any(baseline[field] != candidate[field] for field in identity_fields):
                    raise ValueError(
                        f"Prediction records do not describe the same sample at index {index}."
                    )
                if (
                    baseline["correct"] == baseline_correct
                    and candidate["correct"] == candidate_correct
                ):
                    selected.append((baseline, candidate))
                if len(selected) == 2:
                    break

            if len(selected) < 2:
                raise ValueError(
                    f"Expected 2 matching transition samples for {run_name} and "
                    f"{relation!r}, but found {len(selected)}."
                )

            figure, axes = plt.subplots(
                len(selected), 2, figsize=(14, 6 * len(selected)), squeeze=False
            )
            for row_number, (baseline, candidate) in enumerate(selected):
                labeled_records = (
                    ("Baseline", baseline),
                    (run_name.replace("_", " ").title(), candidate),
                )
                for column, (label, record) in enumerate(labeled_records):
                    axis = axes[row_number, column]
                    dataset_row = ds[record["index"]]
                    dataset_captions = captions_of(dataset_row)
                    if (
                        relation_of(dataset_row) != relation
                        or dataset_captions
                        != (record["true_caption"], record["false_caption"])
                    ):
                        raise ValueError(
                            "The seed-17 dataset reconstruction does not match "
                            f"prediction index {record['index']}."
                        )
                    caption_a = (
                        record["true_caption"]
                        if record["gold"] == "A"
                        else record["false_caption"]
                    )
                    caption_b = (
                        record["false_caption"]
                        if record["gold"] == "A"
                        else record["true_caption"]
                    )
                    marker_a = "★ SELECTED " if record["prediction"] == "A" else ""
                    marker_b = "★ SELECTED " if record["prediction"] == "B" else ""
                    caption_text = (
                        f"Gold caption: {textwrap.fill(record['true_caption'], 58)}\n\n"
                        f"{marker_a}A: {textwrap.fill(caption_a, 58)}\n"
                        f"{marker_b}B: {textwrap.fill(caption_b, 58)}"
                    )
                    axis.imshow(image_of(dataset_row))
                    axis.set_title(
                        f"{label}: {'Correct' if record['correct'] else 'Wrong'} | Sample {record['index']}",
                        color="green" if record["correct"] else "red",
                        fontsize=13,
                    )
                    axis.text(
                        0.5,
                        -0.08,
                        caption_text,
                        transform=axis.transAxes,
                        ha="center",
                        va="top",
                        fontsize=10,
                    )
                    axis.axis("off")

            figure.suptitle(
                f"{relation.title()}: {title.format(run=run_name.replace('_', ' '))}",
                fontsize=16,
            )
            figure.subplots_adjust(hspace=0.65, wspace=0.15)
            filename = filename_prefix + relation.replace(" ", "_") + ".png"
            figure.savefig(run_output_dir / filename, dpi=180, bbox_inches="tight")
            plt.close(figure)


def plot_prediction_examples(ds: Dataset, output_dir: Path) -> None:
    """Plot examples fixed by full finetuning and by LoRA, relative to baseline."""
    _plot_prediction_transitions(
        ds,
        output_dir,
        baseline_correct=False,
        candidate_correct=True,
        filename_prefix="",
        title="baseline errors corrected by {run}",
    )


def plot_wrong_prediction_examples(ds: Dataset, output_dir: Path) -> None:
    """Plot baseline-correct examples made wrong by full finetuning and LoRA."""
    _plot_prediction_transitions(
        ds,
        output_dir,
        baseline_correct=True,
        candidate_correct=False,
        filename_prefix="wrong_",
        title="baseline successes made wrong by {run}",
    )


def load_vision_adapter(model: LlavaOnevisionForConditionalGeneration, path: Path) -> None:
    """Attach a local PEFT adapter to LLaVA's vision tower for this process only."""
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} was not found. Pass the directory produced by "
            "vision_encoder.save_pretrained(...)."
        )
    try:
        from peft import PeftModel
    except ImportError as error:
        raise RuntimeError(
            "Loading --vision-adapter requires PEFT. Install it with: pip install peft"
        ) from error

    model.model.vision_tower = PeftModel.from_pretrained(
        model.model.vision_tower,
        str(path),
        is_trainable=False,
    )
    print(f"Loaded vision adapter from {path}")


def load_vision_checkpoint(model: LlavaOnevisionForConditionalGeneration, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    model.model.vision_tower.load_state_dict(checkpoint["vision_state_dict"])
    print(f"Loaded vision checkpoint from {path}")


def main() -> None:
    args = parse_args()

    if args.plot_examples or args.plot_wrong_examples:
        random.seed(17)
        ds = prepare_dataset(args)
        if args.plot_examples:
            plot_prediction_examples(ds, QUALITATIVE_RESULTS_DIR)
        if args.plot_wrong_examples:
            plot_wrong_prediction_examples(ds, QUALITATIVE_RESULTS_DIR)
        print(f"Saved qualitative comparisons to {QUALITATIVE_RESULTS_DIR}")
        return

    if args.vision_checkpoint is not None and args.vision_adapter is not None:
        raise ValueError("Pass either --vision-checkpoint or --vision-adapter, not both.")
    if args.use_qlora and args.vision_adapter is None:
        raise ValueError("--use-qlora requires --vision-adapter.")

    # Select device and data type
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    if args.use_qlora and device.type != "cuda":
        raise ValueError("QLoRA requires CUDA and bitsandbytes.")

    # Load model
    print(f"Loading {args.model_id} on {device} with {dtype} ...")
    model_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if args.use_qlora:
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    if args.vision_checkpoint is not None:
        load_vision_checkpoint(model, args.vision_checkpoint)
    if args.vision_adapter is not None:
        load_vision_adapter(model, args.vision_adapter)
    if not args.use_qlora:
        model.to(device=device, dtype=dtype)
    model.eval()

    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        ds = prepare_dataset(args)

        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "left"
        token_a = answer_token_id(processor, "A")
        token_b = answer_token_id(processor, "B")

        seed_output_dir = args.output_dir / f"seed_{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = seed_output_dir / "predictions.jsonl"
        relation_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        total_correct = 0
        total = 0

        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate,
        )
        with predictions_path.open("w", encoding="utf-8") as output, torch.inference_mode():
            for batch_index, rows in enumerate(tqdm(loader, desc="ARO")):
                images, prompts, metadata = [], [], []
                for offset, row in enumerate(rows):
                    true_caption, false_caption = captions_of(row)
                    # Stable randomization independent of batch size and worker scheduling.
                    sample_index = batch_index * args.batch_size + offset
                    true_is_a = random.Random(seed + sample_index).random() < 0.5
                    caption_a, caption_b = (
                        (true_caption, false_caption) if true_is_a
                        else (false_caption, true_caption)
                    )
                    images.append(image_of(row))
                    prompts.append(make_prompt(processor, caption_a, caption_b))
                    metadata.append((sample_index, relation_of(row), true_caption, false_caption, true_is_a))

                inputs = processor(
                    text=prompts, images=images, padding=True, return_tensors="pt"
                ).to(device=device, dtype=dtype)
                logits = model(**inputs, use_cache=False).logits[:, -1, :]
                pair_log_probs = F.log_softmax(logits[:, [token_a, token_b]].float(), dim=-1)
                predictions = pair_log_probs.argmax(dim=-1).cpu().tolist()
                pair_log_probs = pair_log_probs.cpu().tolist()

                for pred, logps, meta in zip(predictions, pair_log_probs, metadata):
                    sample_index, relation, true_caption, false_caption, true_is_a = meta
                    gold = 0 if true_is_a else 1
                    correct = pred == gold
                    total += 1
                    total_correct += int(correct)
                    relation_stats[relation][0] += int(correct)
                    relation_stats[relation][1] += 1
                    output.write(json.dumps({
                        "index": sample_index,
                        "relation": relation,
                        "true_caption": true_caption,
                        "false_caption": false_caption,
                        "gold": "A" if gold == 0 else "B",
                        "prediction": "A" if pred == 0 else "B",
                        "log_p_A_normalized_over_AB": logps[0],
                        "log_p_B_normalized_over_AB": logps[1],
                        "correct": correct,
                    }, ensure_ascii=False) + "\n")

        per_relation = {
            relation: {"correct": c, "total": n, "accuracy": c / n}
            for relation, (c, n) in sorted(relation_stats.items())
        }
        macro = sum(x["accuracy"] for x in per_relation.values()) / len(per_relation)
        summary = {
            "model_id": args.model_id,
            "vision_adapter": (
                str(args.vision_adapter.resolve()) if args.vision_adapter is not None else None
            ),
            "vision_checkpoint": (
                str(args.vision_checkpoint.resolve()) if args.vision_checkpoint is not None else None
            ),
            "use_qlora": args.use_qlora,
            "dataset_id": args.dataset_id,
            "split": args.split,
            "spatial_relations": sorted(DEFAULT_SPATIAL_RELATIONS),
            "seed": seed,
            "num_examples": total,
            "micro_accuracy": total_correct / total,
            "macro_relation_accuracy": macro,
            "per_relation": per_relation,
        }
        summary_path = seed_output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved {predictions_path} and {summary_path}")
        if args.generate_visualization:
            print(
                "--generate-visualization is deprecated; run this script with "
                "--plot-examples after all three prediction files are available."
            )


if __name__ == "__main__":
    main()
