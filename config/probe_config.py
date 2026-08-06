import os
import sys
import json
import torch

CONFIG = {
    "concept": {
        "image_storage_path": "visualizations/dataset",
        "result_storage_path": "results/linear_probe_metrics",
        "checkpoint_path": "checkpoints",
        "model": "/scratch/common_models/llava-onevision-qwen2-7b-si-hf",
        "probe_type": "concept",
        "data_path": "data/metadata/concept_metadata.json",
        "concept_path":"data/metadata/concepts.json",
        "object_path": "data/metadata/objects.json",
        "predicate_path": "data/metadata/predicates.json",
        "activation":torch.nn.Softmax,
        "num_classes":14094,
        "num_epochs":100,
        "criterion":torch.nn.BCEWithLogitsLoss
    },
    "predicate": {
        "image_storage_path": "visualizations/dataset",
        "result_storage_path": "results/linear_probe_metrics",
        "checkpoint_path": "checkpoints",
        "model": "/scratch/common_models/llava-onevision-qwen2-7b-si-hf",
        "probe_type": "concept",
        "data_path": "data/metadata/concept_metadata.json",
        "concept_path":"data/metadata/concepts.json",
        "object_path": "data/metadata/objects.json",
        "predicate_path": "data/metadata/predicates.json",
        "activation":torch.nn.Softmax,
        "num_classes":18,
        "num_epochs":100,
        "criterion":torch.nn.BCEWithLogitsLoss
    },
    "concept": {
        "image_storage_path": "visualizations/dataset",
        "result_storage_path": "results/linear_probe_metrics",
        "checkpoint_path": "checkpoints",
        "model": "/scratch/common_models/llava-onevision-qwen2-7b-si-hf",
        "probe_type": "concept",
        "data_path": "data/metadata/concept_metadata.json",
        "concept_path":"data/metadata/concepts.json",
        "object_path": "data/metadata/objects.json",
        "predicate_path": "data/metadata/predicates.json",
        "activation":torch.nn.Softmax,
        "num_classes":3679,
        "num_epochs":100,
        "criterion":torch.nn.BCEWithLogitsLoss
    }
}