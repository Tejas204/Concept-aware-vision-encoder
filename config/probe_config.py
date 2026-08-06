import os
import sys
import json
import torch

CONFIG = {
    "concept": {
        "image_storage_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/visualizations/dataset",
        "result_storage_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/results/linear_probe_metrics",
        "checkpoint_path": "checkpoints",
        "model": "/scratch/common_models/llava-onevision-qwen2-7b-si-hf",
        "probe_type": "concept",
        "data_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/concept_metadata.json",
        "concept_path":"/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/concepts.json",
        "object_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/objects.json",
        "predicate_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/predicates.json",
        "activation":torch.nn.Sigmoid,
        "num_classes":14094,
        "num_epochs":100,
        "criterion":torch.nn.BCEWithLogitsLoss()
    },
    "predicate": {
        "image_storage_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/visualizations/dataset",
        "result_storage_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/results/linear_probe_metrics",
        "checkpoint_path": "checkpoints",
        "model": "/scratch/common_models/llava-onevision-qwen2-7b-si-hf",
        "probe_type": "concept",
        "data_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/concept_metadata.json",
        "concept_path":"/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/concepts.json",
        "object_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/objects.json",
        "predicate_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/predicates.json",
        "activation":torch.nn.Sigmoid,
        "num_classes":18,
        "num_epochs":100,
        "criterion":torch.nn.BCEWithLogitsLoss()
    },
    "object": {
        "image_storage_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/visualizations/dataset",
        "result_storage_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/results/linear_probe_metrics",
        "checkpoint_path": "checkpoints",
        "model": "/scratch/common_models/llava-onevision-qwen2-7b-si-hf",
        "probe_type": "concept",
        "data_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/concept_metadata.json",
        "concept_path":"/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/concepts.json",
        "object_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/objects.json",
        "predicate_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/predicates.json",
        "activation":torch.nn.Sigmoid,
        "num_classes":3679,
        "num_epochs":100,
        "criterion":torch.nn.BCEWithLogitsLoss()
    }
}