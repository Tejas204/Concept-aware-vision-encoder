import os
import sys
import json
import torch

CONFIG = {
    "image_storage_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/visualizations/finetune_concept",
    "result_storage_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/results/linear_probe_metrics",
    "checkpoint_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/checkpoints",
    "model": "/scratch/common_models/llava-onevision-qwen2-0.5b-si-hf",
    "siglip_model": "/scratch/common_models/siglip-so400m-patch14-384",
    "data_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/concept_metadata.json",
    "concept_path":"/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/concepts.json",
    "object_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/objects.json",
    "predicate_path": "/nethome/tadhopavkar/Concept-aware-vision-encoder/data/metadata/predicates.json",
    "num_concepts":14094,
    "num_predicates":18,
    "num_objects":3679,
    "num_epochs":15
    }