import os
import sys
import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
import torch.functional as functional
import torch.nn as nn
import copy
import huggingface_hub
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, LlavaOnevisionConfig, SiglipVisionConfig
from sklearn.metrics import (f1_score, precision_score, recall_score,)

class FineTuneSiglip(nn.Module):
    def __init__(self, model, require_object_bottleneck, require_concept_bottleneck, require_predicate_bottleneck):
        """
        --------------------------------------------------------------------------------------------
        Initialize the SigLIP vision-tower fine-tuning module.

        Loads a pretrained LLaVA-OneVision model, extracts its SigLIP vision tower, freezes the
        embedding layer and the first 17 encoder layers, and records the vision hidden size used
        by the optional bottleneck heads.

        Args:
            model (str): Hugging Face model identifier or path to a pretrained model.
            require_object_bottleneck (bool): Whether to enable the object bottleneck head.
            require_concept_bottleneck (bool): Whether to enable the concept bottleneck head.
            require_predicate_bottleneck (bool): Whether to enable the predicate bottleneck head.

        --------------------------------------------------------------------------------------------
        """
        super().__init__()
        self.model_name = model
        self.require_object_bottleneck = require_object_bottleneck
        self.require_concept_bottleneck = require_concept_bottleneck
        self.require_predicate_bottleneck = require_predicate_bottleneck
        print("Initiating fine-tuning!")

        # Set device and arguements
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        kwargs = {"torch_dtype": torch.float16}
        kwargs["device_map"] = "auto" if self.device == "cuda" else "cpu"

        # Load the model
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(model, **kwargs)
        if self.device != "cuda":
            self.model.to(self.device)

        # Load the vision tower
        self.vision = self.model.model.vision_tower

        # Freeze the projection layer
        for param in self.vision.embeddings.parameters():
            param.requires_grad = False

        # Vision tower has 26 encoder layers, freeze first 17 layers - no updates to fine-grained information
        LAYERS_TO_FREEZE = 17
        for layer_idx in range(LAYERS_TO_FREEZE):
            for param in self.vision.encoder.layers[layer_idx].parameters():
                param.requires_grad = False

        # Find the dimensions of hidden layers - useful for projection
        self.hidden_size = self.vision.config.hidden_size


    def create_bottleneck(self):
        """
        --------------------------------------------------------------------------------------------
        Create the requested linear bottleneck heads on top of the vision hidden representation.

        The concept, object, and predicate heads project each vision feature vector to 10,240,
        3,000, and 18 logits, respectively. A head is created only when its corresponding
        ``require_*_bottleneck`` flag is enabled.

        Returns:
            None

        --------------------------------------------------------------------------------------------
        """
        if self.require_concept_bottleneck:
            self.concept_bottleneck = nn.Sequential(
                nn.Linear(self.hidden_size, 10240)
            )

        if self.require_object_bottleneck:
            self.object_bottleneck = nn.Sequential(
                nn.Linear(self.hidden_size, 3000)
            )

        if self.require_predicate_bottleneck:
            self.predicate_bottleneck = nn.Sequential(
                nn.Linear(self.hidden_size, 18)
            )

    def loss_function(self, predictions, ground_truth):
        """
        --------------------------------------------------------------------------------------------
        Compute the weighted binary cross-entropy loss for the enabled bottleneck tasks.

        Args:
            predictions (torch.Tensor): Predicted logits for the enabled bottleneck heads.
            ground_truth (torch.Tensor): Binary target labels matching the predictions.

        Returns:
            torch.Tensor: Weighted sum of the concept, predicate, and object losses.

        --------------------------------------------------------------------------------------------
        """
        LAMBDA_1 = 0.1
        LAMBDA_2 = 0.1
        LAMBDA_3 = 0.1

        if self.require_predicate_bottleneck:
            predicate_loss = nn.BCEWithLogitsLoss()(predictions, ground_truth)
        else:
            predicate_loss = 0

        if self.require_concept_bottleneck:
            concept_loss = nn.BCEWithLogitsLoss()(predictions, ground_truth)
        else:
            concept_loss = 0

        if self.require_object_bottleneck:
            object_loss = nn.BCEWithLogitsLoss()(predictions, ground_truth)
        else:
            object_loss = 0

        loss = LAMBDA_1*concept_loss + LAMBDA_2*predicate_loss + LAMBDA_3*object_loss


obj = FineTuneSiglip(model="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/CAVE/models/llava-onevision-qwen2-0.5b-si-hf")

                    
