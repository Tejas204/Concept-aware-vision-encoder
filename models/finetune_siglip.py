import os
import sys
import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import copy
import huggingface_hub
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, SiglipModel
from sklearn.metrics import (f1_score, precision_score, recall_score,)

class FineTuneSiglip(nn.Module):
    def __init__(self, model, train_loader, val_loader, test_loader, num_epochs,siglip_model = "google/siglip-so400m-patch14-384",require_object_bottleneck=False, require_concept_bottleneck=False, require_predicate_bottleneck=False):
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
        self.siglip_model_name = siglip_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        print("Initiating fine-tuning!")

        # Set device and arguements
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        kwargs = {"torch_dtype": torch.float16}
        kwargs["device_map"] = "auto" if self.device == "cuda" else "cpu"

        # --------------------------------------------------------------------------------------------
        # Llava OneVision vision encoder for image encoding
        # --------------------------------------------------------------------------------------------
        # Load LLAVA model
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(model, **kwargs)
        if self.device != "cuda":
            self.model.to(self.device)

        # Load the vision tower from LLAVA
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

        # --------------------------------------------------------------------------------------------
        # SigLip model for text encoding
        # --------------------------------------------------------------------------------------------
        self.siglip = SiglipModel.from_pretrained(self.siglip_model_name)
        self.siglip_text_encoder = self.siglip.text_model
        self.siglip_processor = AutoProcessor.from_pretrained(self.siglip_model_name)

        # Freeze text encoder
        self.siglip_text_encoder.requires_grad_(False)
        self.siglip_text_encoder.eval()

        # --------------------------------------------------------------------------------------------
        # Vision Pooler from SigLip - converts token embeddings to single vector, with scale and bias
        # scale and bias for the matching loss
        # --------------------------------------------------------------------------------------------
        self.vision_pooler = self.siglip.vision_model.head
        self.logit_scale = self.siglip.logit_scale
        self.logit_bias = self.siglip.logit_bias

        # Freeze vision pooler, scale and bias
        for p in self.siglip.vision_model.head.parameters():
            p.requires_grad = False
        self.siglip.logit_scale.requires_grad_(False)
        self.siglip.logit_bias.requires_grad_(False)

        # Create bottlenecks
        self.create_bottleneck()

        assert self.vision.config.hidden_size == self.siglip.vision_model.config.hidden_size



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

    def loss_function(self, image_emb, text_emb, matching_labels, predictions, targets,):
        """
        --------------------------------------------------------------------------------------------

        --------------------------------------------------------------------------------------------
        """
        # SigLIP matching loss
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        logits = (-self.logit_scale.exp() * image_emb @ text_emb.T + self.logit_bias)
        matching_loss = -F.logsigmoid(matching_labels * logits).mean()
        total_loss = matching_loss

        losses = {"matching": matching_loss.detach()}

        # Concept loss
        if self.require_concept_bottleneck:
            concept_loss = self.concept_criterion(predictions["concept"], targets["concept"].float())
            total_loss = (total_loss + self.lambda_concept * concept_loss)
            losses["concept"] = concept_loss.detach()

        # Predicate loss
        if self.require_predicate_bottleneck:
            predicate_loss = F.binary_cross_entropy_with_logits(predictions["predicate"], targets["predicate"].float())
            total_loss = (total_loss + self.lambda_predicate * predicate_loss)
            losses["predicate"] = predicate_loss.detach()

        # Object loss
        if self.require_object_bottleneck:
            object_loss = F.binary_cross_entropy_with_logits(predictions["object"], targets["object"].float())
            total_loss = (total_loss + self.lambda_object * object_loss)
            losses["object"] = object_loss.detach()

        return total_loss, losses

    def forward(self, optimizer):
        if len(self.train_loader) == 0:
            raise ValueError("Training dataloader is empty.")

        for epoch in self.num_epochs:
            train_loss = 0
            for batch in self.train_loader:
                images = batch["images"].to(self.device)
                concept_vector = batch["concept_vector"].to(self.device)
                predicate_vector = batch["predicate_vector"].to(self.device)
                object_vector = batch["object_vector"].to(self.device)
                captions_per_image = batch["captions"]

                # Collate captions and get text embeddings
                flat_captions = [caption for image_captions in captions_per_image for caption in image_captions]
                text_inputs = self.siglip_processor(text = flat_captions, padding = True, truncation = True, return_tensors = "pt")
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

                # Keep track of captions
                caption_to_image = []
                for image_idx, image_captions in enumerate(captions_per_image):
                    caption_to_image.extend([image_idx] * len(image_captions))

                caption_to_image = torch.tensor(caption_to_image, device=self.device)

                # Construct siglip label matrix, +1 for matching captions, -1 for non-matching captions
                batch_size = len(captions_per_image)
                image_ids = torch.arange(batch_size, device=self.device).unsqueeze(1)
                labels = torch.where(image_ids == caption_to_image.unsqueeze(0), 1.0, -1.0,)

                # Get vision embeddings
                outputs = self.vision(pixel_values=images, output_hidden_states=True, return_dict=True)

                # Pool vision embeddings
                with torch.no_grad():
                    feats = outputs.last_hidden_state[:, 0].float()
                    pooled_image_feats = self.vision_pooler(feats)

                # TODO: Compute bottleneck - how to pass inputs to bottleneck for concept vector, predicate vector and
                # object vector, this should return a dict like
                # predictions = {
                #   "concepts": [0, 1.0, 0, 0, ..],
                #   "predicates": [0, 1.0, 0, 0, ..],
                #   "objects": [0, 1.0, 0, 0, ..],
                # }
                self.create_bottleneck()

                loss = (1/len(images))*self.loss_function(image_emb=pooled_image_feats,
                                                          text_emb=text_inputs,
                                                          matching_labels=labels,
                                                          predictions="",
                                                          targets="")

                train_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
            )





obj = FineTuneSiglip(model="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/CAVE/models/llava-onevision-qwen2-0.5b-si-hf")

                    
