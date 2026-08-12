import os
import sys
import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import copy
import itertools
import huggingface_hub
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration, SiglipModel
from sklearn.metrics import (f1_score, precision_score, recall_score,)

class FineTuneSiglip(nn.Module):
    def __init__(self, model, train_loader, val_loader, test_loader, num_epochs,siglip_model = "google/siglip-so400m-patch14-384",require_object_bottleneck=False, require_concept_bottleneck=False, require_predicate_bottleneck=False, lambda_concept=1.0, lambda_object=1.0, lambda_predicate=1.0, concept_pos_weight=None, trainable_vision_layers=17, enable_lora=False, use_qlora=False, num_concepts=14094, num_objects=3679, num_predicates=18):
        """
        --------------------------------------------------------------------------------------------
        Initialize the SigLIP vision-tower fine-tuning module.

        Loads a pretrained LLaVA-OneVision model, extracts its SigLIP vision tower, freezes the
        embedding layer and early encoder layers, leaves the later 17 layers trainable, and records
        the vision hidden size used by the optional bottleneck heads.

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
        self.lambda_concept = lambda_concept
        self.lambda_predicate = lambda_predicate
        self.lambda_object = lambda_object
        print("Initiating fine-tuning!")

        # Set device and arguements
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        kwargs = {"torch_dtype": torch.float16}
        kwargs["device_map"] = "auto" if self.device == "cuda" else "cpu"
        if use_qlora:
            if self.device != "cuda":
                raise ValueError("QLoRA requires CUDA and bitsandbytes.")
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

        # --------------------------------------------------------------------------------------------
        # Llava OneVision vision encoder for image encoding
        # --------------------------------------------------------------------------------------------
        # Load LLAVA model
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(model, **kwargs)
        if self.device != "cuda":
            self.model.to(self.device)

        # Load the vision tower from LLAVA
        self.vision = self.model.model.vision_tower
        self.device = next(self.vision.parameters()).device

        # Freeze every parameter first; only the requested later vision layers are enabled below
        self.model.requires_grad_(False)

        # Freeze the projection layer
        for param in self.vision.embeddings.parameters():
            param.requires_grad = False

        # Vision tower has 26 encoder layers, freeze early layers - no updates to fine-grained information
        LAYERS_TO_FREEZE = len(self.vision.encoder.layers) - trainable_vision_layers
        for layer_idx in range(LAYERS_TO_FREEZE):
            for param in self.vision.encoder.layers[layer_idx].parameters():
                param.requires_grad = False

        if not use_qlora:
            for layer_idx in range(LAYERS_TO_FREEZE, len(self.vision.encoder.layers)):
                for param in self.vision.encoder.layers[layer_idx].parameters():
                    param.requires_grad = True

        self.layers_to_freeze = LAYERS_TO_FREEZE
        self.vision_layers = self.vision.encoder.layers
        self.use_qlora = use_qlora
        self.lora_enabled = False

        # Find the dimensions of hidden layers - useful for projection
        self.hidden_size = self.vision.config.hidden_size

        # --------------------------------------------------------------------------------------------
        # SigLip model for text encoding
        # --------------------------------------------------------------------------------------------
        self.siglip = SiglipModel.from_pretrained(self.siglip_model_name)
        self.siglip_text_encoder = self.siglip.text_model.to(self.device)
        self.siglip_processor = AutoProcessor.from_pretrained(self.siglip_model_name)

        # Freeze text encoder
        self.siglip.requires_grad_(False)
        self.siglip_text_encoder.eval()

        # --------------------------------------------------------------------------------------------
        # Vision Pooler from SigLip
        # converts token embeddings to single vector, with scale and bias for the matching loss
        # --------------------------------------------------------------------------------------------
        # self.vision_pooler = self.siglip.vision_model.head
        self.logit_scale = nn.Parameter(self.siglip.logit_scale.detach().clone().to(self.device))
        self.logit_bias = nn.Parameter(self.siglip.logit_bias.detach().clone().to(self.device))

        # Freeze vision pooler, scale and bias
        # for p in self.siglip.vision_model.head.parameters():
        #     p.requires_grad = False
        # self.siglip.logit_scale.requires_grad_(False)
        # self.siglip.logit_bias.requires_grad_(False)

        # Create bottlenecks
        self.create_bottleneck(num_concepts, num_objects, num_predicates)
        for name in ["concept_bottleneck", "predicate_bottleneck", "object_bottleneck"]:
            if hasattr(self, name):
                getattr(self, name).to(self.device)
        concept_pos_weight = torch.as_tensor(concept_pos_weight, dtype=torch.float32, device=self.device) if concept_pos_weight is not None else None
        self.concept_criterion = nn.BCEWithLogitsLoss(pos_weight=concept_pos_weight)

        if enable_lora or use_qlora:
            self.configure_lora(enable=True, use_qlora=use_qlora)

        self.sanity_checks(stage="initialization", trainable_vision_layers=trainable_vision_layers)



    def create_bottleneck(self, num_concepts, num_objects, num_predicates):
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
                nn.Linear(self.hidden_size, num_concepts)
            )

        if self.require_object_bottleneck:
            self.object_bottleneck = nn.Sequential(
                nn.Linear(self.hidden_size, num_objects)
            )

        if self.require_predicate_bottleneck:
            self.predicate_bottleneck = nn.Sequential(
                nn.Linear(self.hidden_size, num_predicates)
            )

    def configure_lora(self, enable=False, use_qlora=False, rank=8, alpha=16, dropout=0.05):
        """
        --------------------------------------------------------------------------------------------
        Optionally add LoRA/QLoRA adapters to the later vision encoder layers.

        --------------------------------------------------------------------------------------------
        """
        if not enable:
            return

        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as error:
            raise ImportError("Install peft to enable LoRA/QLoRA.") from error

        if use_qlora:
            self.vision = prepare_model_for_kbit_training(self.vision)

        lora_config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], layers_to_transform=list(range(self.layers_to_freeze, len(self.vision_layers))), layers_pattern="layers")
        self.vision = get_peft_model(self.vision, lora_config)
        self.model.model.vision_tower = self.vision
        self.lora_enabled = True

    def train(self, mode=True):
        super().train(mode)
        self.siglip_text_encoder.eval()
        # self.vision_pooler.eval()
        return self

    def sanity_checks(self, stage, trainable_vision_layers=None, image_emb=None, text_emb=None, matching_labels=None, captions_per_image=None, logits=None, total_loss=None, losses=None):
        """
        --------------------------------------------------------------------------------------------
        Run initialization, shape, loss, and gradient sanity checks from one function.

        --------------------------------------------------------------------------------------------
        """
        if stage == "initialization":
            assert self.vision.config.hidden_size == self.siglip_text_encoder.config.hidden_size
            # assert self.vision.config.num_hidden_layers == self.siglip.vision_model.config.num_hidden_layers
            # assert self.vision.config.patch_size == self.siglip.vision_model.config.patch_size
            # assert torch.allclose(self.vision_layers[0].self_attn.q_proj.weight.detach().float().cpu(), self.siglip.vision_model.encoder.layers[0].self_attn.q_proj.weight.detach().float().cpu(), atol=1e-5, rtol=1e-4)
            assert all(not parameter.requires_grad for parameter in self.siglip.parameters())
            assert all(not parameter.requires_grad for parameter in self.siglip_text_encoder.parameters())
            assert self.logit_scale.requires_grad and self.logit_bias.requires_grad
            assert all(not parameter.requires_grad for layer in self.vision_layers[:self.layers_to_freeze] for parameter in layer.parameters())
            assert len(self.vision_layers[self.layers_to_freeze:]) == trainable_vision_layers
            if not self.lora_enabled and not self.use_qlora:
                assert all(parameter.requires_grad for layer in self.vision_layers[self.layers_to_freeze:] for parameter in layer.parameters())
            allowed_trainable_parameters = {id(parameter) for layer in self.vision_layers[self.layers_to_freeze:] for parameter in layer.parameters()}
            assert all(not parameter.requires_grad or id(parameter) in allowed_trainable_parameters for parameter in self.model.parameters())
            for name in ["concept_bottleneck", "predicate_bottleneck", "object_bottleneck"]:
                if hasattr(self, name):
                    assert all(parameter.requires_grad for parameter in getattr(self, name).parameters())

        if stage == "shapes":
            batch_size = len(captions_per_image)
            total_captions = sum(len(captions) for captions in captions_per_image)
            assert torch.isfinite(image_emb).all(), "Image embeddings contain NaN or Inf."
            assert torch.isfinite(text_emb).all(), "Text embeddings contain NaN or Inf."
            assert image_emb.shape == (batch_size, self.hidden_size)
            assert text_emb.shape == (total_captions, self.hidden_size)
            assert matching_labels.shape == (batch_size, total_captions)
            assert all(len(captions) > 0 for captions in captions_per_image)
            assert matching_labels.eq(1).sum(dim=1).tolist() == [len(captions) for captions in captions_per_image]
            if logits is not None:
                assert torch.isfinite(logits).all(), "Matching logits contain NaN or Inf."
                assert logits.shape == (batch_size, total_captions)

        if stage == "loss":
            for name, loss in losses.items():
                assert torch.isfinite(loss), f"{name.capitalize()} loss is NaN or Inf: {loss.item()}"
            assert torch.isfinite(total_loss), f"Total loss is NaN or Inf: {total_loss.item()}"

        if stage == "gradients":
            assert all(parameter.grad is None for parameter in self.siglip_text_encoder.parameters())
            assert all(parameter.grad is None for parameter in self.vision.embeddings.parameters())
            assert all(parameter.grad is None for layer in self.vision_layers[:self.layers_to_freeze] for parameter in layer.parameters())
            assert any(parameter.grad is not None for layer in self.vision_layers[self.layers_to_freeze:] for parameter in layer.parameters() if parameter.requires_grad)
            assert self.logit_scale.grad is not None
            assert self.logit_bias.grad is not None
            for name in ["concept_bottleneck", "predicate_bottleneck", "object_bottleneck"]:
                if hasattr(self, name):
                    assert any(parameter.grad is not None for parameter in getattr(self, name).parameters())

    def loss_function(self, image_emb, text_emb, matching_labels, predictions, targets, use_bottlenecks=True):
        """
        --------------------------------------------------------------------------------------------

        --------------------------------------------------------------------------------------------
        """
        # SigLIP matching loss
        image_emb = F.normalize(image_emb.float(), dim=-1)
        text_emb = F.normalize(text_emb.float(), dim=-1)

        logits = (self.logit_scale.float().clamp(max=np.log(100.0)).exp() * image_emb @ text_emb.T + self.logit_bias.float())
        matching_loss = -F.logsigmoid(matching_labels * logits).mean(dim=1).mean()
        total_loss = matching_loss

        losses = {"matching": matching_loss.detach()}

        # Concept loss
        if self.require_concept_bottleneck and use_bottlenecks:
            concept_loss = self.concept_criterion(predictions["concept"], targets["concept"].float())
            total_loss = (total_loss + self.lambda_concept * concept_loss)
            losses["concept"] = concept_loss.detach()

        # Predicate loss
        if self.require_predicate_bottleneck and use_bottlenecks:
            predicate_loss = F.binary_cross_entropy_with_logits(predictions["predicate"], targets["predicate"].float())
            total_loss = (total_loss + self.lambda_predicate * predicate_loss)
            losses["predicate"] = predicate_loss.detach()

        # Object loss
        if self.require_object_bottleneck and use_bottlenecks:
            object_loss = F.binary_cross_entropy_with_logits(predictions["object"], targets["object"].float())
            total_loss = (total_loss + self.lambda_object * object_loss)
            losses["object"] = object_loss.detach()

        losses["total"] = total_loss.detach()
        self.sanity_checks(stage="loss", total_loss=total_loss, losses=losses)
        return total_loss, losses, logits

    def forward(self, optimizer):
        """
        --------------------------------------------------------------------------------------------

        --------------------------------------------------------------------------------------------
        """
        if len(self.train_loader) == 0:
            raise ValueError("Training dataloader is empty.")

        self.train()
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch["images"].to(self.device)
                concept_vector = batch["concept_vector"].to(self.device)
                predicate_vector = batch["predicate_vector"].to(self.device)
                object_vector = batch["object_vector"].to(self.device)
                captions_per_image = batch["captions"]

                # Collate captions and get text embeddings
                flat_captions = [caption for image_captions in captions_per_image for caption in image_captions]
                text_inputs = self.siglip_processor(text = flat_captions, padding = True, truncation = True, return_tensors = "pt")
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                with torch.no_grad():
                    text_outputs = self.siglip_text_encoder(**text_inputs)
                    text_emb = text_outputs.pooler_output

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
                outputs = self.vision(pixel_values=images, output_hidden_states=False, return_dict=True)

                # Pool vision embeddings to get one embedding per image
                # pooler_dtype = next(self.vision_pooler.parameters()).dtype
                # feats = outputs.last_hidden_state.to(pooler_dtype)
                # pooled_image_feats = self.vision_pooler(feats)
                pooled_image_feats = outputs.last_hidden_state.mean(dim=1).float()

                # Pass features to concept bottleneck and get predictions
                predictions = {}
                targets = {}
                if self.require_concept_bottleneck:
                    predictions["concept"] = self.concept_bottleneck(pooled_image_feats.float())
                    targets["concept"] = concept_vector
                if self.require_predicate_bottleneck:
                    predictions["predicate"] = self.predicate_bottleneck(pooled_image_feats.float())
                    targets["predicate"] = predicate_vector
                if self.require_object_bottleneck:
                    predictions["object"] = self.object_bottleneck(pooled_image_feats.float())
                    targets["object"] = object_vector


                # Compute loss
                total_loss, losses, logits = self.loss_function(image_emb=pooled_image_feats,
                                                          text_emb=text_emb,
                                                          matching_labels=labels,
                                                          predictions=predictions,
                                                          targets=targets)

                self.sanity_checks(stage="shapes", image_emb=pooled_image_feats, text_emb=text_emb, matching_labels=labels, captions_per_image=captions_per_image, logits=logits)
                if batch_idx < 5:
                    print(f"Batch {batch_idx+1} | Images: {tuple(images.shape)} | Image embeddings: {tuple(pooled_image_feats.shape)} | Text embeddings: {tuple(text_emb.shape)} | Labels/Logits: {tuple(labels.shape)}")
                    print(" | ".join([f"{name.capitalize()} Loss: {value.item():.4f}" for name, value in losses.items()]))
                train_loss += total_loss.item()

                optimizer.zero_grad()
                total_loss.backward()
                self.sanity_checks(stage="gradients")
                torch.nn.utils.clip_grad_norm_(filter(lambda parameter: parameter.requires_grad, self.parameters()), max_norm=1.0)
                optimizer.step()

            print(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Train Loss: {train_loss/len(self.train_loader):.4f} | "
            )

    def evaluate(self, loader, use_bottlenecks=True):
        """
        --------------------------------------------------------------------------------------------
        

        --------------------------------------------------------------------------------------------
        """
        if len(loader) == 0:
            raise ValueError("Evaluation dataloader is empty.")

        self.eval()
        running_losses = {"matching": 0.0, "total": 0.0}
        for name in ["concept", "predicate", "object"]:
            if getattr(self, f"require_{name}_bottleneck") and use_bottlenecks:
                running_losses[name] = 0.0
        total_images = 0

        with torch.no_grad():
            for batch in loader:
                images = batch["images"].to(self.device)
                captions_per_image = batch["captions"]
                flat_captions = [caption for image_captions in captions_per_image for caption in image_captions]
                text_inputs = self.siglip_processor(text=flat_captions, padding=True, truncation=True, return_tensors="pt")
                text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
                text_emb = self.siglip_text_encoder(**text_inputs).pooler_output

                caption_to_image = []
                for image_idx, image_captions in enumerate(captions_per_image):
                    caption_to_image.extend([image_idx] * len(image_captions))
                caption_to_image = torch.tensor(caption_to_image, device=self.device)
                image_ids = torch.arange(len(captions_per_image), device=self.device).unsqueeze(1)
                labels = torch.where(image_ids == caption_to_image.unsqueeze(0), 1.0, -1.0)

                outputs = self.vision(pixel_values=images, output_hidden_states=False, return_dict=True)
                # pooler_dtype = next(self.vision_pooler.parameters()).dtype
                # pooled_image_feats = self.vision_pooler(outputs.last_hidden_state.to(pooler_dtype))
                pooled_image_feats = outputs.last_hidden_state.mean(dim=1).float()

                predictions = {}
                targets = {}
                if self.require_concept_bottleneck and use_bottlenecks:
                    predictions["concept"] = self.concept_bottleneck(pooled_image_feats.float())
                    targets["concept"] = batch["concept_vector"].to(self.device)
                if self.require_predicate_bottleneck and use_bottlenecks:
                    predictions["predicate"] = self.predicate_bottleneck(pooled_image_feats.float())
                    targets["predicate"] = batch["predicate_vector"].to(self.device)
                if self.require_object_bottleneck and use_bottlenecks:
                    predictions["object"] = self.object_bottleneck(pooled_image_feats.float())
                    targets["object"] = batch["object_vector"].to(self.device)

                _, losses, _ = self.loss_function(pooled_image_feats, text_emb, labels, predictions, targets, use_bottlenecks=use_bottlenecks)
                batch_size = images.size(0)
                total_images += batch_size
                for name in running_losses:
                    if name in losses:
                        running_losses[name] += losses[name].item() * batch_size

        return {name: value / total_images for name, value in running_losses.items()}

    def hyperparameter_search(self, lambda_values, learning_rate=1e-5, patience=3, min_delta=1e-3, save_path="checkpoints/best_finetuned_siglip.pt", plot_path=None):
        """
        --------------------------------------------------------------------------------------------
        

        --------------------------------------------------------------------------------------------
        """
        if not lambda_values:
            raise ValueError("lambda_values cannot be empty.")

        active_heads = [name for name in ["concept", "predicate", "object"] if getattr(self, f"require_{name}_bottleneck")]
        if not active_heads:
            raise ValueError("Lambda search requires at least one active bottleneck.")
        if not isinstance(lambda_values, dict):
            if active_heads != ["concept"]:
                raise ValueError("Pass lambda_values as a dictionary when multiple bottlenecks are active.")
            lambda_values = {"concept": lambda_values}
        if any(name not in active_heads for name in lambda_values):
            raise ValueError("Lambda values were provided for an inactive bottleneck.")
        for name in active_heads:
            lambda_values.setdefault(name, [getattr(self, f"lambda_{name}")])

        original_epochs = self.num_epochs
        initial_vision_state = {name: value.detach().cpu().clone() for name, value in self.vision.state_dict().items()}
        initial_head_states = {name: {key: value.detach().cpu().clone() for key, value in getattr(self, f"{name}_bottleneck").state_dict().items()} for name in active_heads}
        initial_logit_scale = self.logit_scale.detach().cpu().clone()
        initial_logit_bias = self.logit_bias.detach().cpu().clone()
        best_validation_loss = float("inf")
        best_lambdas = None
        best_vision_state = None
        best_head_states = None
        best_logit_scale = None
        best_logit_bias = None
        results = {}

        lambda_combinations = itertools.product(*[lambda_values[name] for name in active_heads])
        for combination in lambda_combinations:
            current_lambdas = dict(zip(active_heads, combination))
            run_name = ", ".join([f"{name}={value}" for name, value in current_lambdas.items()])
            print(f"Starting lambda search for: {run_name}")
            self.vision.load_state_dict(initial_vision_state)
            with torch.no_grad():
                self.logit_scale.copy_(initial_logit_scale.to(self.device))
                self.logit_bias.copy_(initial_logit_bias.to(self.device))
            for name in active_heads:
                getattr(self, f"{name}_bottleneck").load_state_dict(initial_head_states[name])
                setattr(self, f"lambda_{name}", current_lambdas[name])
            optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, self.parameters()), lr=learning_rate, eps=1e-6)
            train_losses = []
            validation_losses = []
            epochs_without_improvement = 0
            run_best_loss = float("inf")
            self.num_epochs = 1

            for epoch in range(original_epochs):
                self.forward(optimizer)
                train_metrics = self.evaluate(self.train_loader)
                validation_metrics = self.evaluate(self.val_loader)
                train_losses.append(train_metrics["total"])
                validation_losses.append(validation_metrics["total"])
                selection_loss = validation_metrics["matching"]
                print(f"Lambdas: {run_name} | Epoch: {epoch+1}/{original_epochs} | Train Loss: {train_metrics['total']:.4f} | Validation Loss: {validation_metrics['total']:.4f} | Validation Matching Loss: {selection_loss:.4f}")

                if selection_loss < run_best_loss - min_delta:
                    run_best_loss = selection_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if selection_loss < best_validation_loss:
                    best_validation_loss = selection_loss
                    best_lambdas = current_lambdas.copy()
                    best_vision_state = {name: value.detach().cpu().clone() for name, value in self.vision.state_dict().items()}
                    best_head_states = {name: {key: value.detach().cpu().clone() for key, value in getattr(self, f"{name}_bottleneck").state_dict().items()} for name in active_heads}
                    best_logit_scale = self.logit_scale.detach().cpu().clone()
                    best_logit_bias = self.logit_bias.detach().cpu().clone()

                if epochs_without_improvement >= patience:
                    print(f"Early stopping for {run_name}: validation loss did not improve by more than {min_delta} for {patience} epochs.")
                    break

            results[run_name] = {"train_losses": train_losses, "validation_losses": validation_losses, "best_validation_loss": run_best_loss}

        self.num_epochs = original_epochs
        self.vision.load_state_dict(best_vision_state)
        with torch.no_grad():
            self.logit_scale.copy_(best_logit_scale.to(self.device))
            self.logit_bias.copy_(best_logit_bias.to(self.device))
        for name in active_heads:
            getattr(self, f"{name}_bottleneck").load_state_dict(best_head_states[name])
            setattr(self, f"lambda_{name}", best_lambdas[name])
        checkpoint = {"vision_state_dict": best_vision_state, "logit_scale": best_logit_scale, "logit_bias": best_logit_bias, "lambdas": best_lambdas, "validation_loss": best_validation_loss, "lora_enabled": self.lora_enabled, "use_qlora": self.use_qlora, "model_name": self.model_name}
        for name in active_heads:
            checkpoint[f"{name}_bottleneck_state_dict"] = best_head_states[name]
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.lora_enabled:
            checkpoint["vision_adapter_path"] = os.path.splitext(save_path)[0] + "_vision_adapter"
            self.vision.save_pretrained(checkpoint["vision_adapter_path"])
        torch.save(checkpoint, save_path)

        if plot_path is not None:
            self.plot_curves(results, plot_path)

        print(f"Best lambdas: {best_lambdas} | Best validation matching loss: {best_validation_loss:.4f}")
        return results, best_lambdas

    def plot_curves(self, results, save_path):
        """
        --------------------------------------------------------------------------------------------
        

        --------------------------------------------------------------------------------------------
        """
        plt.figure(figsize=(8, 5))
        for lambda_value, history in results.items():
            epochs = range(1, len(history["train_losses"]) + 1)
            plt.plot(epochs, history["train_losses"], label=f"Train lambda={lambda_value}")
            plt.plot(epochs, history["validation_losses"], linestyle="--", label=f"Validation lambda={lambda_value}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Fine-tuning Training and Validation Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def evaluate_best_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.vision.load_state_dict(checkpoint["vision_state_dict"])
        with torch.no_grad():
            self.logit_scale.copy_(checkpoint["logit_scale"].to(self.device))
            self.logit_bias.copy_(checkpoint["logit_bias"].to(self.device))
        test_metrics = self.evaluate(self.test_loader, use_bottlenecks=False)
        print(f"Test losses: {test_metrics}")
        return test_metrics


                    
