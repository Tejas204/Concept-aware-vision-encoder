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
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from sklearn.metrics import (f1_score, precision_score, recall_score,)



class LinearProbe(nn.Module):
    def __init__(self, num_classes, model, num_epochs, criterion, train_loader, test_loader, val_loader, type):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.type = type
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Beginning probe for: {model}")


        # Extract the vision tower and set to eval
        kwargs = {"torch_dtype": torch.float16}

        if self.device == "cuda":
            kwargs["device_map"] = "auto"

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(model, **kwargs)

        if self.device != "cuda":
            self.model.to(self.device)

        self.vision = self.model.model.vision_tower
        self.vision.eval()

        for p in self.vision.parameters():
            p.requires_grad = False

        hidden_size = self.vision.config.hidden_size
        self.probe = nn.Sequential(
            nn.Linear(hidden_size, self.num_classes),
        )
        self.probe.to(self.device)
        
    def fit(self, optimizer, patience=3, plot_path=None):
        if len(self.train_loader) == 0:
            raise ValueError("Training dataloader is empty.")
        
        best_f1 = -1.0
        best_state = None
        epochs_without_improvement = 0

        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):

            # ------------------------
            # Training
            # ------------------------
            self.probe.train()
            self.vision.eval()

            running_train_loss = 0.0

            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["concept_vector"].float().to(self.device)

                with torch.no_grad():
                    outputs = self.vision(
                        pixel_values=images,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                    # feats = outputs.last_hidden_state[:, 0].float()
                    feats = outputs.last_hidden_state.mean(dim=1).float()

                logits = self.probe(feats)

                loss = self.criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            train_loss = running_train_loss / len(self.train_loader)
            train_losses.append(train_loss)

            # ------------------------
            # Validation
            # ------------------------
            val_metrics = self.evaluate(self.val_loader)

            val_losses.append(val_metrics["loss"])

            print(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Micro F1: {val_metrics['f1']:.4f} | "
                f"Sample Acc: {val_metrics['sample_accuracy']:.4f}"
            )

            # ------------------------
            # Early stopping
            # ------------------------
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_state = copy.deepcopy(self.probe.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping.")
                break

        if best_state is not None:
            self.probe.load_state_dict(best_state)
            self.probe.to(self.device)

        if plot_path is not None:
            self.plot_curves(
                train_losses,
                val_losses,
                plot_path,
                "Training vs Validation Loss",
            )

        return {
            "best_f1": best_f1,
            "best_state": best_state,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

    def evaluate(self, loader):
        if len(loader) == 0:
            raise ValueError("Evaluation dataloader is empty.")
        
        self.probe.eval()
        self.vision.eval()

        total_loss = 0.0
        total_examples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                print(f"Processing batch {idx+1}/{len(loader)}")
                images = batch["image"].to(self.device)
                labels = batch["concept_vector"].float().to(self.device)

                outputs = self.vision(
                    pixel_values=images,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Same feature extraction as training
                feats = outputs.last_hidden_state[:, 0].float()

                logits = self.probe(feats)

                loss = self.criterion(logits, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss = total_loss / total_examples

        if not all_preds:
            raise ValueError("No predictions were generated during evaluation.")
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Label-wise accuracy
        accuracy = (all_preds == all_labels).mean()

        # Sample-wise (exact match) accuracy
        sample_accuracy = (all_preds == all_labels).all(axis=1).mean()

        # Overall metrics
        precision = precision_score(
            all_labels,
            all_preds,
            average="micro",
            zero_division=0,
        )

        recall = recall_score(
            all_labels,
            all_preds,
            average="micro",
            zero_division=0,
        )

        f1 = f1_score(
            all_labels,
            all_preds,
            average="micro",
            zero_division=0,
        )

        # Per-concept metrics
        per_concept_accuracy = (all_preds == all_labels).mean(axis=0)

        per_concept_precision = precision_score(
            all_labels,
            all_preds,
            average=None,
            zero_division=0,
        )

        per_concept_recall = recall_score(
            all_labels,
            all_preds,
            average=None,
            zero_division=0,
        )

        per_concept_f1 = f1_score(
            all_labels,
            all_preds,
            average=None,
            zero_division=0,
        )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "sample_accuracy": sample_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_concept_accuracy": per_concept_accuracy,
            "per_concept_precision": per_concept_precision,
            "per_concept_recall": per_concept_recall,
            "per_concept_f1": per_concept_f1,
        }



    def hyperparameter_search(
        self,
        learning_rates,
        save_path="checkpoints/best_probe.pt"
    ):

        initial_weights = copy.deepcopy(self.probe.state_dict())

        best_f1 = -1.0
        best_lr = None
        best_state = None
        results = {}

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs("visualizations", exist_ok=True)

        for lr in learning_rates:
            print("=" * 60)
            print(f"Training probe with learning rate = {lr}")
            print("=" * 60)

            # Reset to the same initialization for each LR
            self.probe.load_state_dict(initial_weights)

            optimizer = torch.optim.Adam(self.probe.parameters(), lr=lr)

            run_plot_path = f"/nethome/tadhopavkar/Concept-aware-vision-encoder/visualizations/{self.type}/train_val_lr_{lr}.png"
            history = self.fit(
                optimizer=optimizer,
                patience=3,
                plot_path=run_plot_path,
            )

            results[lr] = history

            if history["best_f1"] > best_f1:
                best_f1 = history["best_f1"]
                best_lr = lr
                best_state = copy.deepcopy(history["best_state"])

        # Restore and save best model
        if best_state is not None:
            self.probe.load_state_dict(best_state)

        checkpoint = {
            "probe_state_dict": best_state,
            "learning_rate": best_lr,
            "validation_f1": best_f1,
            "num_classes": self.num_classes,
            "epochs": self.num_epochs,
            "vision_model": self.model_name,
        }

        if best_state is None:
            raise RuntimeError("No best model was found during hyperparameter search.")

        torch.save(checkpoint, save_path)

        print("\n" + "=" * 60)
        print("Hyperparameter search finished")
        print(f"Best learning rate: {best_lr}")
        print(f"Best validation loss: {best_f1:.6f}")
        print(f"Checkpoint saved to: {save_path}")
        print("=" * 60)

        return results, best_lr


    def plot_curves(self, train_losses, val_losses, save_path=None, title="Training and Validation Curves"):
        plt.figure(figsize=(8, 5))

        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", color="blue")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color="orange")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")

