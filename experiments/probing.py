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
    def __init__(self, num_classes, model, num_epochs, criterion, train_loader, test_loader, val_loader):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
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
        print(self.vision.config)
        self.vision.eval()

        for p in self.vision.parameters():
            p.requires_grad = False

        hidden_size = self.vision.config.hidden_size
        self.probe = nn.Sequential(
            nn.Linear(hidden_size, self.num_classes),
        )
        self.probe.to(self.device)
        
    def fit(self, optimizer, patience=5, plot_path=None):
        best_val_loss = float("inf")
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
                    feats = outputs.last_hidden_state[:, 0]

                feats = feats.float()
                logits = self.probe(feats)
                loss = self.criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Loss per item in batch
                running_train_loss += loss.item()

            print(f"Loss after epoch {epoch + 1} = {running_train_loss}")

            # Average loss per batch
            train_loss = running_train_loss / len(self.train_loader)
            train_losses.append(train_loss)

            # ------------------------
            # Validation
            # ------------------------
            val_metrics = self.evaluate(self.val_loader)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            # ------------------------
            # Early stopping checkpoint
            # ------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.probe.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

        # Restore best weights for this run
        if best_state is not None:
            self.probe.load_state_dict(best_state)
            self.probe.to(self.device)

        # Plot curves if requested
        if plot_path is not None:
            self.plot_curves(
                train_losses=train_losses,
                val_losses=val_losses,
                save_path=plot_path,
                title="Probe Training vs Validation Loss",
            )

        return {
            "best_loss": best_val_loss,
            "best_state": best_state,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

    def evaluate(self, loader):
        self.probe.eval()
        self.vision.eval()

        total_loss = 0.0
        total_examples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
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

        best_loss = float("inf")
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

            run_plot_path = f"visualizations/train_val_lr_{lr}.png"
            history = self.fit(
                optimizer=optimizer,
                patience=5,
                plot_path=run_plot_path,
            )

            results[lr] = history

            if history["best_loss"] < best_loss:
                best_loss = history["best_loss"]
                best_lr = lr
                best_state = copy.deepcopy(history["best_state"])

        # Restore and save best model
        if best_state is not None:
            self.probe.load_state_dict(best_state)

        checkpoint = {
            "probe_state_dict": best_state,
            "learning_rate": best_lr,
            "validation_loss": best_loss,
            "num_classes": self.num_classes,
            "epochs": self.num_epochs,
            "activation": self.activation.__name__ if hasattr(self.activation, "__name__") else str(self.activation),
            "vision_model": self.model_name,
        }

        torch.save(checkpoint, save_path)

        print("\n" + "=" * 60)
        print("Hyperparameter search finished")
        print(f"Best learning rate: {best_lr}")
        print(f"Best validation loss: {best_loss:.6f}")
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

