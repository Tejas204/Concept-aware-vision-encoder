import os
import sys
import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
import torch.functional as functional
import torch.nn as nn
import copy
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


class LlavaOneVisionProbe(nn.Module):
    def __init__(self, activation, num_classes, model, num_epochs, criterion, train_loader, test_loader, val_loader):
        super().__init__()
        self.activation = activation
        self.num_classes = num_classes
        self.model = model
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Beginning probe for: {model}")


        # Extract the vision tower and set to eval
        self.model = LlavaOnevisionForConditionalGeneration(model, torch_dtype = torch.float16, device_map = "auto")
        self.vision = model.model.vision_tower
        self.vision.eval()
        self.vision.requires_grad(False)

        self.probe = nn.Sequential(
            nn.Linear(self.vision.config_size, self.num_classes),
            self.activation()
        )
        
    
    def train(self, optimizer):

        # Train the probe over epochs
        for epoch in range(self.num_epochs):
            self.probe.train()

            for batch in self.train_loader:
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.no_grad():
                    outputs = self.vision(
                        pixel_values=images,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    feats = outputs.pooler_output   # shape: [B, D]

                logits = self.probe(feats)
                loss = self.criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch + 1) % 100 == 0:
                    print(f"Epoch: {epoch+1} / {self.epochs}, step {batch+1}/{self.n_total_steps}, loss = {loss.item():.4f}")

        
        print("-"*50)
        print("\nFInished training")

    def validate(self, epochs, learning_rates):
        """Run grid search over `learning_rates` and select the best epoch (with early stopping).

        Args:
            epochs (int): maximum epochs to train for each learning rate.
            learning_rates (iterable): candidate learning rates to try.

        Returns:
            best_info (dict): contains `lr`, `epoch`, and `history` (per-epoch errors) for the best run.
            results (dict): mapping lr -> run information (histories, best_epoch, best_val_loss).
        """
        # Save initial probe parameters so each LR starts from same init
        initial_state = copy.deepcopy(self.probe.state_dict())

        results = {}
        best_overall = {
            "lr": None,
            "epoch": None,
            "val_loss": float("inf"),
            "state": None,
        }

        # Ensure device placement
        device = self.device
        try:
            self.vision.to(device)
        except Exception:
            pass

        for lr in learning_rates:
            # reset probe to initial weights
            self.probe.load_state_dict(initial_state)
            self.probe.to(device)

            optimizer = torch.optim.Adam(self.probe.parameters(), lr=lr)

            val_errors = []
            val_losses = []

            # early stopping params
            patience = 5
            min_delta = 1e-4
            epochs_no_improve = 0
            best_val_loss = float("inf")
            best_epoch = 0
            best_state = None

            for epoch in range(1, epochs + 1):
                # Train one epoch
                self.probe.train()
                for batch in self.train_loader:
                    images = batch["images"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.no_grad():
                        outputs = self.vision(
                            pixel_values=images,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        feats = outputs.pooler_output

                    feats = feats.to(next(self.probe.parameters()).device)

                    logits = self.probe(feats)
                    if self.criterion is not None:
                        loss = self.criterion(logits, labels)
                    else:
                        # fallback: use negative accuracy as loss proxy
                        preds = torch.argmax(logits, dim=1)
                        loss = (preds != labels).float().mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Validation after epoch
                metrics = self.evaluate()
                acc = metrics.get("accuracy", 0.0)
                err = 1.0 - acc
                val_errors.append(err)
                val_loss = metrics.get("loss", err)
                val_losses.append(val_loss)

                # check improvement
                if val_loss + min_delta < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.probe.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

            results[lr] = {
                "val_errors": val_errors,
                "val_losses": val_losses,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_state": best_state,
            }

            # update global best
            if best_val_loss < best_overall["val_loss"]:
                best_overall["lr"] = lr
                best_overall["epoch"] = best_epoch
                best_overall["val_loss"] = best_val_loss
                best_overall["state"] = best_state

        # restore probe to best found state
        if best_overall["state"] is not None:
            self.probe.load_state_dict(best_overall["state"])
            self.probe.to(device)

        # Plot error vs epoch for each learning rate
        plt.figure()
        for lr, info in results.items():
            epochs_range = list(range(1, len(info["val_errors"]) + 1))
            plt.plot(epochs_range, info["val_errors"], label=f"lr={lr}")

        plt.xlabel("Epoch")
        plt.ylabel("Validation Error (1 - accuracy)")
        plt.title("Validation Error vs Epoch for different learning rates")
        plt.legend()
        os.makedirs("visualizations/", exist_ok=True)
        plot_path = os.path.join("visualizations", "validation_error_vs_epoch.png")
        plt.savefig(plot_path)
        plt.close()

        best_info = {
            "lr": best_overall["lr"],
            "epoch": best_overall["epoch"],
            "val_loss": best_overall["val_loss"],
            "plot": plot_path,
        }

        return best_info, results

    def evaluate(self, return_confusion_matrix=False):
        """Evaluate the trained probe on `self.test_loader`.

        Returns:
            metrics (dict): contains 'accuracy' and 'loss' (if criterion provided).
            confusion (np.ndarray, optional): confusion matrix if `return_confusion` True.
        """
        self.probe.eval()
        total = 0
        correct = 0
        total_loss = 0.0

        # prepare confusion matrix if requested
        num_classes = self.num_classes
        if return_confusion_matrix:
            confusion = np.zeros((num_classes, num_classes), dtype=int)

        with torch.no_grad():
            for batch in self.test_loader:
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.vision(
                    pixel_values=images,
                    output_hidden_states=True,
                    return_dict=True,
                )
                feats = outputs.pooler_output  # [B, D]

                feats = feats.to(next(self.probe.parameters()).device) if any(p.requires_grad for p in self.probe.parameters()) else feats.to(self.device)

                logits = self.probe(feats)

                preds = torch.argmax(logits, dim=1)
                batch_size = labels.size(0)
                total += batch_size
                correct += (preds == labels).sum().item()

                if self.criterion is not None:
                    loss = self.criterion(logits, labels)
                    total_loss += loss.item() * batch_size

                if return_confusion_matrix:
                    for t, p in zip(labels.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy()):
                        confusion[int(t), int(p)] += 1

        accuracy = correct / total if total > 0 else 0.0
        metrics = {"accuracy": accuracy}
        if self.criterion is not None:
            metrics["loss"] = total_loss / total if total > 0 else 0.0

        if return_confusion_matrix:
            return metrics, confusion

        return metrics



class LinearProbe(nn.Module):
    def __init__(self, activation, num_classes, model, num_epochs, criterion, train_loader, test_loader, val_loader):
        super().__init__()
        self.activation = activation
        self.num_classes = num_classes
        self.model = model
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Beginning probe for: {model}")


        # Extract the vision tower and set to eval
        self.model = LlavaOnevisionForConditionalGeneration(model, torch_dtype = torch.float16, device_map = "auto")
        self.vision = model.model.vision_tower
        self.vision.eval()
        self.vision.requires_grad(False)

        self.probe = nn.Sequential(
            nn.Linear(self.vision.config_size, self.num_classes),
            self.activation()
        )

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
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.no_grad():
                    outputs = self.vision(
                        pixel_values=images,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    feats = outputs.pooler_output

                feats = feats.to(next(self.probe.parameters()).device)
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

        # Plot curves if requested
        if plot_path is not None:
            plot_curves(
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

        total = 0
        correct = 0
        total_loss = 0

        with torch.no_grad():

            for batch in loader:
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Extract frozen vision features
                outputs = self.vision(
                    pixel_values=images,
                    output_hidden_states=True,
                    return_dict=True,
                )
                feats = outputs.pooler_output  # [B, D]

                feats = feats.to(next(self.probe.parameters()).device)
                logits = self.probe(feats)

                loss = self.criterion(logits, labels)
                batch_size = labels.size(0)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += batch_size

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(loader) if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "loss": avg_loss,
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

@staticmethod
def plot_curves(train_losses, val_losses, save_path=None, title="Training and Validation Curves"):
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

