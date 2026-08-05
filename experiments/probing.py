import os
import sys
import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
import torch.functional as functional
import torch.nn as nn
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


class LlavaOneVisionProbe(nn.Module):
    def __init__(self, num_layers, activation, num_classes, model, num_epochs, criterion, optimizer, train_loader, test_loader, learning_rate):
        super().__init__()
        # self.num_layers = num_layers
        self.activation = activation
        self.num_classes = num_classes
        self.model = model
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
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
        
    
    def train(self):

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

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (batch + 1) % 100 == 0:
                    print(f"Epoch: {epoch+1} / {self.epochs}, step {batch+1}/{self.n_total_steps}, loss = {loss.item():.4f}")

        
        print("-"*50)
        print("\nFInished training")

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
