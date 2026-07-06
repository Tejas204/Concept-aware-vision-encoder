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
    def __init__(self, num_layers, activation, num_classes, model, num_epochs, criterion, optimizer, train_loader):
        super().__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.num_classes = num_classes
        self.model = model
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Beginning probe for: {model}")

    def vision_tower(self):
        # Extract the vision tower
        model = LlavaOnevisionForConditionalGeneration(model, torch_dtype = torch.float16, device_map = "auto")
        vision = model.model.vision_tower

        # Set to eval mode
        vision.eval()
        vision.requires_grad(False)

        return vision

    
    def vision_probe(self):
        vision_encoder = self.VisionTower()

        # Setup linear probe
        probe = nn.Linear(vision_encoder.config.hidden_size, self.num_classes).to(vision_encoder.device)
        return vision_encoder, probe
    
    def train_probe(self):

        vision_encoder, probe = self.vision_probe()

        # Train the probe over epochs
        for epoch in range(self.num_epochs):
            probe.train()

            for batch in self.train_loader:
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.no_grad():
                    outputs = vision_encoder(
                        pixel_values=images,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    feats = outputs.pooler_output   # shape: [B, D]

                logits = probe(feats)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (batch + 1) % 100 == 0:
                    print(f"Epoch: {epoch+1} / {self.epochs}, step {batch+1}/{self.n_total_steps}, loss = {loss.item():.4f}")

        
        print("-"*50)
        print("\nFInished training")
