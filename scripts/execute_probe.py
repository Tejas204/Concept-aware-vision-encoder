import sys
import os
import json
import torch
from torch.nn import Softmax
import torchvision.transforms as transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pipeline import dataloader
from experiments import probing
 
if __name__ == "__main__":

    # -----------------------------------------------------------------------------------
    # define storage filepath:
    storage_path = ""

    # -----------------------------------------------------------------------------------
    # define transform
    transform = transforms.Compose([transforms.ToTensor()])

    # -----------------------------------------------------------------------------------
    # Get training data loader
    training_data = dataloader.SpatialSenseDataset(
        data_path="data/metadata/concept_metadata.json",
        concept_path="data/metadata/concepts.json",
        transform=transform,
        split="train"
    )
    train_loader = training_data.data_loaders(split="train", batch_size=4)
    for i in range(5):
        training_data.visualize_images(storage_path=f"visualizations/dataset/training_data_{i+1}.png")

    # -----------------------------------------------------------------------------------
    # Get validation data loader
    validation_data = dataloader.SpatialSenseDataset(
        data_path="data/metadata/concept_metadata.json",
        concept_path="data/metadata/concepts.json",
        transform=transform,
        split="valid"
    )
    valudation_loader = validation_data.data_loaders(split="valid", batch_size=4)
    for i in range(5):
        training_data.visualize_images(storage_path=f"visualizations/dataset/validation_data_{i+1}.png")

    # -----------------------------------------------------------------------------------
    # Get testing data loader
    testing_data = dataloader.SpatialSenseDataset(
        data_path="data/metadata/concept_metadata.json",
        concept_path="data/metadata/concepts.json",
        transform=transform,
        split="test"
    )
    test_loader = testing_data.data_loaders(split="test", batch_size=4)
    for i in range(5):
        testing_data.visualize_images(storage_path=f"visualizations/dataset/testing_data_{i+1}.png")

    # -----------------------------------------------------------------------------------
    # Create object of the probe
    probe = probing.LlavaOneVisionProbe(activation=Softmax,
                                        num_classes=10240,
                                        model="",
                                        num_epochs=10,
                                        criterion=torch.nn.BCEWithLogitsLoss(),
                                        train_loader=train_loader,
                                        test_loader=test_loader)

    # -----------------------------------------------------------------------------------
    # Train and test the probe.
    probe.train(optimizer=torch.optim.Adam(probe.probe.parameters(), lr=0.01))

    # -----------------------------------------------------------------------------------
    # Save weights
    torch.save(probe.state_dict(), 'model_weights.pth')

    # -----------------------------------------------------------------------------------
    # Test
    metrics = probe.evaluate()

    # -----------------------------------------------------------------------------------
    # Save the results in the file
    with open("", "w") as metfile:
        json.dump(metrics, metfile, indent=4)

