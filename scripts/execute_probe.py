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
    image_storage_path = "visualizations/dataset"
    result_storage_path = "results/linear_probe_metrics"
    checkpoint_path = "checkpoints"
    model = "/scratch/common_models/llava-onevision-qwen2-7b-si-hf"

    # -----------------------------------------------------------------------------------
    # define transform
    transform = transforms.Compose([transforms.ToTensor()])

    # -----------------------------------------------------------------------------------
    # Get training data loader
    training_data = dataloader.SpatialSenseDataset(
        data_path="data/metadata/concept_metadata.json",
        concept_path="data/metadata/concepts.json",
        object_path="data/metadata/objects.json",
        predicate_path="data/metadata/predicates.json",
        probe_type="concepts",
        transform=transform,
        split="train"
    )
    train_loader = training_data.data_loaders(split="train", batch_size=4)

    for i in range(5):
        training_data.visualize_images(storage_path=f"{image_storage_path}/training_data_{i+1}.png")

    # -----------------------------------------------------------------------------------
    # Get validation data loader
    validation_data = dataloader.SpatialSenseDataset(
        data_path="data/metadata/concept_metadata.json",
        concept_path="data/metadata/concepts.json",
        object_path="data/metadata/objects.json",
        predicate_path="data/metadata/predicates.json",
        probe_type="concepts",
        transform=transform,
        split="valid"
    )
    validation_loader = validation_data.data_loaders(split="valid", batch_size=4)
    for i in range(5):
        training_data.visualize_images(storage_path=f"{image_storage_path}/validation_data_{i+1}.png")

    # -----------------------------------------------------------------------------------
    # Get testing data loader
    testing_data = dataloader.SpatialSenseDataset(
        data_path="data/metadata/concept_metadata.json",
        concept_path="data/metadata/concepts.json",
        object_path="data/metadata/objects.json",
        predicate_path="data/metadata/predicates.json",
        probe_type="concepts",
        transform=transform,
        split="test"
    )
    test_loader = testing_data.data_loaders(split="test", batch_size=4)

    for i in range(5):
        testing_data.visualize_images(storage_path=f"{image_storage_path}/testing_data_{i+1}.png")

    # -----------------------------------------------------------------------------------
    # Create object of the probe, train, validate and test
    probe = probing.LlavaOneVisionProbe(
            activation=Softmax,
            num_classes=10240,
            model=model,
            num_epochs=100,
            criterion=torch.nn.BCEWithLogitsLoss,
            train_loader=train_loader,
            val_loader=validation_loader,
            test_loader=test_loader,
        )

    results, best_lr = probe.hyperparameter_search(
        learning_rates=[1e-2, 3e-3, 1e-3, 3e-4],
        save_path=f"{checkpoint_path}/llava_probe_concept.pt",
    )

    metrics = probe.evaluate(probe.test_loader)

    print(metrics)

    # -----------------------------------------------------------------------------------
    # Save the results in the file
    with open(f"{results}/concept_metric.json", "w") as metfile:
        json.dump(metrics, metfile, indent=4)

