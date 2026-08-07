import sys
import os
import json
import torch
from pathlib import Path
from torch.nn import Softmax
import torchvision.transforms as transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pipeline import dataloader
from experiments import probing
from config.probe_config import CONFIG
 
if __name__ == "__main__":

    # -----------------------------------------------------------------------------------
    type = "predicate"

    # define static variables:
    image_storage_path = CONFIG[type]["image_storage_path"]
    result_storage_path = CONFIG[type]["result_storage_path"]
    checkpoint_path = CONFIG[type]["checkpoint_path"]
    model = CONFIG[type]["model"]
    probe_type = CONFIG[type]["probe_type"]

    # -----------------------------------------------------------------------------------
    # define transform
    transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor()])

    # -----------------------------------------------------------------------------------
    # Get training data loader
    training_data = dataloader.SpatialSenseDataset(
        data_path=CONFIG[type]["data_path"],
        concept_path=CONFIG[type]["concept_path"],
        object_path=CONFIG[type]["object_path"],
        predicate_path=CONFIG[type]["predicate_path"],
        probe_type=probe_type,
        transform=transform,
        split="train"
    )
    train_loader = training_data.data_loaders(split="train", batch_size=16)

    # for i in range(5):
    #     training_data.visualize_images(storage_path=f"{image_storage_path}/training_data_{i+1}.png")

    # -----------------------------------------------------------------------------------
    # Get validation data loader
    validation_data = dataloader.SpatialSenseDataset(
        data_path=CONFIG[type]["data_path"],
        concept_path=CONFIG[type]["concept_path"],
        object_path=CONFIG[type]["object_path"],
        predicate_path=CONFIG[type]["predicate_path"],
        probe_type=probe_type,
        transform=transform,
        split="valid"
    )
    validation_loader = validation_data.data_loaders(split="valid", batch_size=16)
    # for i in range(5):
    #     training_data.visualize_images(storage_path=f"{image_storage_path}/validation_data_{i+1}.png")

    # -----------------------------------------------------------------------------------
    # Get testing data loader
    testing_data = dataloader.SpatialSenseDataset(
        data_path=CONFIG[type]["data_path"],
        concept_path=CONFIG[type]["concept_path"],
        object_path=CONFIG[type]["object_path"],
        predicate_path=CONFIG[type]["predicate_path"],
        probe_type=probe_type,
        transform=transform,
        split="test"
    )
    test_loader = testing_data.data_loaders(split="test", batch_size=16)

    # for i in range(5):
    #     testing_data.visualize_images(storage_path=f"{image_storage_path}/testing_data_{i+1}.png")

    # -----------------------------------------------------------------------------------
    # Create object of the probe, train, validate and test
    probe = probing.LinearProbe(
            num_classes=CONFIG[type]["num_classes"],
            model=model,
            num_epochs=CONFIG[type]["num_epochs"],
            criterion=CONFIG[type]["criterion"],
            train_loader=train_loader,
            val_loader=validation_loader,
            test_loader=test_loader,
            type=type
        )

    results, best_lr = probe.hyperparameter_search(
        learning_rates=[1e-2, 1e-3, 1e-4],
        save_path=f"{checkpoint_path}/llava_probe_{probe_type}.pt",
    )

    metrics = probe.evaluate(probe.test_loader)

    print(metrics)

    # -----------------------------------------------------------------------------------
    # Save the results in the file
    result_dir = Path(CONFIG[type]["result_storage_path"])
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / f"{probe_type}_metric.json", "w") as metfile:
        json.dump(metrics, metfile, indent=4)

