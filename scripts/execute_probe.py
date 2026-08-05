import sys
import os
import torch
from torch.nn import Softmax
import torchvision.transforms as transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pipeline import dataloader
from experiments import probing


if __name__ == "__main__":

    # define storage filepath:
    storage_path = ""

    # define transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Get training data loader
    training_data = dataloader.SpatialSenseDataset(
        data_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/metadata/concept_metadata.json",
        concept_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/metadata/concepts.json",
        transform=transform,
        split="train"
    )
    train_loader = training_data.data_loaders(split="train", batch_size=4)
    training_data.visualize_images(storage_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/visualizations/dataset/training_data.png")

    # Get testing data loader
    testing_data = dataloader.SpatialSenseDataset(
        data_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/metadata/concept_metadata.json",
        concept_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/metadata/concepts.json",
        transform=transform,
        split="test"
    )
    test_loader = testing_data.data_loaders(split="test", batch_size=4)
    testing_data.visualize_images(storage_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/visualizations/dataset/testing_data.png")

    # Create object of the probe
    probe = probing.LlavaOneVisionProbe(activation=Softmax,
                                        num_classes=10240,
                                        model="",
                                        num_epochs=10,
                                        criterion="")




    # Train and test the probe.
   

    # Save the results in the file
