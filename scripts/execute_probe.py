import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pipeline import dataloader


if __name__ == "__main__":
    # Get training and testing data loaders
    training_data = dataloader.SpatialSenseDataset(
        data_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/metadata/concept_metadata.json",
        concept_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/metadata/concepts.json",
        transform="",
        split="train",
        storage_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/visualizations/dataset"
    )
    train_loader = training_data.data_loaders(dataset=training_data, split="train")

    testing_data = dataloader.SpatialSenseDataset(
        data_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/metadata/concept_metadata.json",
        concept_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/metadata/concepts.json",
        transform="",
        split="test",
        storage_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/visualizations/dataset"
    )
    test_loader = testing_data.data_loaders(dataset=training_data, split="test")
   
