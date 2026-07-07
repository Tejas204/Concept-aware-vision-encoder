import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SpatialSenseDataset(Dataset):
    def __init__(self, data_path, concept_path, split, transform = None):
        self.data_path = data_path
        self.concept_path = concept_path
        self.transform = transform
        self.split = split

        # Open the json file
        with open(self.data_path, "r") as file:
            self.data = json.load(file)

        with open(self.concept_path, "r") as file:
            self.concepts = json.load(file)

        # Flatten the samples for a given split
        self.samples = []
        for img_id, entry in self.data.items():

            if entry["split"] != self.split:
                continue

            annotations = entry["annotations"]
            for ann in annotations:
                self.samples.append({
                    "img_id": img_id,
                    "url": entry["url"],
                    "predicate": ann["predicate"],
                    "subject": ann["subject"],
                    "object": ann["object"],
                    "label": ann["label"],
                    "caption": ann["caption"],
                    "concept_indices": ann["concept_indices"],
                    "path": entry["path"]
                })


    def __len__(self):
        """
        
        """
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        
        """
        sample = self.samples[index]

        # Fetch and open the image
        image_name = sample["url"].split("/")[-1]
        image_path = os.path.join(sample.get("path", ""), image_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")

        # Generate the concept one-hot vector
        n_concepts = len(self.concepts)
        one_hot_concept_vector = np.zeros(n_concepts)
        for idx in sample["concept_indices"]:
            one_hot_concept_vector[int(idx)] = 1

        # Transform the vector to tensor
        concept_tensor = torch.from_numpy(one_hot_concept_vector)

        if self.transform:
            image = self.transform(image)

        return {"image": image, "concept": concept_tensor, "caption": sample.get("caption", "")}

    

    def data_loaders(self, dataset=None, split="train", batch_size=4, num_workers=0):
        """
        Create a DataLoader for the given dataset.

        Args:
            dataset: The dataset to create a loader for.
            type (str): The type of loader ('train' or 'test'), determines if shuffling is enabled.

        Returns:
            DataLoader: A PyTorch DataLoader for the dataset.
        """
        if dataset is None:
            dataset = self

        shuffle = True if split == "train" else False

        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader