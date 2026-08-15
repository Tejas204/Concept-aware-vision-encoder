import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def finetune_collate_fn(batch):
    """
    --------------------------------------------------------------------------------------------
    Stack fixed-size tensors and preserve variable-length caption lists.

    Args:
        batch: Sequence of fine-tuning samples to combine.

    Returns:
        A batch dictionary containing stacked tensors and per-image caption lists.
    --------------------------------------------------------------------------------------------
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    captions = [sample["captions"] for sample in batch]
    if any(len(image_captions) == 0 for image_captions in captions):
        raise ValueError("Every image must have at least one caption.")
    return {
        "images": torch.stack([sample["images"] for sample in batch]),
        "concept_vector": torch.stack([sample["concept_vector"] for sample in batch]),
        "predicate_vector": torch.stack([sample["predicate_vector"] for sample in batch]),
        "object_vector": torch.stack([sample["object_vector"] for sample in batch]),
        "captions": captions,
    }

class FineTuneLoader(Dataset):
    def __init__(self, data_path, concept_path, object_path, predicate_path, split, probe_type, transform = None):
        self.data_path = data_path
        self.concept_path = concept_path
        self.object_path = object_path
        self.predicate_path = predicate_path
        self.transform = transform
        self.split = split
        self.probe_type = probe_type

        # Open the json file
        with open(self.data_path, "r") as file:
            self.data = json.load(file)

        with open(self.concept_path, "r") as cfile:
            self.concepts = json.load(cfile)

        with open(self.object_path, "r") as ofile:
            self.objects = json.load(ofile)

        with open(self.predicate_path, "r") as pfile:
            self.predicates = json.load(pfile)

        # Create a dictionary of number of entities for each probe type
        entity_dict = {
            "concept": len(self.concepts),
            "object": len(self.objects),
            "predicate": len(self.predicates)
        }


        # Flatten the samples for a given split
        self.samples = []
        for img_id, entry in self.data.items():

            if entry["split"] != self.split:
                continue

            # Construct binary concept vectors
            concept_vector = np.zeros(entity_dict["concept"], dtype=int)
            predicate_vector = np.zeros(entity_dict["predicate"], dtype=int)
            object_vector = np.zeros(entity_dict["object"], dtype=int)
   
            for idx in entry["predicate_indices"]:
                predicate_vector[idx] = 1

            for idx in entry["concept_indices"]:
                concept_vector[idx] = 1

            for idx in entry["object_indices"]:
                object_vector[idx] = 1

            # Create a list of captions for the image
            captions = []
            for annot in entry["annotations"]:
                captions.append(annot["caption"])

            self.samples.append({
                "img_id": img_id,
                "concept_vector": torch.from_numpy(concept_vector),
                "predicate_vector": torch.from_numpy(predicate_vector),
                "object_vector": torch.from_numpy(object_vector),
                "captions": captions,
                "path": entry["path"],
                "url": entry['url'],
            })


    def __len__(self):
        """
        --------------------------------------------------------------------------------------------
        Return the number of fine-tuning samples in the selected split.

        Args:
            None.

        Returns:
            Number of samples in the dataset.
        --------------------------------------------------------------------------------------------
        """
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        --------------------------------------------------------------------------------------------
        Load and transform one image with all of its supervision targets.

        Args:
            index: Zero-based sample index.

        Returns:
            A dictionary containing the image, target vectors, and captions.
        --------------------------------------------------------------------------------------------
        """
        sample = self.samples[index]

        # Fetch and open the image
        image_name = sample["url"].split("/")[-1]
        image_path = os.path.join(sample.get("path", ""), image_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"images": image,
                "concept_vector": sample["concept_vector"],
                "predicate_vector": sample["predicate_vector"],
                "object_vector": sample["object_vector"],
                "captions": sample["captions"]}

    

    def data_loaders(self, split="train", batch_size=4, num_workers=0):
        """
        --------------------------------------------------------------------------------------------
        Create a DataLoader for the given dataset.

        Args:
            split: Loader split name; training loaders are shuffled.
            batch_size: Number of samples per batch.
            num_workers: Number of worker processes used for loading.

        Returns:
            DataLoader: A PyTorch DataLoader for the dataset.

        --------------------------------------------------------------------------------------------
        """

        shuffle = True if split == "train" else False

        loader = torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=finetune_collate_fn)
        return loader
