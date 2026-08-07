import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SpatialSenseDataset(Dataset):
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

            # Construct binary concept vector, depending on probe type:
            # 1. Probing for objects only: cat, dog, table
            # 2. Probing for positional predicates only: on, under.
            # 3. Probing for compositional concepts: cat_on_table
            concept_vector = np.zeros(entity_dict[self.probe_type], dtype=int)
            if self.probe_type == "predicate":
                for idx in entry["predicate_indices"]:
                    concept_vector[idx] = 1
            elif self.probe_type == "concept":
                for idx in entry["concept_indices"]:
                    concept_vector[idx] = 1
            else:
                for idx in entry["object_indices"]:
                    concept_vector[idx] = 1

            self.samples.append({
                "img_id": img_id,
                "url": entry['url'],
                "concept_vector": torch.from_numpy(concept_vector),
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

        if self.transform:
            image = self.transform(image)

        return {"image": image, "concept_vector": sample["concept_vector"], "caption": sample.get("caption", "")}

    

    def data_loaders(self, split="train", batch_size=4, num_workers=0):
        """
        Create a DataLoader for the given dataset.

        Args:
            dataset: The dataset to create a loader for.
            type (str): The type of loader ('train' or 'test'), determines if shuffling is enabled.

        Returns:
            DataLoader: A PyTorch DataLoader for the dataset.
        """

        shuffle = True if split == "train" else False

        loader = torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader
    
    def visualize_images(self, storage_path=None, max_images=9):
        """
        Visualize up to `max_images` random samples from this dataset.
        
        Args:
            storage_path (str, optional): If provided, save the figure here.
            max_images (int): Maximum number of images to display.
        """
        n = min(max_images, len(self))
        if n == 0:
            print("Dataset is empty.")
            return

        indices = random.sample(range(len(self)), n)

        rows = int(np.ceil(n / 3))
        cols = min(3, n)
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

        axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

        for i, ax in enumerate(axes):
            ax.axis("off")

            if i >= n:
                continue

            item = self[indices[i]]
            img = item.get("image")
            caption = item.get("caption", "")

            if isinstance(img, Image.Image):
                arr = np.array(img)
            elif isinstance(img, torch.Tensor):
                t = img.detach().cpu()
                if t.dim() == 4:
                    t = t[0]
                if t.dim() == 3:
                    t = t.permute(1, 2, 0)
                arr = t.numpy()
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255)
                arr = arr.astype(np.uint8)
            else:
                arr = np.array(img)

            ax.imshow(arr)
            ax.set_title(caption, fontsize=8)

        plt.tight_layout()

        if storage_path:
            directory = os.path.dirname(storage_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(storage_path, bbox_inches="tight")