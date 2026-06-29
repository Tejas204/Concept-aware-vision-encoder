import os
import sys
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

class DataPipeline():
    def __init__(self, annotations_path: str, images_path: str, annotation_distionary_path: str):
        """
        --------------------------------------------------------------------------------------------
        Initialize the pipeline with the annotation and image paths.

        The annotations file is loaded immediately so later methods can work
        with the in-memory annotation list.

        Args:
            annotations_path: Path to the source annotations JSON file.
            images_path: Path to the image directory or image source.
            annotation_distionary_path: Path where the dictionary version of
                the annotations will be written.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        self.annotation_distionary_path = annotation_distionary_path
        self.annotations_path = annotations_path
        self.images_path = images_path

        with open(self.annotations_path, "r") as annotfile:
            self.annotations_list = json.load(annotfile)


    def annotations_to_dictionary(self):
        """
        --------------------------------------------------------------------------------------------
        Convert the annotation list into a dictionary indexed by sample id.

        The resulting dictionary is stored on the instance and written to the
        configured output path as formatted JSON.

        Args:
            None.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        self.annotations = {}
        for sample_id, annot in enumerate(self.annotations_list):
            self.annotations[str(sample_id+1)] = annot

        with open(self.annotation_distionary_path, "w") as annotdictfile:
            json.dump(self.annotations, annotdictfile, indent=4)


    def run_basic_analysis(self):
        """
        --------------------------------------------------------------------------------------------
        Print a compact summary of the loaded annotation data.

        The analysis reports the total sample count, split distribution,
        unique predicates, and the number of true and false predicate labels.

        Args:
            None.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        print(f"Total number of images are: {len(self.annotations)}")
        
        predicates = set()
        true_predicates, false_predicates = 0, 0
        train_counter, test_counter, val_counter = 0, 0, 0

        # Find training, validation, testing samples
        for _, annot in self.annotations.items():
            if annot["split"].lower() == "train":
                train_counter += 1
            elif annot["split"].lower() == "valid":
                val_counter += 1
            else:
                test_counter += 1

            # Find unique predicate values
            for item in annot["annotations"]:
                predicates.add(item["predicate"])
                if item["label"]:
                    true_predicates += 1
                else:
                    false_predicates += 1

        print(f"\nTrain images: {train_counter}, Validation images: {val_counter}, Test images: {test_counter}")
        print(f"\nUnique predicates are {len(predicates)}: {predicates}")


    def filter_samples(self, remove_all_false_annotations: bool, storage_path: str):
        """
        --------------------------------------------------------------------------------------------
        Identify samples that have no annotations or only false predicates.

        The method collects sample ids for both cases and prints the counts.

        Args:
            None.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        # Find samples where annotations list is empty
        no_annot_samples = []
        for sample_id, annot in self.annotations.items():
            if len(annot["annotations"]) == 0:
                no_annot_samples.append(sample_id)

        print(f"\nImages with no annotaions: {len(no_annot_samples)}")

        # Find samples where all predicates are false
        if remove_all_false_annotations:
            all_false_predicates = []
            for sample_id, annot in self.annotations.items():
                all_false_flag = True
                for item in annot["annotations"]:
                    if not item["label"]:
                        continue
                    else:
                        all_false_flag = False
                        break

                if all_false_flag:
                    all_false_predicates.append(sample_id)
            print(f"\nImages with all false predicates: {len(all_false_predicates)}")

        # Filter out samples with no annotations
        self.filtered_annotations = {}
        for sample_id, annot in self.annotations.items():
            if remove_all_false_annotations:
                if sample_id in no_annot_samples or sample_id in all_false_predicates:
                    continue
                else:
                    self.filtered_annotations[sample_id] = annot
            else:
                if sample_id in no_annot_samples:
                    continue
                else:
                    self.filtered_annotations[sample_id] = annot

        print(f"\nTotal images removed: {len(no_annot_samples) + len(all_false_predicates)}")
        print(f"\nImages left after filtering are: {len(self.filtered_annotations)}")


        # Save the filtered annotations
        with open(storage_path, "w") as file:
            json.dump(self.filtered_annotations, file, indent=4)


    def extract_concepts(self, storage_path: str):
        """
        --------------------------------------------------------------------------------------------
        Extract the unique concept labels from the filtered annotations.

        Positive relations are encoded as subject_predicate_object, while
        negative relations are encoded with a _not_ marker.

        Args:
            None.

        Returns:
            A list of unique concept strings.

        --------------------------------------------------------------------------------------------
        """
        unique_concepts = set()

        # Find all unique concepts
        for _, annot in self.filtered_annotations.items():
            for relation in annot["annotations"]:
                # Positive concept - presence of concept
                if relation["label"]:
                    concept = relation["subject"]["name"] + "_" + relation["predicate"] + "_" + relation["object"]["name"]
                # Negative concept - absence of concept
                else:
                    concept = relation["subject"]["name"] + "_not_" + relation["predicate"] + "_" + relation["object"]["name"]
                unique_concepts.add(concept)

        # Store concept vector
        self.unique_concept_list = list(unique_concepts)
        with open(storage_path, "w") as file:
            json.dump(self.unique_concept_list, file, indent=4)

        return self.unique_concept_list
    

    def build_caption(self, storage_path: str):
        """
        --------------------------------------------------------------------------------------------
        Build and store captions for each relation in the filtered dataset.

        Positive relations are written as plain subject-predicate-object text,
        while negative relations use the word "not" in the caption.

        Args:
            storage_path: Path where the caption-enriched dataset is saved.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        # Build positive and negative captions
        for _, annot in self.filtered_annotations.items():
            for relation in annot["annotations"]:
                if relation["label"]:
                    caption = relation["subject"]["name"] + " " + relation["predicate"] + " " + relation["object"]["name"]
                else:
                    caption = relation["subject"]["name"] + " not " + relation["predicate"] + " " + relation["object"]["name"]

                relation["caption"] = caption

        # Save the dataset with captions
        with open(storage_path, "w") as file:
            json.dump(self.filtered_annotations, file, indent=4)


    def build_concept_vector(self, unique_concepts: list, storage_path: str):
        """
        --------------------------------------------------------------------------------------------
        Build a concept vector for each relation in the filtered annotations.

        Each relation is mapped to a one-hot vector over the provided concept
        vocabulary and a normalized copy is stored alongside it.

        Args:
            None.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        for _, annot in self.filtered_annotations.items():
            for relation in annot["annotations"]:
                # Store only concept indices for a relation, dense vector generated at run-time
                concept_indices = []

                # Build individual concepts
                if relation["label"]:
                    concept = relation["subject"]["name"] + "_" + relation["predicate"] + "_" + relation["object"]["name"]
                else:
                    concept = relation["subject"]["name"] + "_not_" + relation["predicate"] + "_" + relation["object"]["name"]
                
                if concept in unique_concepts:
                    # Find the index of concept and mark it as 1
                    concept_indices.append(unique_concepts.index(concept))

                relation["concept_indices"] = concept_indices


        # Save the final dataset
        with open(storage_path, "w") as file:
            json.dump(self.filtered_annotations, file, indent=4)

        print("\nData processing is finished!")

        
