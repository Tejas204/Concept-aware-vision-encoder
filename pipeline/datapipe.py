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
        Initialize the pipeline with input and output paths.

        The annotations file is loaded immediately so the rest of the
        pipeline methods can operate on the in-memory annotation list.

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
        configured annotations dictionary path as formatted JSON.

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
        Print a small summary of the loaded annotation data.

        The analysis reports the total sample count, split distribution,
        unique predicates, and the number of true and false predicate labels.

        Args:
            None.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        print(f"Total number of annotations are: {len(self.annotations)}")
        
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

        print(f"\nTrain samples: {train_counter}, Validation samples: {val_counter}, Test samples: {test_counter}")
        print(f"\nUnique predicates/concepts are {len(predicates)}: {predicates}")
        print(f"\nTrue predicates: {true_predicates}, False predicates: {false_predicates}\n")


    def filter_samples(self):
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
        # Filter samples where annotations list is empty
        # Filter samples where all predicates are false

        no_annot_samples = []
        for sample_id, annot in self.annotations.items():
            if len(annot["annotations"]) == 0:
                no_annot_samples.append(sample_id)

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

        
        print(f"Images with no annotaions: {len(no_annot_samples)}")
        print(f"Images with all false predicates: {len(all_false_predicates)}")


    def build_captions(self):
        """
        --------------------------------------------------------------------------------------------
        Placeholder for caption generation logic.

        This method is reserved for building textual captions from the
        annotation data.

        Args:
            None.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        pass

    def build_concept_vector(self):
        """
        --------------------------------------------------------------------------------------------
        Placeholder for concept vector construction logic.

        This method is reserved for converting annotations into a concept
        representation suitable for downstream modeling.

        Args:
            None.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        pass
