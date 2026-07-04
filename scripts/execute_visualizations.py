import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import dataset_visualization


if __name__ == "__main__":
    visualizer_object = dataset_visualization.DataVisualizer(
        data_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/metadata/concept_metadata.json",
        storage_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/visualizations/dataset"
    )

    visualizer_object.compute_subject_object_frequency(k=20)
    visualizer_object.compute_pairwise_frequency(k=20)
    visualizer_object.compute_predicate_frequency()
