import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pipeline import datapipe


if __name__ == "__main__":
    datapipe_object = datapipe.DataPipeline(
        annotations_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/annotations.json",
        annotation_distionary_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/raw_metadata.json",
        images_path="None"
    )

    datapipe_object.annotations_to_dictionary()
    datapipe_object.run_basic_analysis()
    datapipe_object.filter_samples(remove_all_false_annotations=True, storage_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/filtered_metadata.json")
    unique_concepts = datapipe_object.extract_concepts(storage_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/concepts.json")
    datapipe_object.build_caption(storage_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/captioned_metadata.json")
    datapipe_object.build_concept_vector(unique_concepts=unique_concepts, storage_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/concept_metadata.json")
