from pipeline import datapipe


if __name__ == "__main__":
    datapipe_object = datapipe.DataPipeline(
        annotations_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/annotations.json",
        annotation_distionary_path="/Users/tejasdhopavkar/Documents/MS/Saarland_University/Semester_3/MLU/Project/data/raw_metadata.json",
        images_path="None"
    )

    datapipe_object.annotations_to_dictionary()
    datapipe_object.run_basic_analysis()
    datapipe_object.filter_samples()
