import sys
import os
import torch
from pathlib import Path
import torchvision.transforms as transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pipeline import finetune_dataloader
from models import finetune_siglip
from config.finetune_config import CONFIG


if __name__ == "__main__":

    print("||Starting SigLIP fine-tuning||")

    # define static variables:
    model = CONFIG["model"]
    checkpoint_path = CONFIG["checkpoint_path"]
    image_storage_path = CONFIG["image_storage_path"]

    # -----------------------------------------------------------------------------------
    # define transform
    transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    # -----------------------------------------------------------------------------------
    # Get training data loader
    training_data = finetune_dataloader.FineTuneLoader(
        data_path=CONFIG["data_path"],
        concept_path=CONFIG["concept_path"],
        object_path=CONFIG["object_path"],
        predicate_path=CONFIG["predicate_path"],
        probe_type="concept",
        transform=transform,
        split="train"
    )
    train_loader = training_data.data_loaders(split="train", batch_size=4)
    positive_count = torch.zeros(CONFIG["num_concepts"])
    for sample in training_data.samples:
        positive_count += sample["concept_vector"]
    negative_count = len(training_data) - positive_count
    concept_pos_weight = torch.where((positive_count > 0) & (negative_count > 0), negative_count / positive_count.clamp_min(1), torch.ones_like(positive_count)).clamp(max=20.0)

    # -----------------------------------------------------------------------------------
    # Get validation data loader
    validation_data = finetune_dataloader.FineTuneLoader(
        data_path=CONFIG["data_path"],
        concept_path=CONFIG["concept_path"],
        object_path=CONFIG["object_path"],
        predicate_path=CONFIG["predicate_path"],
        probe_type="concept",
        transform=transform,
        split="valid"
    )
    validation_loader = validation_data.data_loaders(split="valid", batch_size=4)

    # -----------------------------------------------------------------------------------
    # Get testing data loader
    testing_data = finetune_dataloader.FineTuneLoader(
        data_path=CONFIG["data_path"],
        concept_path=CONFIG["concept_path"],
        object_path=CONFIG["object_path"],
        predicate_path=CONFIG["predicate_path"],
        probe_type="concept",
        transform=transform,
        split="test"
    )
    test_loader = testing_data.data_loaders(split="test", batch_size=4)

    # -----------------------------------------------------------------------------------
    # Create object of the fine-tuning model and train
    fine_tuner = finetune_siglip.FineTuneSiglip(
        model=model,
        siglip_model=CONFIG["siglip_model"],
        train_loader=train_loader,
        val_loader=validation_loader,
        test_loader=test_loader,
        num_epochs=CONFIG["num_epochs"],
        concept_pos_weight=concept_pos_weight,
        require_concept_bottleneck=True,
        require_object_bottleneck=False,
        require_predicate_bottleneck=False,
        num_concepts=CONFIG["num_concepts"],
        num_objects=CONFIG["num_objects"],
        num_predicates=CONFIG["num_predicates"],
        enable_lora=False,
        use_qlora=False
    )

    checkpoint_dir = Path(checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "best_finetuned_siglip.pt"
    plot_file = Path(image_storage_path) / "finetuning_lambda_curves.png"

    results, best_lambdas = fine_tuner.hyperparameter_search(lambda_values={"concept": [0.1, 0.5, 1.0]}, learning_rate=1e-5, patience=3, save_path=str(checkpoint_file), plot_path=str(plot_file))
    test_metrics = fine_tuner.evaluate_best_model(str(checkpoint_file))
    print(f"Best lambdas: {best_lambdas}")
    print(f"Test metrics: {test_metrics}")
