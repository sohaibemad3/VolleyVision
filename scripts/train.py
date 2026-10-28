
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from training import Trainer, Evaluator
from dataset.volleyball_dataset import load_dataset
from dataset.baseline_dataset_preparation import BaselinePreparation
from dataset import transforms

from models import (
    Baseline1,
    Baseline4,
    Baseline5,
    Baseline6,
    Baseline7,
    Baseline8,
)


# ---------------------------------------------------------
# Registries
# ---------------------------------------------------------

MODEL_REGISTRY = {
    "Baseline1": Baseline1,
    "Baseline4": Baseline4,
    "Baseline5": Baseline5,
    "Baseline6": Baseline6,
    "Baseline7": Baseline7,
    "Baseline8": Baseline8,
}

CRITERION_REGISTRY = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
}

OPTIMIZER_REGISTRY = {
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}

SCHEDULER_REGISTRY = {
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}

TRANSFORM_REGISTRY = {
    1: transforms.load_image_classification_transforms,
    2: transforms.load_cropped_player_transforms,
    3: transforms.load_image_sequence_transforms,
    4: transforms.load_cropped_player_sequence_transforms,
}


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration."""
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a dictionary.")

    return config


def set_seed(seed: int | None) -> None:
    """Set random seeds when a seed is provided."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: dict[str, Any]) -> torch.device:
    requested_device = config.get("training", {}).get("device", "auto")

    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")

    return torch.device(requested_device)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def build_model(config: dict[str, Any]) -> nn.Module:
    model_config = config["model"].copy()

    model_name = model_config.pop("name")

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )

    model_class = MODEL_REGISTRY[model_name]

    try:
        return model_class(**model_config)
    except TypeError as exc:
        raise TypeError(
            f"Could not initialize {model_name} with parameters "
            f"{model_config}. Check the model constructor and YAML."
        ) from exc


# ---------------------------------------------------------
# Transforms
# ---------------------------------------------------------

def build_transform(transform_config: Any):
    """
    Supported configuration:
      - Integer choices: 1, 2, 3, 4
      - Name of a no-argument transform factory in dataset.transforms
    """
    if transform_config is None:
        return None

    if isinstance(transform_config, int):
        if transform_config not in TRANSFORM_REGISTRY:
            raise ValueError(
                f"Unknown transform choice: {transform_config}"
            )

        return TRANSFORM_REGISTRY[transform_config]()

    if isinstance(transform_config, str):
        factory = getattr(transforms, transform_config, None)

        if factory is None or not callable(factory):
            raise ValueError(
                f"Transform factory '{transform_config}' was not found "
                "in dataset.transforms."
            )

        return factory()

    raise TypeError(
        "A transform must be configured using an integer choice "
        "or a transform factory name."
    )


# ---------------------------------------------------------
# Dataset preparation and DataLoaders
# ---------------------------------------------------------

def build_dataloaders(config: dict[str, Any]):
    data_config = config["data"]
    loader_config = data_config.get("loader", {})

    videos_root = data_config["videos_root"]
    videos_annot = data_config["videos_annot"]

    preparation = BaselinePreparation(
        videos_root=videos_root,
        videos_annot=videos_annot,
    )

    # Each YAML chooses the preparation method for its baseline.
    builder_name = data_config["dataset_builder"]
    builder = getattr(preparation, builder_name, None)

    if builder is None or not callable(builder):
        raise ValueError(
            f"Dataset builder '{builder_name}' does not exist "
            "on BaselinePreparation."
        )

    train_ids = data_config["train_video_ids"]
    val_ids = data_config["val_video_ids"]
    test_ids = data_config.get("test_video_ids", [])

    train_data = builder(train_ids)
    val_data = builder(val_ids)
    test_data = builder(test_ids) if test_ids else None

    train_transform = build_transform(
        data_config.get("train_transform")
    )
    eval_transform = build_transform(
        data_config.get("eval_transform")
    )

    choice_baseline = data_config.get("choice_baseline")
    batch_size = loader_config.get("batch_size", 2)
    num_workers = loader_config.get("num_workers", 4)
    pin_memory = loader_config.get(
        "pin_memory",
        torch.cuda.is_available(),
    )
    max_players = loader_config.get("max_players", 12)

    common_loader_args = {
        "choice_baseline": choice_baseline,
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "max_players": max_players,
    }

    train_loader = load_dataset(
        train_data,
        transform=train_transform,
        shuffle=True,
        **common_loader_args,
    )

    val_loader = load_dataset(
        val_data,
        transform=eval_transform,
        shuffle=False,
        **common_loader_args,
    )

    test_loader = None
    if test_data is not None:
        test_loader = load_dataset(
            test_data,
            transform=eval_transform,
            shuffle=False,
            **common_loader_args,
        )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------
# Loss, optimizer, and scheduler
# ---------------------------------------------------------

def build_criterion(config: dict[str, Any]) -> nn.Module:
    loss_config = config.get("loss", {"name": "CrossEntropyLoss"}).copy()
    loss_name = loss_config.pop("name")

    if loss_name not in CRITERION_REGISTRY:
        raise ValueError(f"Unsupported loss: {loss_name}")

    return CRITERION_REGISTRY[loss_name](**loss_config)


def build_optimizer(
    model: nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"].copy()
    optimizer_name = optimizer_config.pop("name")

    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Supports either spelling in the YAML.
    if "learning_rate" in optimizer_config:
        optimizer_config["lr"] = optimizer_config.pop("learning_rate")

    trainable_only = optimizer_config.pop(
        "trainable_parameters_only", True
    )

    parameters = model.parameters()

    if trainable_only:
        parameters = filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        )

    return OPTIMIZER_REGISTRY[optimizer_name](
        parameters,
        **optimizer_config,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
):
    scheduler_config = config.get("scheduler", {}).copy()

    if not scheduler_config:
        return None, None

    scheduler_name = scheduler_config.pop("name", None)

    if scheduler_name in (None, "None", "none"):
        return None, None

    if scheduler_name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    # These are Trainer settings, not constructor arguments.
    monitor = scheduler_config.pop("monitor", "val_loss")

    scheduler_class = SCHEDULER_REGISTRY[scheduler_name]
    scheduler = scheduler_class(optimizer, **scheduler_config)

    return scheduler, monitor


# ---------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------

def train(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(config.get("project", {}).get("seed"))
    device = get_device(config)

    model = build_model(config).to(device)

    train_loader, val_loader, test_loader = build_dataloaders(config)

    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scheduler, scheduler_monitor = build_scheduler(optimizer, config)

    training_config = config.get("training", {})
    data_config = config["data"]
    output_config = config.get("outputs", {})
    evaluation_config = config.get("evaluation", {})

    class_names = data_config.get("class_names", [])
    num_classes = config["model"].get(
        "num_classes",
        len(class_names) if class_names else 8,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        scheduler_monitor=scheduler_monitor,
        monitor_mode="auto",
        num_classes=num_classes,
        use_amp=training_config.get("use_amp", True),
        early_stopping_patience=training_config.get(
            "early_stopping_patience"
        ),
        forward_fn=None,
        grad_clip_norm=training_config.get("grad_clip_norm"),
        class_names=class_names or None,
    )

    checkpoint_path = output_config.get(
        "checkpoint_path",
        "checkpoints/best_val_acc.pth",
    )

    Path(checkpoint_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config.get("epochs", 10),
        checkpoint_path=checkpoint_path,
        print_confusion_matrix=output_config.get(
            "print_training_confusion_matrix",
            False,
        ),
    )

    test_results = None

    if test_loader is not None:
        test_results_dir = output_config.get(
            "test_results_dir",
            "results/test",
        )
        Path(test_results_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

        evaluator = Evaluator(
            model=model,
            criterion=criterion,
            device=device,
            use_amp=training_config.get("use_amp", True),
            class_names=class_names or None,
        )

        test_results = evaluator.evaluate(
            dataloader=test_loader,
            save_dir=test_results_dir,
            plot_matrix=evaluation_config.get(
                "plot_confusion_matrix",
                True,
            ),
            print_results=evaluation_config.get(
                "print_results",
                True,
            ),
        )

    return {
        "history": history,
        "test_results": test_results,
        "checkpoint_path": checkpoint_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a volleyball activity recognition baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the baseline YAML configuration.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    results = train(config)
    print(f"Training finished. Checkpoint: {results['checkpoint_path']}")


if __name__ == "__main__":
    main()
