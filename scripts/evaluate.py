

"""
Evaluate a trained volleyball baseline using the project's existing Evaluator.

Run from the repository root:
    python -m scripts.evaluate --config configs/baseline1.yaml

Optional:
    python -m scripts.evaluate --config configs/baseline1.yaml --checkpoint checkpoints/baseline1/best_val_acc.pth

This script reuses the helpers and Evaluator API shown in scripts/train.py.
It loads the configured test split, restores saved model weights, and evaluates
the test set. It does not train the model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from scripts.train import (
    build_criterion,
    build_dataloaders,
    build_model,
    get_device,
    load_config,
)
from training import Evaluator


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first or pass the correct path with --checkpoint."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint).__name__}")

    # Supports raw state_dict, {'model_state_dict': ...}, or {'state_dict': ...}.
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise TypeError("Could not find a model state_dict in the checkpoint.")

    # Remove DataParallel prefix if present.
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained volleyball baseline.")
    parser.add_argument("--config", type=Path, required=True, help="Path to baseline YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint override; defaults to outputs.checkpoint_path in YAML.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional results directory.")
    args = parser.parse_args()

    config: dict[str, Any] = load_config(args.config)
    device = get_device(config)

    output_config = config.get("outputs", {})
    checkpoint_path = args.checkpoint or Path(
        output_config.get("checkpoint_path", "checkpoints/best_val_acc.pth")
    )
    output_dir = args.output_dir or Path(
        output_config.get("test_results_dir", "results/test")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config).to(device)
    load_model_checkpoint(model, checkpoint_path, device)

    _, _, test_loader = build_dataloaders(config)
    if test_loader is None:
        raise RuntimeError("No test_loader was created. Check data.test_video_ids in the YAML.")

    class_names = config.get("data", {}).get("class_names", [])
    training_config = config.get("training", {})
    evaluation_config = config.get("evaluation", {})

    criterion = build_criterion(config)
    evaluator = Evaluator(
        model=model,
        criterion=criterion,
        device=device,
        use_amp=training_config.get("use_amp", True),
        class_names=class_names or None,
    )

    results = evaluator.evaluate(
        dataloader=test_loader,
        save_dir=str(output_dir),
        plot_matrix=evaluation_config.get("plot_confusion_matrix", True),
        print_results=evaluation_config.get("print_results", True),
    )

    print(f"Evaluation finished. Checkpoint: {checkpoint_path}")
    print(f"Test results saved in: {output_dir}")
    if results is not None:
        print("Evaluator returned results.")


if __name__ == "__main__":
    main()