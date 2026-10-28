
from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer=None,
    epoch=None,
    best_metric=None,
    scheduler=None,
    scaler=None,
    extra=None,
):
    """
    Save model and training state.

    Args:
        path: Destination checkpoint path.
        model: PyTorch model.
        optimizer: Optional optimizer.
        epoch: Current epoch.
        best_metric: Best validation metric so far.
        scheduler: Optional learning-rate scheduler.
        scaler: Optional AMP GradScaler.
        extra: Optional additional information (e.g. config).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device="cpu",
):
    """
    Load a checkpoint into a model and optionally restore
    optimizer, scheduler and AMP scaler states.

    Returns:
        checkpoint: The loaded checkpoint dictionary.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        state = checkpoint.get("optimizer_state_dict")
        if state is not None:
            optimizer.load_state_dict(state)

    if scheduler is not None:
        state = checkpoint.get("scheduler_state_dict")
        if state is not None:
            scheduler.load_state_dict(state)

    if scaler is not None:
        state = checkpoint.get("scaler_state_dict")
        if state is not None:
            scaler.load_state_dict(state)

    return checkpoint

