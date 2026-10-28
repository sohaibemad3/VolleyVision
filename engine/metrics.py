
from pathlib import Path

import torch
import matplotlib.pyplot as plt


class ClassificationMetrics:
    """Accumulate mean-reduced loss, accuracy, and confusion matrix."""

    def __init__(self, num_classes):
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.num_samples = 0
        self.num_correct = 0
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
        )

    @torch.no_grad()
    def update(self, loss, logits, labels):
        labels = labels.detach().reshape(-1).long()
        logits = logits.detach()

        if logits.ndim != 2:
            raise ValueError(
                f"Expected logits shape [batch_size, num_classes], "
                f"got {tuple(logits.shape)}."
            )

        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} output classes, "
                f"got {logits.shape[1]}."
            )

        predictions = logits.argmax(dim=1).reshape(-1)
        batch_size = labels.numel()

        if batch_size == 0:
            return

        if predictions.numel() != batch_size:
            raise ValueError(
                f"Prediction count ({predictions.numel()}) does not match "
                f"label count ({batch_size})."
            )

        if torch.any((labels < 0) | (labels >= self.num_classes)):
            raise ValueError(
                f"Labels must be in [0, {self.num_classes - 1}]."
            )

        if loss.numel() != 1:
            raise ValueError(
                "Expected a scalar batch loss. Use a criterion with "
                "reduction='mean' or a scalar mean-reduced loss."
            )

        # This assumes the criterion returns mean loss for the batch.
        self.loss_sum += float(loss.detach().item()) * batch_size
        self.num_samples += batch_size
        self.num_correct += int((predictions == labels).sum().item())

        encoded = labels.cpu() * self.num_classes + predictions.cpu()
        counts = torch.bincount(
            encoded,
            minlength=self.num_classes ** 2,
        )

        self.confusion_matrix += counts.reshape(
            self.num_classes, self.num_classes
        )

    def compute(self):
        if self.num_samples == 0:
            raise ValueError(
                "Cannot compute metrics: the dataloader produced no samples."
            )

        return {
            "loss": self.loss_sum / self.num_samples,
            "accuracy": 100.0 * self.num_correct / self.num_samples,
            "confusion_matrix": self.confusion_matrix.clone(),
            "num_samples": self.num_samples,
        }


def plot_history(history, save_dir, show=False):
    """Save training and validation loss/accuracy curves."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not history:
        raise ValueError("history is empty; there is nothing to plot.")

    epochs = [row["epoch"] for row in history]

    plots = [
        ("loss", "Loss", "loss_curves.png",
         "Training and Validation Loss"),
        ("accuracy", "Accuracy (%)", "accuracy_curves.png",
         "Training and Validation Accuracy"),
    ]

    for metric, ylabel, filename, title in plots:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            epochs,
            [row[f"train_{metric}"] for row in history],
            marker="o",
            label="Train",
        )
        ax.plot(
            epochs,
            [row[f"val_{metric}"] for row in history],
            marker="o",
            label="Validation",
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / filename, dpi=160)

        if show:
            plt.show()

        plt.close(fig)


def plot_confusion_matrix(
    matrix, class_names, save_path=None, normalize=False, show=False
):
    """Plot a confusion matrix: rows=true, columns=predicted."""
    matrix = torch.as_tensor(matrix).detach().cpu().numpy()

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be a square 2D matrix.")

    if len(class_names) != matrix.shape[0]:
        raise ValueError(
            f"Expected {matrix.shape[0]} class names, "
            f"got {len(class_names)}."
        )

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix.astype(float) / row_sums.clip(min=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    positions = range(len(class_names))
    ax.set(
        xticks=list(positions),
        yticks=list(positions),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Normalized Confusion Matrix"
        if normalize else "Confusion Matrix",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = float(matrix.max()) / 2 if matrix.size else 0

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            label = f"{value:.2f}" if normalize else f"{int(value)}"

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

