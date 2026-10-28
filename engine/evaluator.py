
from contextlib import nullcontext
from pathlib import Path

import torch

# Import these from your project modules when using separate files:
from .batch_utils import unpack_batch
from .metrics import ClassificationMetrics, plot_confusion_matrix


def _forward(model, inputs, player_mask=None, forward_fn=None):
    if forward_fn is not None:
        return forward_fn(model, inputs, player_mask)

    if player_mask is not None:
        return model(inputs, player_mask)

    return model(inputs)


class Evaluator:
    """Evaluate a classification model."""

    def __init__(
        self,
        model,
        criterion,
        device,
        num_classes=8,
        use_amp=False,
        forward_fn=None,
        class_names=None,
    ):
        self.model = model
        self.criterion = criterion
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.use_amp = use_amp and self.device.type == "cuda"
        self.forward_fn = forward_fn
        self.class_names = (
            class_names
            if class_names is not None
            else [str(i) for i in range(num_classes)]
        )

        if len(self.class_names) != num_classes:
            raise ValueError("class_names length must equal num_classes.")

    def evaluate(
        self,
        dataloader,
        split_name="evaluation",
        save_dir=None,
        print_results=True,
        plot_matrix=True,
        normalize_matrix=False,
    ):
        self.model.to(self.device)
        self.model.eval()
        metrics = ClassificationMetrics(self.num_classes)

        with torch.inference_mode():
            for batch in dataloader:
                inputs, labels, player_mask = unpack_batch(batch)

                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(
                    self.device, non_blocking=True
                ).long().reshape(-1)

                if player_mask is not None:
                    player_mask = player_mask.to(
                        self.device, non_blocking=True
                    )

                amp_context = (
                    torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    )
                    if self.use_amp else nullcontext()
                )

                with amp_context:
                    logits = _forward(
                        self.model, inputs, player_mask, self.forward_fn
                    )
                    loss = self.criterion(logits, labels)

                metrics.update(loss, logits, labels)

        results = metrics.compute()

        if print_results:
            print(f"\n{split_name.title()} Results")
            print(f"Loss: {results['loss']:.4f}")
            print(f"Accuracy: {results['accuracy']:.2f}%")
            print("Confusion Matrix (rows=true, columns=predicted):")
            print(results["confusion_matrix"])

        if save_dir is not None and plot_matrix:
            matrix_path = (
                Path(save_dir) / f"{split_name}_confusion_matrix.png"
            )
            plot_confusion_matrix(
                results["confusion_matrix"],
                self.class_names,
                save_path=matrix_path,
                normalize=normalize_matrix,
            )

        return results

