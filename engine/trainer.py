
import csv
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


from .batch_utils import unpack_batch
from .evaluator import Evaluator, _forward
from .metrics import ClassificationMetrics, plot_history


class Trainer:
    """Trainer with validation, checkpointing, early stopping, and history."""

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        num_classes=8,
        scheduler=None,
        scheduler_monitor="val_loss",
        monitor_mode="auto",
        use_amp=False,
        early_stopping_patience=None,
        forward_fn=None,
        grad_clip_norm=None,
        class_names=None,
    ):
        if scheduler_monitor not in {"val_loss", "val_accuracy"}:
            raise ValueError(
                "scheduler_monitor must be 'val_loss' or 'val_accuracy'."
            )

        if monitor_mode == "auto":
            monitor_mode = (
                "min" if scheduler_monitor == "val_loss" else "max"
            )

        if monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'auto', 'min', or 'max'.")

        if (
            early_stopping_patience is not None
            and early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be >= 1 or None.")

        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive or None.")

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.scheduler = scheduler
        self.scheduler_monitor = scheduler_monitor
        self.monitor_mode = monitor_mode
        self.use_amp = use_amp and self.device.type == "cuda"
        self.early_stopping_patience = early_stopping_patience
        self.forward_fn = forward_fn
        self.grad_clip_norm = grad_clip_norm

        self.class_names = (
            class_names
            if class_names is not None
            else [str(i) for i in range(num_classes)]
        )

        if len(self.class_names) != num_classes:
            raise ValueError("class_names length must equal num_classes.")

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp
        )

        self.evaluator = Evaluator(
            model=self.model,
            criterion=self.criterion,
            device=self.device,
            num_classes=self.num_classes,
            use_amp=self.use_amp,
            forward_fn=self.forward_fn,
            class_names=self.class_names,
        )

    def train_one_epoch(self, train_loader):
        self.model.train()
        metrics = ClassificationMetrics(self.num_classes)

        for batch in train_loader:
            inputs, labels, player_mask = unpack_batch(batch)

            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(
                self.device, non_blocking=True
            ).long().reshape(-1)

            if player_mask is not None:
                player_mask = player_mask.to(
                    self.device, non_blocking=True
                )

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = _forward(
                        self.model, inputs, player_mask, self.forward_fn
                    )
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()

                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                logits = _forward(
                    self.model, inputs, player_mask, self.forward_fn
                )
                loss = self.criterion(logits, labels)
                loss.backward()

                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm,
                    )

                self.optimizer.step()

            metrics.update(loss.detach(), logits.detach(), labels)

        return metrics.compute()

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        checkpoint_path="checkpoints/best_model.pth",
        results_dir="results",
        print_confusion_matrix=False,
    ):
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        self.model.to(self.device)

        checkpoint_path = Path(checkpoint_path)
        results_dir = Path(results_dir)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        best_value = (
            float("inf")
            if self.monitor_mode == "min"
            else -float("inf")
        )
        best_epoch = 0
        no_improvement = 0
        history = []

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(train_loader)

            val_metrics = self.evaluator.evaluate(
                val_loader,
                split_name="validation",
                print_results=False,
                plot_matrix=False,
            )

            # FIX: evaluator returns "loss" and "accuracy", not
            # "val_loss" and "val_accuracy".
            metric_key = self.scheduler_monitor.removeprefix("val_")
            monitored = val_metrics[metric_key]

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(monitored)
            elif self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history.append(row)

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {row['train_loss']:.4f} | "
                f"Train Acc: {row['train_accuracy']:.2f}% | "
                f"Val Loss: {row['val_loss']:.4f} | "
                f"Val Acc: {row['val_accuracy']:.2f}%"
            )

            if self.monitor_mode == "min":
                improved = monitored < best_value
            else:
                improved = monitored > best_value

            if improved:
                best_value = monitored
                best_epoch = epoch
                no_improvement = 0

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_metric": best_value,
                    "monitor": self.scheduler_monitor,
                    "monitor_mode": self.monitor_mode,
                    "train_metrics": {
                        "loss": train_metrics["loss"],
                        "accuracy": train_metrics["accuracy"],
                    },
                    "val_metrics": {
                        "loss": val_metrics["loss"],
                        "accuracy": val_metrics["accuracy"],
                    },
                }

                if self.scheduler is not None:
                    checkpoint["scheduler_state_dict"] = (
                        self.scheduler.state_dict()
                    )

                if self.use_amp:
                    checkpoint["scaler_state_dict"] = (
                        self.scaler.state_dict()
                    )

                torch.save(checkpoint, checkpoint_path)
                print(f"Saved best checkpoint: {checkpoint_path}")

            else:
                no_improvement += 1

            if (
                self.early_stopping_patience is not None
                and no_improvement >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best epoch was {best_epoch}."
                )
                break

        if best_epoch == 0 or not checkpoint_path.exists():
            raise RuntimeError(
                "No best checkpoint was saved. Check the training and "
                "validation loaders and metric values."
            )

        self._save_history(history, results_dir / "history.csv")
        plot_history(history, results_dir)

        print(f"\nTraining finished. Best epoch: {best_epoch}")
        print(f"Best {self.scheduler_monitor}: {best_value:.4f}")

        # Load the best checkpoint saved during this run.
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])

        final_val = self.evaluator.evaluate(
            val_loader,
            split_name="best_validation",
            save_dir=results_dir,
            print_results=print_confusion_matrix,
            plot_matrix=True,
        )

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_metric": best_value,
            "checkpoint_path": str(checkpoint_path),
            "best_validation_metrics": final_val,
        }

    @staticmethod
    def _save_history(history, path):
        if not history:
            return

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(history[0].keys()),
            )
            writer.writeheader()
            writer.writerows(history)

