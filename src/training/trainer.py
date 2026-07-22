from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_dir: str | Path = "outputs/checkpoints",
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")

    def train_one_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            signals = batch["signal"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(signals)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            batch_size = signals.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    @torch.no_grad()
    def validate_one_epoch(self, val_loader: DataLoader) -> float:
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for batch in val_loader:
            signals = batch["signal"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(signals)
            loss = self.criterion(logits, labels)

            batch_size = signals.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_name: str = "best_model.pt",
        early_stopping_patience: int | None = None,
    ) -> Dict[str, list]:
        history = {
            "train_loss": [],
            "val_loss": [],
        }

        epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate_one_epoch(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            print(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            # if val_loss < self.best_val_loss:
            #     self.best_val_loss = val_loss
            #     self.save_checkpoint(checkpoint_name)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_checkpoint(checkpoint_name)
            else:
                epochs_without_improvement += 1

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best val loss: {self.best_val_loss:.4f}"
                )
                break

        return history

    def save_checkpoint(self, checkpoint_name: str) -> None:
        checkpoint_path = self.save_dir / checkpoint_name

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            checkpoint_path,
        )

        print(f"Saved best model to: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_name: str) -> None:
        checkpoint_path = self.save_dir / checkpoint_name

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Loaded checkpoint from: {checkpoint_path}")