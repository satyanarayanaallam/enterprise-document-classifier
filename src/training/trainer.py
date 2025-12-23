"""Training utilities and loops."""

import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class Trainer:
    """Training loop manager."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        scheduler: Optional[LRScheduler] = None,
        gradient_accumulation_steps: int = 1,
        checkpoint_dir: str = "checkpoints",
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler
            gradient_accumulation_steps: Gradient accumulation steps
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Handle batch format from collate_fn
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, texts, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                # Forward pass for classification
                outputs = self.model(images)
            else:
                # Handle dict format
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(batch.get('images', batch.get('image')))
                labels = batch.get('label', batch.get('labels'))

            # Compute loss
            loss = self.criterion(outputs, labels)
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Optimizer step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            pbar.set_postfix({"loss": total_loss / num_batches})

        avg_loss = total_loss / num_batches
        if self.scheduler:
            self.scheduler.step()

        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Validate model.

        Args:
            val_loader: Validation data loader
            epoch: Epoch number

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        for batch in pbar:
            # Handle batch format from collate_fn
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, texts, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                # Forward pass for classification
                outputs = self.model(images)
            else:
                # Handle dict format
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(batch.get('images', batch.get('image')))
                labels = batch.get('label', batch.get('labels'))

            # Compute loss
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": total_loss / num_batches})

        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
    ) -> Dict:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs

        Returns:
            Dictionary with training history
        """
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["train_loss"])
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_metrics['train_loss']:.4f}")

            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader, epoch)
                history["val_loss"].append(val_metrics["val_loss"])
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_metrics['val_loss']:.4f}")

                # Save best model
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint(f"best_model.pt")
                    logger.info("Saved best model checkpoint")
            else:
                self.save_checkpoint(f"epoch_{epoch+1}.pt")

        return history

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint: {path}")


def get_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    """Create warmup scheduler."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda)
