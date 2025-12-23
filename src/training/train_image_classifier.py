"""Training script for image classifier."""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import ImageClassifier
from src.data import DocumentDataset, collate_fn_classification
from src.training import Trainer
from src.utils import Config, setup_logger

logger = setup_logger(__name__)


def main(args):
    """Main training function."""
    # Load config
    config = Config()
    config.device = args.device
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs

    logger.info(f"Config: {config}")

    # Create model
    model = ImageClassifier(
        num_classes=10,
        embedding_dim=config.embedding_dim,
        pretrained=True,
    )
    logger.info(f"Created model: {model}")

    # Create dataset
    dataset = DocumentDataset(
        metadata_dir=args.metadata_dir,
        image_dir=args.image_dir,
        image_size=config.image_size,
        max_text_length=config.max_text_length,
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn_classification,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn_classification,
        num_workers=config.num_workers,
    )

    logger.info(f"Train dataset size: {train_size}, Val dataset size: {val_size}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=config.device,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        checkpoint_dir=config.model_checkpoint_dir,
    )

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
    )

    logger.info(f"Training complete. Best val loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--metadata-dir", type=str, required=True, help="Metadata directory")
    parser.add_argument("--image-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")

    args = parser.parse_args()
    main(args)
