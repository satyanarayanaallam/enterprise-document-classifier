"""
Model Training Notebook

This notebook demonstrates:
1. Creating and initializing models
2. Setting up training pipeline
3. Training image classifier
4. Monitoring training progress
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.models import ImageClassifier
from src.data import DocumentDataset, collate_fn_classification
from src.training import Trainer
from src.utils import Config, setup_logger

logger = setup_logger(__name__)


def main():
    """Run training pipeline."""
    # Initialize config
    config = Config()
    logger.info(f"Config: {config}")

    # Create model
    model = ImageClassifier(
        num_classes=10,
        embedding_dim=config.embedding_dim,
        pretrained=True,
    )
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    try:
        # Load dataset
        dataset = DocumentDataset(
            metadata_dir="data/processed/metadata",
            image_dir="data/processed/images",
            image_size=config.image_size,
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
            num_workers=0,  # Set to 0 for notebooks
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_fn_classification,
            num_workers=0,
        )

        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=config.device,
            checkpoint_dir=config.model_checkpoint_dir,
        )

        # Train
        logger.info("Starting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,  # Short training for demo
        )

        logger.info("Training complete!")
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")

    except FileNotFoundError:
        logger.warning("Dataset not found. Please prepare data first.")


if __name__ == "__main__":
    main()
