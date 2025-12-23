"""
Exploratory Data Analysis Notebook

This notebook demonstrates:
1. Loading and exploring document datasets
2. Visualizing document images and OCR text
3. Analyzing label distributions
4. Creating sample batches for training
"""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from src.data import DocumentDataset, collate_fn_classification
from src.utils import setup_logger

logger = setup_logger(__name__)


def main():
    """Run exploratory analysis."""
    # Initialize dataset (requires data to be prepared)
    try:
        dataset = DocumentDataset(
            metadata_dir="data/processed/metadata",
            image_dir="data/processed/images",
            image_size=224,
        )

        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Labels: {dataset.label_to_idx}")

        # Create dataloader for visualization
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn_classification,
        )

        # Get a sample batch
        images, texts, labels = next(iter(loader))
        logger.info(f"Batch images shape: {images.shape}")
        logger.info(f"Batch texts: {texts}")
        logger.info(f"Batch labels: {labels}")

    except FileNotFoundError:
        logger.warning("Dataset not found. Please prepare data first.")
        logger.info("Steps to prepare data:")
        logger.info("1. Download datasets (RVL-CDIP, FUNSD, DocVQA)")
        logger.info("2. Extract and place in data/raw/")
        logger.info("3. Run OCR: python -m src.data.ocr")
        logger.info("4. Create metadata files in data/processed/metadata/")


if __name__ == "__main__":
    main()
