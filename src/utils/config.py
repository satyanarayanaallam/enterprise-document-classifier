"""Configuration management."""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for the document classifier pipeline."""

    # Data paths
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    samples_dir: str = "data/samples"

    # Model paths
    model_checkpoint_dir: str = "experiments/checkpoints"
    faiss_index_dir: str = "experiments/faiss_indices"

    # Model architecture
    image_encoder_model: str = "resnet50"
    text_encoder_model: str = "distilbert-base-uncased"
    embedding_dim: int = 768
    image_size: int = 224
    max_text_length: int = 512

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    device: str = "cuda"
    num_workers: int = 4
    gradient_accumulation_steps: int = 1

    # Inference
    faiss_nprobe: int = 10
    num_retrieved_docs: int = 5
    inference_batch_size: int = 64

    # Logging
    log_level: str = "INFO"
    mlflow_tracking_uri: str = "http://localhost:5000"

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment variables."""
        return cls(
            device=os.getenv("DEVICE", "cuda"),
            batch_size=int(os.getenv("BATCH_SIZE", "32")),
            learning_rate=float(os.getenv("LEARNING_RATE", "1e-4")),
            num_epochs=int(os.getenv("NUM_EPOCHS", "10")),
        )
