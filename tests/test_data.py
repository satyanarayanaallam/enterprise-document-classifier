"""Unit tests for data module."""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from src.data import DocumentDataset


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return {
        "image_path": "sample.png",
        "text": "This is a sample document",
        "label": "invoice",
    }


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_document_dataset(temp_data_dir, sample_metadata):
    """Test DocumentDataset loading."""
    # Create sample metadata file
    metadata_dir = Path(temp_data_dir) / "metadata"
    image_dir = Path(temp_data_dir) / "images"
    metadata_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    metadata_file = metadata_dir / "sample.json"
    with open(metadata_file, "w") as f:
        json.dump(sample_metadata, f)

    # Create sample image
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np

    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(image_dir / "sample.png")

    # Test dataset
    dataset = DocumentDataset(
        metadata_dir=str(metadata_dir),
        image_dir=str(image_dir),
        image_size=224,
    )

    assert len(dataset) == 1
    image, text, label_idx = dataset[0]
    assert image.shape == (3, 224, 224)
    assert isinstance(text, str)
    assert isinstance(label_idx, int)


if __name__ == "__main__":
    pytest.main([__file__])
