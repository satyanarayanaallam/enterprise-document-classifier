"""Unit tests for models."""

import pytest
import torch
from src.models import ImageEncoder, TextEncoder, ImageClassifier


@pytest.fixture
def device():
    return "cpu"


def test_image_encoder(device):
    """Test ImageEncoder."""
    model = ImageEncoder(embedding_dim=768, pretrained=False).to(device)
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    embeddings = model(x)

    assert embeddings.shape == (batch_size, 768)


def test_image_classifier(device):
    """Test ImageClassifier."""
    model = ImageClassifier(num_classes=10, embedding_dim=768, pretrained=False).to(device)
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    logits = model(x)

    assert logits.shape == (batch_size, 10)

    embeddings = model.get_embeddings(x)
    assert embeddings.shape == (batch_size, 768)


def test_text_encoder(device):
    """Test TextEncoder."""
    model = TextEncoder(embedding_dim=768).to(device)
    texts = ["Sample document text", "Another document"]
    embeddings = model.encode_texts(texts)

    assert embeddings.shape == (2, 768)


if __name__ == "__main__":
    pytest.main([__file__])
