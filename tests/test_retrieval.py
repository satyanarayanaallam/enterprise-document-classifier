"""Unit tests for retrieval."""

import pytest
import numpy as np
from src.retrieval import FAISSRetriever


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings."""
    return np.random.randn(100, 768).astype(np.float32)


def test_faiss_retriever(sample_embeddings):
    """Test FAISS retriever."""
    retriever = FAISSRetriever(embedding_dim=768, device="cpu")

    # Add embeddings
    metadata = [{"doc_id": i, "content": f"Doc {i}"} for i in range(100)]
    retriever.add_embeddings(sample_embeddings, metadata)

    # Search
    query = np.random.randn(1, 768).astype(np.float32)
    distances, indices = retriever.search(query, k=5)

    assert distances.shape == (1, 5)
    assert indices.shape == (1, 5)

    # Get metadata
    result_metadata = retriever.get_metadata(indices)
    assert len(result_metadata) == 1
    assert len(result_metadata[0]) == 5


if __name__ == "__main__":
    pytest.main([__file__])
