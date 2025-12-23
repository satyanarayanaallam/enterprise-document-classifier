"""FAISS retriever for dense retrieval."""

import os
import numpy as np
import torch
from typing import List, Tuple, Optional

try:
    import faiss
except ImportError:
    faiss = None


class FAISSRetriever:
    """FAISS-based dense retriever."""

    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "flat",
        device: str = "cpu",
    ):
        """Initialize FAISS retriever.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('flat' or 'ivf')
            device: Device for index ('cpu' or 'gpu')
        """
        if faiss is None:
            raise ImportError("faiss not installed. Install faiss-cpu or faiss-gpu.")

        self.embedding_dim = embedding_dim
        self.device = device
        self.index = None
        self.embeddings = None
        self.metadata = []

        if index_type == "flat":
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        if device == "gpu" and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[dict]] = None,
    ) -> None:
        """Add embeddings to index.

        Args:
            embeddings: Array of embeddings (N, D)
            metadata: Optional metadata for each embedding
        """
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        if metadata:
            self.metadata.extend(metadata)

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors.

        Args:
            query_embeddings: Query embeddings (Q, D)
            k: Number of results to return

        Returns:
            Tuple of (distances, indices)
        """
        query_embeddings = query_embeddings.astype(np.float32)
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices

    def get_metadata(self, indices: np.ndarray) -> List[List[dict]]:
        """Get metadata for retrieved indices.

        Args:
            indices: Array of indices

        Returns:
            List of metadata lists
        """
        results = []
        for idx_list in indices:
            result = [self.metadata[i] for i in idx_list if i < len(self.metadata)]
            results.append(result)
        return results

    def save(self, path: str) -> None:
        """Save index to disk.

        Args:
            path: Path to save index
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save index
        if isinstance(self.index, faiss.Index):
            if self.device == "gpu":
                index_cpu = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(index_cpu, path)
            else:
                faiss.write_index(self.index, path)

        # Save metadata
        metadata_path = path.replace(".index", ".metadata.npy")
        if self.metadata:
            np.save(metadata_path, self.metadata, allow_pickle=True)

    def load(self, path: str) -> None:
        """Load index from disk.

        Args:
            path: Path to load index from
        """
        # Load index
        index = faiss.read_index(path)
        if self.device == "gpu" and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            self.index = index

        # Load metadata
        metadata_path = path.replace(".index", ".metadata.npy")
        if os.path.exists(metadata_path):
            self.metadata = list(np.load(metadata_path, allow_pickle=True))


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""

    def __init__(self, cache_dir: str = "cache"):
        """Initialize cache.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def save_embeddings(
        self,
        name: str,
        embeddings: torch.Tensor,
    ) -> None:
        """Save embeddings to cache.

        Args:
            name: Cache name
            embeddings: Embedding tensor
        """
        path = os.path.join(self.cache_dir, f"{name}.pt")
        torch.save(embeddings, path)

    def load_embeddings(self, name: str) -> Optional[torch.Tensor]:
        """Load embeddings from cache.

        Args:
            name: Cache name

        Returns:
            Embeddings or None if not found
        """
        path = os.path.join(self.cache_dir, f"{name}.pt")
        if os.path.exists(path):
            return torch.load(path)
        return None

    def clear(self) -> None:
        """Clear all cached embeddings."""
        import shutil
        shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)
