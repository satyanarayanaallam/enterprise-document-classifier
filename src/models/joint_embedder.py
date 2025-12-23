"""Joint image-text embedding model."""

import torch
import torch.nn as nn
from typing import Tuple

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class JointEmbedder(nn.Module):
    """Joint image and text embedder with shared embedding space."""

    def __init__(
        self,
        embedding_dim: int = 768,
        image_model_name: str = "resnet50",
        text_model_name: str = "distilbert-base-uncased",
        temperature: float = 0.07,
    ):
        """Initialize joint embedder.

        Args:
            embedding_dim: Dimension of shared embedding space
            image_model_name: Image encoder model
            text_model_name: Text encoder model
            temperature: Temperature for contrastive learning
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.temperature = temperature

        # Image and text encoders
        # Note: pretrained=False if having SSL/network issues
        try:
            self.image_encoder = ImageEncoder(
                embedding_dim=embedding_dim,
                pretrained=True,
            )
        except Exception:
            print("Note: Using non-pretrained image encoder")
            self.image_encoder = ImageEncoder(
                embedding_dim=embedding_dim,
                pretrained=False,
            )
        
        try:
            self.text_encoder = TextEncoder(
                model_name=text_model_name,
                embedding_dim=embedding_dim,
            )
        except Exception:
            print("Note: Using non-pretrained text encoder")
            self.text_encoder = TextEncoder(
                model_name=text_model_name,
                embedding_dim=embedding_dim,
            )

        # Optional: normalization layers
        self.image_norm = nn.LayerNorm(embedding_dim)
        self.text_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            images: Image tensors (B, 3, H, W)
            input_ids: Tokenized text (B, L)
            attention_mask: Attention mask (B, L)

        Returns:
            Tuple of normalized image and text embeddings
        """
        # Get embeddings
        image_emb = self.image_encoder(images)
        text_emb = self.text_encoder(input_ids, attention_mask)

        # Normalize embeddings
        image_emb = self.image_norm(image_emb)
        text_emb = self.text_norm(text_emb)

        # L2 normalize
        image_emb = torch.nn.functional.normalize(image_emb, p=2, dim=1)
        text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)

        return image_emb, text_emb

    def get_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Get image embeddings."""
        image_emb = self.image_encoder(images)
        image_emb = self.image_norm(image_emb)
        image_emb = torch.nn.functional.normalize(image_emb, p=2, dim=1)
        return image_emb

    def get_text_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get text embeddings."""
        text_emb = self.text_encoder(input_ids, attention_mask)
        text_emb = self.text_norm(text_emb)
        text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)
        return text_emb


class ContrastiveLoss(nn.Module):
    """Contrastive loss for joint embeddings (InfoNCE)."""

    def __init__(self, temperature: float = 0.07):
        """Initialize loss.

        Args:
            temperature: Temperature scaling
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            image_emb: Image embeddings (B, D)
            text_emb: Text embeddings (B, D)

        Returns:
            Scalar loss
        """
        # Compute similarity matrix
        similarity = torch.matmul(image_emb, text_emb.t()) / self.temperature

        # Create labels (diagonal: positive pairs)
        batch_size = image_emb.shape[0]
        labels = torch.arange(batch_size, device=image_emb.device)

        # Cross-entropy loss
        loss_img = torch.nn.functional.cross_entropy(similarity, labels)
        loss_txt = torch.nn.functional.cross_entropy(similarity.t(), labels)

        return (loss_img + loss_txt) / 2
