"""Image encoder model based on ResNet50."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ImageEncoder(nn.Module):
    """ResNet50-based image encoder with projection head."""

    def __init__(
        self,
        embedding_dim: int = 768,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """Initialize image encoder.

        Args:
            embedding_dim: Dimension of output embeddings
            pretrained: Use pretrained weights
            freeze_backbone: Freeze backbone parameters
        """
        super().__init__()

        # Load pretrained ResNet50
        try:
            resnet50 = models.resnet50(pretrained=pretrained)
        except Exception as e:
            # If download fails (SSL/network issues), use non-pretrained
            print(f"Warning: Could not load pretrained model ({type(e).__name__}). Using non-pretrained.")
            resnet50 = models.resnet50(pretrained=False)

        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        backbone_output_dim = 2048

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor (B, 3, H, W)

        Returns:
            Embeddings (B, embedding_dim)
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        # Project to embedding space
        embeddings = self.projection(features)
        return embeddings


class ImageClassifier(nn.Module):
    """Image classifier with ResNet50 backbone."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 768,
        pretrained: bool = True,
    ):
        """Initialize classifier.

        Args:
            num_classes: Number of classification classes
            embedding_dim: Embedding dimension
            pretrained: Use pretrained weights
        """
        super().__init__()

        self.encoder = ImageEncoder(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor (B, 3, H, W)

        Returns:
            Logits (B, num_classes)
        """
        embeddings = self.encoder(x)
        logits = self.classifier(embeddings)
        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings without classification head."""
        return self.encoder(x)
