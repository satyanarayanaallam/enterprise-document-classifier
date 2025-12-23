"""Text encoder based on Transformers."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Tuple


class TextEncoder(nn.Module):
    """Text encoder using HuggingFace transformers."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 768,
        max_length: int = 512,
        pretrained: bool = True,
    ):
        """Initialize text encoder.

        Args:
            model_name: HuggingFace model name
            embedding_dim: Output embedding dimension
            max_length: Maximum sequence length
            pretrained: Use pretrained weights
        """
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(model_name)
        except Exception as e:
            # If download fails, try with local cache or raise informative error
            print(f"Warning: Could not load {model_name} ({type(e).__name__})")
            print("Attempting to use cached version or offline mode...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.transformer = AutoModel.from_pretrained(model_name, local_files_only=True)
            except:
                raise RuntimeError(
                    f"Could not load {model_name}. "
                    "Please check your internet connection or download the model manually."
                )

        # Get actual embedding dimension from model
        model_dim = self.transformer.config.hidden_size

        # Projection head if needed
        if model_dim != embedding_dim:
            self.projection = nn.Linear(model_dim, embedding_dim)
        else:
            self.projection = None

    def tokenize(
        self,
        texts: list,
        return_tensors: str = "pt",
    ) -> dict:
        """Tokenize texts.

        Args:
            texts: List of text strings
            return_tensors: Return format

        Returns:
            Tokenized tensors
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors=return_tensors,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Tokenized input (B, L)
            attention_mask: Attention mask (B, L)

        Returns:
            Embeddings (B, embedding_dim)
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Use [CLS] token representation
        embeddings = outputs.last_hidden_state[:, 0, :]

        if self.projection is not None:
            embeddings = self.projection(embeddings)

        return embeddings

    def encode_texts(self, texts: list) -> torch.Tensor:
        """Encode a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            Embeddings (B, embedding_dim)
        """
        tokenized = self.tokenize(texts)
        with torch.no_grad():
            embeddings = self.forward(
                tokenized["input_ids"],
                tokenized["attention_mask"],
            )
        return embeddings


class TextClassifier(nn.Module):
    """Text classifier based on transformer."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 768,
        max_length: int = 512,
    ):
        """Initialize text classifier.

        Args:
            num_classes: Number of classes
            model_name: HuggingFace model name
            embedding_dim: Embedding dimension
            max_length: Maximum sequence length
        """
        super().__init__()

        self.encoder = TextEncoder(
            model_name=model_name,
            embedding_dim=embedding_dim,
            max_length=max_length,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Tokenized input (B, L)
            attention_mask: Attention mask (B, L)

        Returns:
            Logits (B, num_classes)
        """
        embeddings = self.encoder(input_ids, attention_mask)
        logits = self.classifier(embeddings)
        return logits

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get embeddings without classification head."""
        return self.encoder(input_ids, attention_mask)
