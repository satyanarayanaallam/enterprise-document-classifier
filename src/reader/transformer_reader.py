"""Transformer reader for extractive QA."""

import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from typing import Dict, List, Tuple


class TransformerReader(nn.Module):
    """Transformer-based reader for extractive QA."""

    def __init__(
        self,
        model_name: str = "deepset/roberta-base-squad2",
        max_seq_length: int = 512,
    ):
        """Initialize reader.

        Args:
            model_name: HuggingFace model name (should be QA model)
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Input IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs

        Returns:
            Tuple of (start_logits, end_logits)
        """
        outputs = self.qa_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.start_logits, outputs.end_logits

    def predict(
        self,
        question: str,
        context: str,
    ) -> Dict:
        """Predict answer for question given context.

        Args:
            question: Question string
            context: Context/passage string

        Returns:
            Dictionary with prediction info
        """
        inputs = self.tokenizer.encode_plus(
            question,
            context,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
        )

        with torch.no_grad():
            outputs = self.qa_model(**inputs)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Get answer span
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)

        # Convert to tokens and text
        input_ids = inputs["input_ids"][0]
        answer_tokens = input_ids[start_idx : end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens)

        return {
            "answer": answer,
            "start_idx": start_idx.item(),
            "end_idx": end_idx.item(),
            "start_logits": start_logits.max().item(),
            "end_logits": end_logits.max().item(),
        }

    def batch_predict(
        self,
        questions: List[str],
        contexts: List[str],
    ) -> List[Dict]:
        """Batch predict answers.

        Args:
            questions: List of questions
            contexts: List of contexts

        Returns:
            List of predictions
        """
        predictions = []
        for question, context in zip(questions, contexts):
            pred = self.predict(question, context)
            predictions.append(pred)
        return predictions


class ReaderRanker(nn.Module):
    """Ranker for reader outputs."""

    def __init__(self, hidden_size: int = 768):
        """Initialize ranker.

        Args:
            hidden_size: Hidden size
        """
        super().__init__()
        self.ranker = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Rank answers based on logits.

        Args:
            start_logits: Start logits
            end_logits: End logits

        Returns:
            Ranking scores
        """
        combined = torch.cat([start_logits, end_logits], dim=-1)
        scores = self.ranker(combined)
        return scores
