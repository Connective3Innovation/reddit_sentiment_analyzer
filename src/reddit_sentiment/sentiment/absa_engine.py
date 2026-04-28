# src/reddit_sentiment/sentiment/absa_engine.py
"""Aspect-Based Sentiment Analysis (ABSA) engine.

Scores sentiment TOWARD a specific entity/brand, not just overall text sentiment.

Default model: yangheng/deberta-v3-base-absa-v1.1
- Trained on SemEval ABSA tasks
- Input: text + aspect term → Output: sentiment toward that aspect
- 3-class: Negative, Neutral, Positive

Example:
    "I left Capital One for Chase and couldn't be happier"
    - Generic sentiment: POSITIVE (the text is happy)
    - ABSA toward "Capital One": NEGATIVE (user left them)
    - ABSA toward "Chase": POSITIVE (user is happy with them)
"""
from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Final, Sequence

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_LOGGER: Final = logging.getLogger(__name__)

# Model configuration
ABSA_MODEL_ID = os.getenv(
    "ABSA_MODEL_ID",
    "yangheng/deberta-v3-base-absa-v1.1"
)
MAX_LEN = int(os.getenv("ABSA_MAX_LENGTH", "256"))
BATCH_SIZE = int(os.getenv("ABSA_BATCH_SIZE", "16"))

# Prefer a writable cache path in Cloud Run
os.environ.setdefault("HF_HOME", "/tmp/hf")

# Label mapping for ABSA model
# yangheng model uses: Negative, Neutral, Positive
ABSA_LABEL_MAP = {
    "Negative": "NEGATIVE",
    "Neutral": "NEUTRAL",
    "Positive": "POSITIVE",
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
}


@lru_cache(maxsize=1)
def _get_absa_model():
    """Load ABSA model and tokenizer (cached)."""
    _LOGGER.info("ABSA_ENGINE: Loading model '%s' (max_len=%d, batch=%d)", ABSA_MODEL_ID, MAX_LEN, BATCH_SIZE)
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(ABSA_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _LOGGER.info("ABSA_ENGINE: Model loaded in %.2fs, device=%s", time.time() - start, device)
    return tokenizer, model, device


class AbsaEngine:
    """Aspect-Based Sentiment Analysis engine."""

    def __init__(self):
        # Lazy loading via _get_absa_model()
        pass

    def run(
        self,
        texts: Sequence[str],
        target_aspect: str,
    ) -> pd.DataFrame:
        """
        Score sentiment toward a specific aspect/brand in each text.

        Args:
            texts: List of text strings to analyze
            target_aspect: The entity/brand to measure sentiment toward
                           (e.g., "Capital One", "Chase", "customer service")

        Returns:
            DataFrame with columns:
            - sentiment: POSITIVE, NEGATIVE, or NEUTRAL (toward the aspect)
            - prob: confidence score (0-1)
            - brand_sentiment_score: -1 to +1 scale
        """
        n_texts = len(texts)
        _LOGGER.info("ABSA_ENGINE: Analyzing sentiment toward '%s' for %d texts", target_aspect, n_texts)
        start = time.time()

        tokenizer, model, device = _get_absa_model()

        # Clean inputs
        texts = [t if isinstance(t, str) and t.strip() else "" for t in texts]
        empty_count = sum(1 for t in texts if not t)
        if empty_count > 0:
            _LOGGER.debug("ABSA_ENGINE: %d/%d texts are empty", empty_count, n_texts)

        # Format input for ABSA: "[CLS] text [SEP] aspect [SEP]"
        # The model expects the aspect term appended to understand what to analyze
        formatted_inputs = [
            f"{text} [SEP] {target_aspect}" if text else f"no content [SEP] {target_aspect}"
            for text in texts
        ]

        results = []
        label_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        n_batches = (len(formatted_inputs) + BATCH_SIZE - 1) // BATCH_SIZE

        _LOGGER.debug("ABSA_ENGINE: Processing %d batches (batch_size=%d)", n_batches, BATCH_SIZE)

        # Process in batches
        for i in range(0, len(formatted_inputs), BATCH_SIZE):
            batch = formatted_inputs[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            if batch_num % 10 == 0 or batch_num == n_batches:
                _LOGGER.debug("ABSA_ENGINE: Processing batch %d/%d", batch_num, n_batches)

            # Tokenize
            encodings = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            encodings = {k: v.to(device) for k, v in encodings.items()}

            # Inference
            with torch.no_grad():
                outputs = model(**encodings)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            # Process each result in batch
            for j in range(len(batch)):
                prob_values = probs[j].cpu().numpy()
                pred_idx = prob_values.argmax()
                confidence = float(prob_values[pred_idx])

                # Map prediction to label
                # Model typically: 0=Negative, 1=Neutral, 2=Positive
                if pred_idx == 0:
                    label = "NEGATIVE"
                    brand_score = -confidence  # negative score
                elif pred_idx == 2:
                    label = "POSITIVE"
                    brand_score = confidence  # positive score
                else:
                    label = "NEUTRAL"
                    brand_score = 0.0  # neutral

                label_counts[label] += 1
                results.append({
                    "sentiment": label,
                    "prob": confidence,
                    "brand_sentiment_score": brand_score,
                })

        elapsed = time.time() - start
        avg_score = sum(r["brand_sentiment_score"] for r in results) / len(results) if results else 0

        _LOGGER.info(
            "ABSA_ENGINE: Brand sentiment for '%s' complete in %.2fs - %d texts (%.1f texts/sec) | "
            "POSITIVE=%d NEUTRAL=%d NEGATIVE=%d | avg_score=%.3f",
            target_aspect, elapsed, n_texts, n_texts / elapsed if elapsed > 0 else 0,
            label_counts["POSITIVE"], label_counts["NEUTRAL"], label_counts["NEGATIVE"],
            avg_score
        )

        return pd.DataFrame(results)

    def run_multi_aspect(
        self,
        texts: Sequence[str],
        aspects: Sequence[str],
    ) -> dict[str, pd.DataFrame]:
        """
        Score sentiment toward multiple aspects for the same texts.

        Args:
            texts: List of text strings
            aspects: List of aspects/brands to analyze (e.g., ["Capital One", "Chase", "Amex"])

        Returns:
            Dict mapping aspect name to DataFrame of sentiment results
        """
        return {
            aspect: self.run(texts, aspect)
            for aspect in aspects
        }
