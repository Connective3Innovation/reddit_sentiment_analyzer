# src/reddit_sentiment/sentiment/hf_engine.py
"""HuggingFace sentiment analysis engine.

Default model: cardiffnlp/twitter-roberta-base-sentiment-latest
- Trained on 60M+ tweets, excellent for social media text
- 3-class: negative/neutral/positive (not just binary)
- Better at understanding comparisons, informal language, sarcasm
"""
from __future__ import annotations
import logging
import os
import time
from functools import lru_cache
from typing import Final

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

_LOGGER: Final = logging.getLogger(__name__)

# Tunables (env overrides)
# Default: twitter-roberta (3-class, trained on social media)
# Alternative: distilbert-base-uncased-finetuned-sst-2-english (binary, faster)
MODEL_ID = os.getenv("HF_MODEL_ID", "cardiffnlp/twitter-roberta-base-sentiment-latest")
MAX_LEN = int(os.getenv("HF_MAX_LENGTH", "256"))
BATCH = int(os.getenv("HF_BATCH_SIZE", "32"))  # smaller batch for roberta

# Prefer a writable cache path in Cloud Run
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Label mapping for different models
LABEL_MAP = {
    # twitter-roberta uses these labels
    "negative": "NEGATIVE",
    "neutral": "NEUTRAL",
    "positive": "POSITIVE",
    # Some models use LABEL_0/1/2
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
    # distilbert-sst2 uses these
    "NEGATIVE": "NEGATIVE",
    "POSITIVE": "POSITIVE",
}


@lru_cache(maxsize=1)
def _get_pipeline():
    _LOGGER.info("HF_ENGINE: Loading model '%s' (max_len=%d, batch=%d)", MODEL_ID, MAX_LEN, BATCH)
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.model_max_length = MAX_LEN
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    _LOGGER.info("HF_ENGINE: Using device=%s", device_name)

    pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,  # Return all class probabilities for 3-class models
    )

    _LOGGER.info("HF_ENGINE: Model loaded in %.2fs", time.time() - start)
    return pipe


class HfEngine:
    def __init__(self):
        # lazy; creation handled by _get_pipeline() cache
        pass

    def run(self, texts):
        n_texts = len(texts)
        _LOGGER.info("HF_ENGINE: Starting generic sentiment analysis on %d texts", n_texts)
        start = time.time()

        texts = [t if isinstance(t, str) else "" for t in texts]
        empty_count = sum(1 for t in texts if not t.strip())
        if empty_count > 0:
            _LOGGER.debug("HF_ENGINE: %d/%d texts are empty", empty_count, n_texts)

        pipe = _get_pipeline()

        _LOGGER.debug("HF_ENGINE: Running inference (batch_size=%d)", BATCH)
        inference_start = time.time()
        outputs = pipe(
            texts,
            batch_size=BATCH,
            truncation=True,
            max_length=MAX_LEN,
        )
        _LOGGER.debug("HF_ENGINE: Inference completed in %.2fs", time.time() - inference_start)

        # Handle both single-label and multi-label (top_k) outputs
        results = []
        label_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

        for output in outputs:
            if isinstance(output, list):
                # Multi-class output: pick highest scoring label
                # Sort by score descending
                sorted_output = sorted(output, key=lambda x: x["score"], reverse=True)
                best = sorted_output[0]
                label = LABEL_MAP.get(best["label"], best["label"].upper())
                prob = best["score"]

                # For NEUTRAL, we want lower absolute sentiment score
                # Adjust prob to reflect this
                if label == "NEUTRAL":
                    # Neutral gets low absolute score (close to 0)
                    prob = 0.5 + (prob - 0.5) * 0.2  # Compress toward 0.5
            else:
                # Single-label output (binary models like distilbert-sst2)
                label = LABEL_MAP.get(output["label"], output["label"].upper())
                prob = output["score"]

            label_counts[label] = label_counts.get(label, 0) + 1
            results.append({"sentiment": label, "prob": prob})

        elapsed = time.time() - start
        _LOGGER.info(
            "HF_ENGINE: Sentiment complete in %.2fs - %d texts (%.1f texts/sec) | "
            "POSITIVE=%d (%.1f%%) NEUTRAL=%d (%.1f%%) NEGATIVE=%d (%.1f%%)",
            elapsed, n_texts, n_texts / elapsed if elapsed > 0 else 0,
            label_counts["POSITIVE"], 100 * label_counts["POSITIVE"] / n_texts if n_texts > 0 else 0,
            label_counts["NEUTRAL"], 100 * label_counts["NEUTRAL"] / n_texts if n_texts > 0 else 0,
            label_counts["NEGATIVE"], 100 * label_counts["NEGATIVE"] / n_texts if n_texts > 0 else 0,
        )

        return pd.DataFrame(results)
