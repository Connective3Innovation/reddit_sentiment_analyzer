# src/reddit_sentiment/sentiment/hf_engine.py
"""HuggingFace sentiment analysis engine.

Default model: cardiffnlp/twitter-roberta-base-sentiment-latest
- Trained on 60M+ tweets, excellent for social media text
- 3-class: negative/neutral/positive (not just binary)
- Better at understanding comparisons, informal language, sarcasm
"""
from __future__ import annotations
import os
from functools import lru_cache
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.model_max_length = MAX_LEN
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,  # Return all class probabilities for 3-class models
    )


class HfEngine:
    def __init__(self):
        # lazy; creation handled by _get_pipeline() cache
        pass

    def run(self, texts):
        texts = [t if isinstance(t, str) else "" for t in texts]
        pipe = _get_pipeline()
        outputs = pipe(
            texts,
            batch_size=BATCH,
            truncation=True,
            max_length=MAX_LEN,
        )

        # Handle both single-label and multi-label (top_k) outputs
        results = []
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

            results.append({"sentiment": label, "prob": prob})

        return pd.DataFrame(results)
