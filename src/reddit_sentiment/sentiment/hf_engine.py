# src/reddit_sentiment/sentiment/hf_engine.py
from __future__ import annotations
import os
from functools import lru_cache
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Tunables (env overrides)
MODEL_ID = os.getenv("HF_MODEL_ID", "distilbert-base-uncased-finetuned-sst-2-english")
MAX_LEN = int(os.getenv("HF_MAX_LENGTH", "256"))      # shorter = faster
BATCH   = int(os.getenv("HF_BATCH_SIZE", "64"))       # bigger = better CPU throughput

# Prefer a writable cache path in Cloud Run
os.environ.setdefault("HF_HOME", "/tmp/hf")
# Optional: faster HF downloads if they ever happen
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

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
        df = pd.DataFrame(outputs)
        return df.rename(columns={"label": "sentiment", "score": "prob"})
