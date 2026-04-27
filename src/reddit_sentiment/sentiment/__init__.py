"""Sentiment analysis using HuggingFace transformers.

Provides lazy loading to avoid heavy imports at module import-time.
"""
from __future__ import annotations

from typing import Literal, Sequence

import pandas as pd

# Cache of already-instantiated engine
_ENGINE_CACHE: dict[str, "HfEngine"] = {}


class _UnavailableEngine(Exception):
    """Raised when HuggingFace dependencies are missing."""


def _get_engine():
    if "hf" in _ENGINE_CACHE:
        return _ENGINE_CACHE["hf"]

    try:
        from .hf_engine import HfEngine
    except Exception as err:
        raise _UnavailableEngine(
            "HuggingFace engine dependencies missing.\n"
            "Install: transformers, torch, protobuf, sentencepiece."
        ) from err

    _ENGINE_CACHE["hf"] = HfEngine()
    return _ENGINE_CACHE["hf"]


def analyze(texts: Sequence[str], engine: str = "hf") -> pd.DataFrame:
    """Return sentiment scores as a DataFrame for *texts*.

    Args:
        texts: Sequence of text strings to analyze
        engine: Ignored (kept for backward compatibility). Always uses HuggingFace.

    Returns:
        DataFrame with 'sentiment' (POSITIVE/NEGATIVE) and 'prob' (confidence) columns.
    """
    eng = _get_engine()
    return eng.run(texts)