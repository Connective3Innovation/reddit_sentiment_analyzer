"""Sentiment analysis using HuggingFace transformers.

Provides lazy loading to avoid heavy imports at module import-time.

Two analysis modes:
1. analyze() - Generic sentiment (is the text positive/negative?)
2. analyze_toward_brand() - Brand-directed sentiment (is sentiment toward X positive/negative?)
"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .hf_engine import HfEngine
    from .absa_engine import AbsaEngine

# Cache of already-instantiated engines
_ENGINE_CACHE: dict[str, "HfEngine | AbsaEngine"] = {}


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


def _get_absa_engine():
    """Get or create ABSA engine (lazy loading)."""
    if "absa" in _ENGINE_CACHE:
        return _ENGINE_CACHE["absa"]

    try:
        from .absa_engine import AbsaEngine
    except Exception as err:
        raise _UnavailableEngine(
            "ABSA engine dependencies missing.\n"
            "Install: transformers, torch."
        ) from err

    _ENGINE_CACHE["absa"] = AbsaEngine()
    return _ENGINE_CACHE["absa"]


def analyze_toward_brand(
    texts: Sequence[str],
    brand: str,
) -> pd.DataFrame:
    """Return sentiment scores TOWARD a specific brand/entity.

    Unlike analyze(), this measures sentiment directed at the brand,
    not the overall sentiment of the text.

    Example:
        "I left Capital One for Chase and I'm so happy"
        - analyze() → POSITIVE (text is happy)
        - analyze_toward_brand(brand="Capital One") → NEGATIVE (user left them)
        - analyze_toward_brand(brand="Chase") → POSITIVE (user is happy with them)

    Args:
        texts: Sequence of text strings to analyze
        brand: The brand/entity to measure sentiment toward

    Returns:
        DataFrame with columns:
        - sentiment: POSITIVE, NEGATIVE, or NEUTRAL (toward the brand)
        - prob: confidence score (0-1)
        - brand_sentiment_score: -1 to +1 scale
    """
    eng = _get_absa_engine()
    return eng.run(texts, brand)