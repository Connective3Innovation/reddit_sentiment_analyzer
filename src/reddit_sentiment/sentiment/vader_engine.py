"""Rule‑based VADER implementation."""
from __future__ import annotations

import logging
from typing import Final, Sequence

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_LOGGER: Final = logging.getLogger(__name__)


class VaderEngine:
    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def run(self, texts: Sequence[str]) -> pd.DataFrame:
        _LOGGER.debug("Running VADER on %d texts", len(texts))
        scores = [self._analyzer.polarity_scores(t) for t in texts]
        return pd.DataFrame(scores)