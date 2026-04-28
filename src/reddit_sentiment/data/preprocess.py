"""Basic text normalisation before sentiment analysis."""
import logging
import re
from typing import Final

import pandas as pd

_LOGGER: Final = logging.getLogger(__name__)

_URL_RE: Final = re.compile(r"https?://\S+")
_WHITESPACE_RE: Final = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def apply_cleaning(df: pd.DataFrame, column: str = "body") -> pd.DataFrame:
    """Clean text column in DataFrame."""
    if column not in df.columns:
        _LOGGER.warning("Column '%s' not found in DataFrame, skipping cleaning", column)
        return df

    original_len = len(df)
    _LOGGER.info("PREPROCESS: Starting text cleaning on %d rows, column='%s'", original_len, column)

    # Count URLs before cleaning
    url_count = df[column].fillna("").str.contains(r"https?://", regex=True).sum()
    _LOGGER.debug("PREPROCESS: Found %d rows with URLs to strip", url_count)

    df[column] = df[column].fillna("").astype(str).apply(clean_text)

    # Count empty after cleaning
    empty_count = (df[column].str.len() == 0).sum()
    avg_len = df[column].str.len().mean()

    _LOGGER.info(
        "PREPROCESS: Cleaning complete - %d rows, %d empty after cleaning, avg length=%.1f chars",
        original_len, empty_count, avg_len
    )
    return df