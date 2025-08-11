"""Basic text normalisation before sentiment analysis."""
import re
from typing import Final

import pandas as pd

_URL_RE: Final = re.compile(r"https?://\S+")
_WHITESPACE_RE: Final = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def apply_cleaning(df: pd.DataFrame, column: str = "body") -> pd.DataFrame:    
    if column not in df.columns:
        return df
    df[column] = df[column].fillna("").astype(str).apply(clean_text)
    return df