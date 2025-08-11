"""Smoke tests for sentiment engines.

We *always* run the VADER path. The HuggingFace path is optional: if its
dependencies (transformers + model downloads) aren’t available we simply skip
that check so the suite stays lightweight by default.
"""
import pytest

from reddit_sentiment.sentiment import analyze, _ENGINE_CACHE  # type: ignore[attr-defined]


def test_vader():
    df = analyze(["I love pizza", "I hate traffic"], engine="vader")
    assert len(df) == 2
    assert df["compound"].iloc[0] > 0  # positive
    assert df["compound"].iloc[1] < 0  # negative


@pytest.mark.skipif(
    "hf" not in _ENGINE_CACHE,
    reason="HuggingFace engine not available in this environment",
)
def test_hf():
    df = analyze(["I love pizza", "I hate traffic"], engine="hf")
    assert len(df) == 2
    assert set(df.columns).issuperset({"sentiment", "prob"})