"""ETL integration test using a stubbed Reddit client to avoid external calls."""
import pathlib

import pandas as pd
import pytest

from reddit_sentiment.pipeline.etl import run_etl


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("REDDIT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("REDDIT_MAX_POSTS", "1")  # keep tiny


@pytest.mark.skip("Requires live Reddit creds — run manually")
def test_etl_live():
    path = run_etl("openai")
    assert pathlib.Path(path).exists()
    df = pd.read_parquet(path)
    assert not df.empty