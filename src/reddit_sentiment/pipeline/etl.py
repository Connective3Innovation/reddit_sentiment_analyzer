# src/reddit_sentiment/pipeline/etl.py

"""End-to-end orchestrator: collect ➜ clean ➜ analyze ➜ persist (local or GCS)."""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Final

import pandas as pd
import fsspec  # <- enable gs:// writes via gcsfs

from ..config.settings import get_settings
from ..data.collector import collect
from ..data.preprocess import apply_cleaning
from ..sentiment import analyze
from ..cache import make_cache, CacheKey

_LOGGER: Final = logging.getLogger(__name__)


def _write_parquet(df: pd.DataFrame, base: str | Path, filename: str) -> str:
    """
    Write parquet to either a local directory or a GCS URL (gs://...).
    Returns a string path that pandas can read with fsspec.
    """
    base_str = str(base)
    if base_str.startswith("gs://"):
        url = base_str.rstrip("/") + "/" + filename
        with fsspec.open(url, "wb") as f:
            df.to_parquet(f, index=False)
        return url
    else:
        p = Path(base_str) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        return str(p)


def run_etl(keyword: str) -> str:
    """
    Run the full ETL for a keyword and persist the comments+sentiment parquet.

    Returns
    -------
    str
        Path/URL to the written parquet. This is a plain string so that
        'gs://...' URLs remain intact for pandas/fsspec.
    """
    cfg = get_settings()

    # 1) Collect (with caching)
    cache = make_cache(
        cfg.cache_backend,
        local_dir=cfg.output_dir,
        gcs_prefix=cfg.gcs_cache_prefix,
        gcp_project=cfg.gcp_project,
    )
    key = CacheKey(keyword=keyword, days_back=cfg.days_back, max_posts=cfg.max_posts)

    if cache.exists(key) and cache.is_fresh(key, cfg.cache_max_age_hours):
        posts_df, comments_df = cache.load(key)
    else:
        posts_df, comments_df = collect(keyword)
        cache.save(key, posts_df, comments_df)

    # 2) Clean
    comments_df = apply_cleaning(comments_df, column="body")

    # Build an output filename
    ts = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{keyword.replace(' ', '_')}_{ts}.parquet"

    # If no comments (or no 'body'), write an empty file and bail out
    if comments_df.empty or "body" not in comments_df.columns:
        out_path = _write_parquet(comments_df, cfg.output_dir, filename)
        _LOGGER.warning(
            "No comments to analyze for '%s'; wrote empty file: %s", keyword, out_path
        )
        return out_path

    # 3) Sentiment
    sentiment_df = analyze(comments_df["body"], engine=cfg.sentiment_engine)
    comments_df = pd.concat([comments_df.reset_index(drop=True),
                             sentiment_df.reset_index(drop=True)], axis=1)

    # 4) Persist
    out_path = _write_parquet(comments_df, cfg.output_dir, filename)
    _LOGGER.info("Saved %d comment rows to %s", len(comments_df), out_path)
    return out_path

