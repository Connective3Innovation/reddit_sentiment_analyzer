# src/reddit_sentiment/pipeline/etl.py

"""End-to-end orchestrator: collect ➜ clean ➜ analyze ➜ persist."""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Final

import pandas as pd

from ..config.settings import get_settings
from ..data.collector import collect
from ..data.preprocess import apply_cleaning
from ..sentiment import analyze
from pathlib import Path
from ..cache import make_cache, CacheKey
_LOGGER: Final = logging.getLogger(__name__)

def run_etl(keyword: str) -> Path:
    cfg = get_settings()

    # 1️⃣ Collect
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

    # 2️⃣ Clean
    comments_df = apply_cleaning(comments_df, column="body")

    # If no comments (or no 'body' column), skip sentiment and just write an empty file
    if comments_df.empty or "body" not in comments_df.columns:
        ts = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        out_path = cfg.output_dir / f"{keyword.replace(' ', '_')}_{ts}.parquet"
        comments_df.to_parquet(out_path, index=False)
        _LOGGER.warning(
            "No comments to analyze for '%s'; wrote empty file: %s", keyword, out_path
        )
        return out_path

    # 3️⃣ Sentiment
    sentiment_df = analyze(comments_df["body"], engine=cfg.sentiment_engine)
    comments_df = pd.concat([comments_df, sentiment_df], axis=1)

    # 4️⃣ Persist (parquet)
    ts = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_path = cfg.output_dir / f"{keyword.replace(' ', '_')}_{ts}.parquet"
    comments_df.to_parquet(out_path, index=False)
    _LOGGER.info("Saved %d comment rows to %s", len(comments_df), out_path)
    return out_path
