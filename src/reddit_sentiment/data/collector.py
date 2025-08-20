# src/reddit_sentiment/data/collector.py
"""High-level data collection coordinating API search & bounded comment expansion.

Env toggles:
- REDDIT_MORE_LIMIT           (default: "2")   → PRAW replace_more() limit (None = ALL, slow!)
- REDDIT_COMMENTS_PER_POST    (default: "300") → Keep top-N comments per post by score
"""

from __future__ import annotations

import logging
import os
from typing import Final

import pandas as pd
from tqdm import tqdm

from ..api import RedditClient  # keep your existing import path

_LOGGER: Final = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name, str(default))
    try:
        return int(val)
    except Exception:
        _LOGGER.warning("Invalid %s=%r; falling back to %d", name, val, default)
        return default


def collect(keyword: str, *, limit: int, days_back: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return posts & comments DataFrames for a keyword, respecting request limits."""
    client = RedditClient()

    _LOGGER.info(
        "Collecting for keyword=%r with limit=%d, days_back=%d",
        keyword, limit, days_back
    )
    posts = client.search_posts(keyword, limit=limit, days_back=days_back)
    _LOGGER.info("Fetched %d posts", len(posts))

    # Tunables for comment fetching
    more_limit = _env_int("REDDIT_MORE_LIMIT", 2)
    per_post_limit = _env_int("REDDIT_COMMENTS_PER_POST", 300)
    _LOGGER.info(
        "Comment bounds: replace_more(limit=%d), per_post_limit=%d",
        more_limit, per_post_limit
    )

    comments_map: dict[str, list] = {}
    for post in tqdm(posts, desc="Expanding comment trees"):
        comments_map[post.id] = client.fetch_comments(
            post, more_limit=more_limit, per_post_limit=per_post_limit
        )

    df_posts, df_comments = client.to_dataframe(posts, comments_map)
    _LOGGER.info("Parsed %d comments", len(df_comments))
    return df_posts, df_comments
