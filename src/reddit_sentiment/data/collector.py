"""High‑level data collection coordinating API search & comment expansion."""
from __future__ import annotations

import logging
from typing import Final

import pandas as pd
from tqdm import tqdm

from ..api import RedditClient
from ..config.settings import get_settings

_LOGGER: Final = logging.getLogger(__name__)


def collect(keyword: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return posts & comments DataFrames for a keyword respecting config limits."""
    cfg = get_settings()
    client = RedditClient()

    posts = client.search_posts(keyword, limit=cfg.max_posts, days_back=cfg.days_back)
    _LOGGER.info("Fetched %d posts", len(posts))

    comments_map: dict[str, list] = {}
    for post in tqdm(posts, desc="Expanding comment trees"):
        comments_map[post.id] = client.fetch_comments(post)

    df_posts, df_comments = client.to_dataframe(posts, comments_map)
    _LOGGER.info("Parsed %d comments", len(df_comments))
    return df_posts, df_comments