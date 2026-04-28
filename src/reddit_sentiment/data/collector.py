# src/reddit_sentiment/data/collector.py
"""High-level data collection coordinating API search & bounded comment expansion.

Env toggles:
- REDDIT_MORE_LIMIT           (default: "5")   → PRAW replace_more() limit (None = ALL, slow!)
- REDDIT_COMMENTS_PER_POST    (default: "500") → Keep top-N comments per post by score
"""

from __future__ import annotations

import logging
import os
import re
from typing import Final, TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

from ..api import RedditClient  # keep your existing import path

if TYPE_CHECKING:
    from ..config.clients import ClientConfig

_LOGGER: Final = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name, str(default))
    try:
        return int(val)
    except Exception:
        _LOGGER.warning("Invalid %s=%r; falling back to %d", name, val, default)
        return default


def _contains_brand_keywords(text: str, brand_terms: list[str]) -> bool:
    """Check if text contains any of the brand keywords (case-insensitive, word boundary)."""
    if not text or not brand_terms:
        return True  # If no terms to check, don't filter
    text_lower = text.lower()
    for term in brand_terms:
        # Use word boundary matching for accuracy
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        if re.search(pattern, text_lower):
            return True
    return False


def collect(
    keyword: str,
    *,
    limit: int,
    days_back: int,
    subreddits: list[str] | None = None,
    brand_terms: list[str] | None = None,
    require_keyword_match: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return posts & comments DataFrames for a keyword, respecting request limits.

    Args:
        keyword: Search query
        limit: Max posts to fetch
        days_back: How far back to search
        subreddits: Optional list of subreddits to target (reduces noise)
        brand_terms: Optional list of brand terms for post-filter validation
                     (e.g., ["capital one", "capitalone", "cap one"])
        require_keyword_match: If True, filter out posts that don't contain brand_terms
    """
    client = RedditClient()

    _LOGGER.info(
        "Collecting for keyword=%r with limit=%d, days_back=%d, subreddits=%s",
        keyword, limit, days_back, subreddits or "all"
    )
    posts = client.search_posts(keyword, limit=limit, days_back=days_back, subreddits=subreddits)
    _LOGGER.info("Fetched %d posts from Reddit", len(posts))

    # Filter posts that don't actually contain the brand keywords
    if require_keyword_match and brand_terms:
        original_count = len(posts)
        posts = [
            p for p in posts
            if _contains_brand_keywords(
                f"{p.title} {getattr(p, 'selftext', '')}",
                brand_terms
            )
        ]
        filtered_count = original_count - len(posts)
        if filtered_count > 0:
            _LOGGER.info(
                "Filtered %d/%d posts that didn't contain brand keywords",
                filtered_count, original_count
            )
    _LOGGER.info("Processing %d relevant posts", len(posts))

    # Tunables for comment fetching
    more_limit = _env_int("REDDIT_MORE_LIMIT", 5)
    per_post_limit = _env_int("REDDIT_COMMENTS_PER_POST", 500)
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
