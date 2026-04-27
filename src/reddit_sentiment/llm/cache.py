# src/reddit_sentiment/llm/cache.py
"""
Simple LLM result caching to avoid repeated API calls.

Supports:
- In-memory cache (for Streamlit session state)
- File-based cache (for CLI batch processing)
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ".llm_cache"))


def _hash_inputs(primary_brand: str, industry: str, period_days: int, comment_count: int) -> str:
    """Generate cache key from analysis inputs."""
    key_data = f"{primary_brand}|{industry}|{period_days}|{comment_count}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def get_cached_result(
    primary_brand: str,
    industry: str,
    period_days: int,
    comment_count: int,
    max_age_hours: int = 24,
) -> Optional[dict]:
    """
    Retrieve cached LLM result if available and fresh.

    Args:
        primary_brand: Brand analyzed
        industry: Industry sector
        period_days: Analysis period
        comment_count: Number of comments (used as freshness indicator)
        max_age_hours: Maximum age of cache before invalidation

    Returns:
        Cached result dict or None if not found/stale
    """
    cache_key = _hash_inputs(primary_brand, industry, period_days, comment_count)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            cached = json.load(f)

        # Check freshness
        cached_at = datetime.fromisoformat(cached.get("_cached_at", "2000-01-01"))
        if datetime.now() - cached_at > timedelta(hours=max_age_hours):
            logger.info(f"Cache expired for {primary_brand}")
            return None

        # Check if comment count matches (data freshness)
        if cached.get("_comment_count") != comment_count:
            logger.info(f"Cache stale (comment count changed) for {primary_brand}")
            return None

        logger.info(f"Using cached LLM result for {primary_brand}")
        return cached.get("result")

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Cache read failed: {e}")
        return None


def save_to_cache(
    result: dict,
    primary_brand: str,
    industry: str,
    period_days: int,
    comment_count: int,
) -> None:
    """
    Save LLM result to cache.

    Args:
        result: LLM analysis result dict
        primary_brand: Brand analyzed
        industry: Industry sector
        period_days: Analysis period
        comment_count: Number of comments analyzed
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_key = _hash_inputs(primary_brand, industry, period_days, comment_count)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    cache_data = {
        "_cached_at": datetime.now().isoformat(),
        "_primary_brand": primary_brand,
        "_comment_count": comment_count,
        "result": result,
    }

    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2, default=str)
        logger.info(f"Cached LLM result for {primary_brand}")
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


def clear_cache(primary_brand: Optional[str] = None) -> int:
    """
    Clear LLM cache.

    Args:
        primary_brand: If provided, only clear cache for this brand.
                      If None, clear all cache.

    Returns:
        Number of cache files deleted
    """
    if not CACHE_DIR.exists():
        return 0

    deleted = 0
    for cache_file in CACHE_DIR.glob("*.json"):
        if primary_brand:
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                if cached.get("_primary_brand") != primary_brand:
                    continue
            except Exception:
                pass

        cache_file.unlink()
        deleted += 1

    return deleted
