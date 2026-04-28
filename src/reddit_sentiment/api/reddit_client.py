"""Thin wrapper around PRAW providing typed, rate-limited calls and pagination helpers."""
from __future__ import annotations

import datetime as dt
import logging
import os
import time
from typing import Final, Iterable, List

import pandas as pd
from praw.models import Comment, Submission

from ..auth import get_client  # keep existing import path

_LOGGER: Final = logging.getLogger(__name__)


class RedditClient:
    """Encapsulates Reddit read-only operations used by the pipeline."""

    RATE_LIMIT_THRESHOLD = 5  # Reddit sends X-Ratelimit-Remaining header

    def __init__(self) -> None:
        self._client = get_client()

    # ──────────────────────────────
    # Public API
    # ──────────────────────────────

    def search_posts(
        self,
        keyword: str,
        *,
        limit: int,
        days_back: int,
        subreddits: List[str] | None = None,
    ) -> List[Submission]:
        """
        Search for posts with a bounded time window and skip mega-threads.
        - Uses Reddit's native time_filter for faster server-side filtering.
        - Applies an explicit UTC cutoff.
        - Optionally skips posts with very large comment counts.
        - Can target specific subreddits for higher relevance.

        Args:
            keyword: Search query
            limit: Max posts to return
            days_back: How many days back to search
            subreddits: Optional list of subreddits to search (e.g., ["personalfinance", "CreditCards"]).
                        If None or empty, searches all of Reddit.
        """
        # Map days_back to a native Reddit time_filter to reduce scan scope
        if days_back <= 1:
            time_filter = "day"
        elif days_back <= 7:
            time_filter = "week"
        elif days_back <= 31:
            time_filter = "month"
        elif days_back <= 365:
            time_filter = "year"
        else:
            time_filter = "all"

        # Target specific subreddits or search all
        if subreddits:
            subreddit_str = "+".join(subreddits)
            _LOGGER.info(
                "REDDIT_API: Searching %d subreddits: %s",
                len(subreddits), ", ".join(subreddits[:5]) + ("..." if len(subreddits) > 5 else "")
            )
        else:
            subreddit_str = "all"
            _LOGGER.info("REDDIT_API: Searching all of Reddit (no subreddit filter)")

        _LOGGER.info(
            "REDDIT_API: query='%s', limit=%d, time_filter=%s, days_back=%d",
            keyword[:50] + "..." if len(keyword) > 50 else keyword,
            limit, time_filter, days_back
        )

        start = time.time()
        submissions = self._client.subreddit(subreddit_str).search(
            query=keyword,
            sort="new",
            time_filter=time_filter,
            limit=limit,
        )

        cutoff = dt.datetime.utcnow() - dt.timedelta(days=days_back)
        cutoff_ts = cutoff.replace(tzinfo=dt.timezone.utc)

        max_comments = int(os.getenv("REDDIT_SKIP_LARGE_POSTS", "1200"))

        results: List[Submission] = []
        skipped_old = 0
        skipped_mega = 0
        subreddit_counts: dict[str, int] = {}

        for s in submissions:
            created_ts = dt.datetime.fromtimestamp(s.created_utc, tz=dt.timezone.utc)
            if created_ts < cutoff_ts:
                skipped_old += 1
                continue
            if getattr(s, "num_comments", 0) > max_comments:
                skipped_mega += 1
                _LOGGER.debug(
                    "REDDIT_API: Skipping mega-thread %s (%s comments)", s.id, s.num_comments
                )
                continue

            sub_name = str(s.subreddit.display_name)
            subreddit_counts[sub_name] = subreddit_counts.get(sub_name, 0) + 1
            results.append(s)

        elapsed = time.time() - start

        # Log subreddit distribution
        top_subs = sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_subs_str = ", ".join(f"{k}={v}" for k, v in top_subs)

        _LOGGER.info(
            "REDDIT_API: Search complete in %.2fs - %d posts found | "
            "skipped: %d old, %d mega-threads | top subreddits: %s",
            elapsed, len(results), skipped_old, skipped_mega, top_subs_str
        )

        return results

    def fetch_comments(
        self,
        submission: Submission,
        *,
        more_limit: int = 2,
        per_post_limit: int | None = 300,
    ) -> List[Comment]:
        """
        Expand a bounded portion of the comment tree for speed.
        - more_limit: number of MoreComments objects to resolve (None = ALL; slow!)
        - per_post_limit: keep top-N comments by score (None = keep all)
        """
        submission.comments.replace_more(limit=more_limit)
        comments: List[Comment] = submission.comments.list()
        original_count = len(comments)

        if per_post_limit is not None and len(comments) > per_post_limit:
            # Keep the highest-signal comments
            comments.sort(key=lambda c: getattr(c, "score", 0), reverse=True)
            comments = comments[:per_post_limit]
            _LOGGER.debug(
                "REDDIT_API: Post %s - kept %d/%d comments (sorted by score)",
                submission.id, len(comments), original_count
            )

        return comments

    # ──────────────────────────────
    # Utilities
    # ──────────────────────────────

    @staticmethod
    def to_dataframe(
        posts: Iterable[Submission],
        comments_map: dict[str, List[Comment]],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        post_rows, comment_rows = [], []
        for s in posts:
            post_rows.append(
                {
                    "id": s.id,
                    "title": s.title,
                    "selftext": getattr(s, "selftext", ""),
                    "created": dt.datetime.fromtimestamp(s.created_utc, tz=dt.timezone.utc),
                    "score": s.score,
                    "num_comments": s.num_comments,
                    "subreddit": str(s.subreddit.display_name),
                    "url": s.url,
                }
            )
            for c in comments_map.get(s.id, []):
                created_utc = getattr(c, "created_utc", None)
                created_ts = (
                    dt.datetime.fromtimestamp(created_utc, tz=dt.timezone.utc)
                    if created_utc is not None
                    else None
                )
                # Get author name (handle deleted authors)
                author = getattr(c, "author", None)
                author_name = str(author) if author else "[deleted]"

                comment_rows.append(
                    {
                        "post_id": s.id,
                        "comment_id": c.id,
                        "body": getattr(c, "body", ""),
                        "created": created_ts,
                        "created_utc": created_ts,  # Also add created_utc for consistency
                        "score": getattr(c, "score", 0),
                        "author": author_name,
                        "subreddit": str(s.subreddit.display_name),  # Inherit from post
                    }
                )
        return pd.DataFrame(post_rows), pd.DataFrame(comment_rows)

    # ──────────────────────────────
    # Internal helpers
    # ──────────────────────────────

    def _respect_rate_limit(self) -> None:
        remaining = int(self._client.auth.limits.get("remaining", 60))
        if remaining < self.RATE_LIMIT_THRESHOLD:
            reset = float(self._client.auth.limits.get("reset_timestamp", time.time() + 5))
            sleep_for = reset - time.time() + 2  # + buffer seconds
            _LOGGER.warning("Rate-limit nearly hit. Sleeping %.1f seconds…", sleep_for)
            time.sleep(max(sleep_for, 0))
