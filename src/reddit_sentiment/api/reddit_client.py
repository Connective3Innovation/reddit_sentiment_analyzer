"""Thin wrapper around PRAW providing typed, rate‑limited calls and pagination helpers."""
from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Final, Iterable, List

import pandas as pd
from praw.models import Comment, Submission

from ..auth import get_client

_LOGGER: Final = logging.getLogger(__name__)


class RedditClient:
    """Encapsulates Reddit read‑only operations used by the pipeline."""

    RATE_LIMIT_THRESHOLD = 5  # Reddit sends X‑Ratelimit‑Remaining header

    def __init__(self) -> None:
        self._client = get_client()

    # ──────────────────────────────
    # Public API
    # ──────────────────────────────

    def search_posts(self, keyword: str, *, limit: int, days_back: int) -> List[Submission]:
        # Map days_back to Reddit's server-side time_filter when possible
        if days_back <= 1:
            tf = "day"
        elif days_back <= 7:
            tf = "week"
        elif days_back <= 31:
            tf = "month"
        elif days_back <= 365:
            tf = "year"
        else:
            tf = "all"

        gen = self._client.subreddit("all").search(
            query=keyword,
            sort="new",
            time_filter=tf,
            limit=None,  # let PRAW paginate; we will stop at cutoff/limit
        )

        cutoff_dt = dt.datetime.utcnow() - dt.timedelta(days=days_back)
        cutoff_ts = cutoff_dt.replace(tzinfo=dt.timezone.utc)

        results = []
        for s in gen:
            created = dt.datetime.fromtimestamp(s.created_utc, tz=dt.timezone.utc)
            if created < cutoff_ts:
                break  # sorted by new; everything after is older
            results.append(s)
            if len(results) >= limit:
                break
        return results
    
    def fetch_comments(self, submission: Submission) -> List[Comment]:
        """Expand the full comment tree (may be heavy!)."""
        submission.comments.replace_more(limit=None)
        return submission.comments.list()

    # ──────────────────────────────
    # Utilities
    # ──────────────────────────────

    @staticmethod
    def to_dataframe(posts: Iterable[Submission], comments_map: dict[str, List[Comment]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        post_rows, comment_rows = [], []
        for s in posts:
            post_rows.append(
                {
                    "id": s.id,
                    "title": s.title,
                    "selftext": s.selftext,
                    "created": dt.datetime.fromtimestamp(s.created_utc, tz=dt.timezone.utc),
                    "score": s.score,
                    "num_comments": s.num_comments,
                    "subreddit": str(s.subreddit.display_name),
                    "url": s.url,
                }
            )
            for c in comments_map.get(s.id, []):
                comment_rows.append(
                    {
                        "post_id": s.id,
                        "comment_id": c.id,
                        "body": getattr(c, "body", ""),
                        "created": dt.datetime.fromtimestamp(c.created_utc, tz=dt.timezone.utc),
                        "score": c.score,
                    }
                )
        return pd.DataFrame(post_rows), pd.DataFrame(comment_rows)

    # ──────────────────────────────
    # Internal helpers
    # ──────────────────────────────

    def _respect_rate_limit(self) -> None:
        remaining = int(self._client.auth.limits["remaining"])
        if remaining < self.RATE_LIMIT_THRESHOLD:
            reset = float(self._client.auth.limits["reset_timestamp"])
            sleep_for = reset - time.time() + 2  # + buffer seconds
            _LOGGER.warning("Rate‑limit nearly hit. Sleeping %.1f seconds…", sleep_for)
            time.sleep(max(sleep_for, 0))