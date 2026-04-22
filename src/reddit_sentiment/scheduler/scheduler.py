# src/reddit_sentiment/scheduler/scheduler.py
"""
Job scheduler for automated sentiment data collection.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Optional, Literal

from croniter import croniter

from ..analytics.models import TrackedKeyword
from ..analytics.bq_store import BigQueryStore
from ..analytics.aggregator import Aggregator

logger = logging.getLogger(__name__)


class Scheduler:
    """Manages scheduled sentiment analysis jobs."""

    def __init__(
        self,
        bq_project: Optional[str] = None,
        bq_dataset: str = "reddit_sentiment",
    ):
        """
        Initialize scheduler.

        Args:
            bq_project: GCP project ID
            bq_dataset: BigQuery dataset name
        """
        self.bq_project = bq_project or os.getenv("REDDIT_GCP_PROJECT")
        self.bq_dataset = bq_dataset
        self._store: Optional[BigQueryStore] = None
        self._aggregator: Optional[Aggregator] = None

    @property
    def store(self) -> BigQueryStore:
        """Lazy-load BigQuery store."""
        if self._store is None:
            if not self.bq_project:
                raise ValueError("BigQuery project required. Set REDDIT_GCP_PROJECT.")
            self._store = BigQueryStore(self.bq_project, self.bq_dataset)
        return self._store

    @property
    def aggregator(self) -> Aggregator:
        """Lazy-load aggregator."""
        if self._aggregator is None:
            self._aggregator = Aggregator(
                bq_project=self.bq_project,
                bq_dataset=self.bq_dataset,
            )
        return self._aggregator

    def add_keyword(
        self,
        keyword: str,
        schedule: str = "0 6 * * *",
        days_back: int = 1,
        limit: int = 1000,
        engine: Literal["vader", "hf"] = "vader",
    ) -> TrackedKeyword:
        """
        Add a keyword to tracking.

        Args:
            keyword: Keyword to track
            schedule: Cron expression (default: daily at 6 AM)
            days_back: Days of data to collect
            limit: Maximum posts per run
            engine: Sentiment engine

        Returns:
            TrackedKeyword object
        """
        now = datetime.now(timezone.utc)

        # Calculate next run time
        try:
            cron = croniter(schedule, now)
            next_run = cron.get_next(datetime)
            next_run = next_run.replace(tzinfo=timezone.utc)
        except Exception as e:
            logger.warning(f"Invalid cron expression '{schedule}': {e}")
            next_run = now + timedelta(days=1)

        tracked = TrackedKeyword(
            keyword=keyword,
            schedule=schedule,
            last_run=None,
            next_run=next_run,
            enabled=True,
            config={
                "days_back": days_back,
                "limit": limit,
                "engine": engine,
            },
        )

        self.store.save_tracked_keyword(tracked)
        logger.info(f"Added keyword '{keyword}' with schedule '{schedule}'")

        return tracked

    def remove_keyword(self, keyword: str) -> None:
        """Remove a keyword from tracking."""
        self.store.delete_tracked_keyword(keyword)
        logger.info(f"Removed keyword '{keyword}' from tracking")

    def update_keyword(
        self,
        keyword: str,
        schedule: Optional[str] = None,
        enabled: Optional[bool] = None,
        days_back: Optional[int] = None,
        limit: Optional[int] = None,
        engine: Optional[str] = None,
    ) -> Optional[TrackedKeyword]:
        """Update tracking configuration for a keyword."""
        keywords = self.store.get_tracked_keywords(enabled_only=False)
        tracked = next((k for k in keywords if k.keyword == keyword), None)

        if not tracked:
            logger.warning(f"Keyword '{keyword}' not found in tracking")
            return None

        # Update fields
        if schedule is not None:
            tracked.schedule = schedule
            # Recalculate next run
            try:
                cron = croniter(schedule, datetime.now(timezone.utc))
                tracked.next_run = cron.get_next(datetime).replace(tzinfo=timezone.utc)
            except Exception:
                pass

        if enabled is not None:
            tracked.enabled = enabled

        # Update config
        if days_back is not None:
            tracked.config["days_back"] = days_back
        if limit is not None:
            tracked.config["limit"] = limit
        if engine is not None:
            tracked.config["engine"] = engine

        self.store.save_tracked_keyword(tracked)
        return tracked

    def get_tracked_keywords(
        self, enabled_only: bool = True
    ) -> list[TrackedKeyword]:
        """Get all tracked keywords."""
        return self.store.get_tracked_keywords(enabled_only=enabled_only)

    def get_due_jobs(self) -> list[TrackedKeyword]:
        """Get keywords that are due for collection."""
        now = datetime.now(timezone.utc)
        keywords = self.store.get_tracked_keywords(enabled_only=True)

        due = []
        for kw in keywords:
            if kw.next_run and kw.next_run <= now:
                due.append(kw)

        return due

    def run_job(self, keyword: str) -> dict:
        """
        Run aggregation for a single keyword.

        Args:
            keyword: Keyword to process

        Returns:
            Aggregation results dict
        """
        # Get tracking config
        keywords = self.store.get_tracked_keywords(enabled_only=False)
        tracked = next((k for k in keywords if k.keyword == keyword), None)

        config = tracked.config if tracked else {}
        days_back = config.get("days_back", 1)
        limit = config.get("limit", 1000)
        engine = config.get("engine", "vader")

        # Run aggregation
        result = self.aggregator.run_aggregation(
            keyword=keyword,
            days_back=days_back,
            limit=limit,
            engine=engine,
        )

        # Update tracking metadata
        if tracked:
            now = datetime.now(timezone.utc)
            tracked.last_run = now

            # Calculate next run
            try:
                cron = croniter(tracked.schedule, now)
                tracked.next_run = cron.get_next(datetime).replace(tzinfo=timezone.utc)
            except Exception:
                tracked.next_run = now + timedelta(days=1)

            self.store.save_tracked_keyword(tracked)

        return result

    def run_due_jobs(self) -> list[dict]:
        """
        Run all jobs that are due.

        Returns:
            List of aggregation results for each job
        """
        due = self.get_due_jobs()
        if not due:
            logger.info("No jobs due for execution")
            return []

        logger.info(f"Running {len(due)} due jobs")
        results = []

        for tracked in due:
            try:
                result = self.run_job(tracked.keyword)
                results.append(result)
            except Exception as e:
                logger.error(f"Job failed for '{tracked.keyword}': {e}")
                results.append({
                    "keyword": tracked.keyword,
                    "status": "error",
                    "error": str(e),
                })

        return results

    def get_status(self) -> dict:
        """
        Get scheduler status summary.

        Returns:
            Status dict with tracked keywords and next runs
        """
        keywords = self.store.get_tracked_keywords(enabled_only=False)
        due = self.get_due_jobs()

        return {
            "total_tracked": len(keywords),
            "enabled": sum(1 for k in keywords if k.enabled),
            "disabled": sum(1 for k in keywords if not k.enabled),
            "due_now": len(due),
            "keywords": [
                {
                    "keyword": k.keyword,
                    "schedule": k.schedule,
                    "enabled": k.enabled,
                    "last_run": k.last_run.isoformat() if k.last_run else None,
                    "next_run": k.next_run.isoformat() if k.next_run else None,
                }
                for k in keywords
            ],
        }


@lru_cache(maxsize=1)
def get_scheduler() -> Scheduler:
    """Get singleton scheduler instance."""
    return Scheduler()


def init_from_env() -> Scheduler:
    """
    Initialize scheduler from environment variables.

    Reads REDDIT_TRACKED_KEYWORDS (comma-separated) and adds them with
    REDDIT_TRACKING_SCHEDULE.
    """
    scheduler = get_scheduler()

    # Get tracked keywords from env
    keywords_str = os.getenv("REDDIT_TRACKED_KEYWORDS", "")
    if not keywords_str:
        return scheduler

    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
    schedule = os.getenv("REDDIT_TRACKING_SCHEDULE", "0 6 * * *")

    # Add keywords that aren't already tracked
    existing = {k.keyword for k in scheduler.get_tracked_keywords(enabled_only=False)}

    for keyword in keywords:
        if keyword not in existing:
            scheduler.add_keyword(keyword, schedule=schedule)
            logger.info(f"Added keyword '{keyword}' from environment")

    return scheduler
