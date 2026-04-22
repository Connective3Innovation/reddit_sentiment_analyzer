# src/reddit_sentiment/analytics/aggregator.py
"""
Daily aggregation pipeline for sentiment analytics.

Orchestrates the end-to-end process of:
1. Running ETL to collect fresh data
2. Calculating sentiment metrics
3. Creating snapshots
4. Calculating KPIs
5. Checking alerts
6. Identifying opportunities
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal

import pandas as pd
import numpy as np

from .models import SentimentSnapshot, SentimentDistribution, MetricEvent, ContentOpportunity
from .bq_store import BigQueryStore
from .shift_detector import ShiftDetector, calculate_trend
from .opportunity_scorer import OpportunityScorer
from ..metrics.kpi_registry import KPIRegistry
from ..metrics.alerts import AlertChecker

logger = logging.getLogger(__name__)


class Aggregator:
    """Orchestrates daily sentiment aggregation and analysis."""

    def __init__(
        self,
        bq_project: Optional[str] = None,
        bq_dataset: str = "reddit_sentiment",
        openrouter_api_key: Optional[str] = None,
    ):
        """
        Initialize aggregator.

        Args:
            bq_project: GCP project ID (defaults to REDDIT_GCP_PROJECT env var)
            bq_dataset: BigQuery dataset name
            openrouter_api_key: OpenRouter API key for LLM analysis
        """
        self.bq_project = bq_project or os.getenv("REDDIT_GCP_PROJECT")
        self.bq_dataset = bq_dataset
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

        # Initialize components
        self._store: Optional[BigQueryStore] = None
        self._shift_detector = ShiftDetector()
        self._opportunity_scorer = OpportunityScorer()
        self._alert_checker = AlertChecker()
        self._llm_client = None

    @property
    def store(self) -> BigQueryStore:
        """Lazy-load BigQuery store."""
        if self._store is None:
            if not self.bq_project:
                raise ValueError("BigQuery project required. Set REDDIT_GCP_PROJECT env var.")
            self._store = BigQueryStore(self.bq_project, self.bq_dataset)
        return self._store

    @property
    def llm_client(self):
        """Lazy-load OpenRouter client."""
        if self._llm_client is None and self.openrouter_api_key:
            from ..llm.openrouter_client import OpenRouterClient
            self._llm_client = OpenRouterClient(api_key=self.openrouter_api_key)
        return self._llm_client

    def run_aggregation(
        self,
        keyword: str,
        days_back: int = 1,
        limit: int = 1000,
        engine: Literal["vader", "hf"] = "vader",
        with_opportunities: bool = True,
        with_llm: bool = True,
    ) -> dict:
        """
        Run full aggregation pipeline for a keyword.

        Args:
            keyword: Search keyword
            days_back: Number of days to analyze
            limit: Maximum posts to fetch
            engine: Sentiment engine ("vader" or "hf")
            with_opportunities: Whether to score opportunities
            with_llm: Whether to use LLM for deep analysis

        Returns:
            Dict with aggregation results
        """
        logger.info(f"Starting aggregation for '{keyword}' ({days_back} days)")
        start_time = datetime.now(timezone.utc)

        # Step 1: Collect and analyze data
        df, posts_df, comments_df = self._run_etl(keyword, days_back, limit, engine)

        if df.empty:
            logger.warning(f"No data found for '{keyword}'")
            return {"keyword": keyword, "status": "no_data", "volume": 0}

        # Step 2: Create snapshot
        snapshot = self._create_snapshot(df, keyword, days_back, engine)

        # Step 3: Save snapshot to BigQuery
        self.store.save_snapshot(snapshot)

        # Step 4: Calculate and save KPIs
        kpi_values = KPIRegistry.calculate_all(df)
        metric_events = self._create_metric_events(keyword, kpi_values)
        self.store.save_metric_events_batch(metric_events)

        # Step 5: Check for alerts
        alerts = self._alert_checker.check_snapshot(snapshot)

        # Compare to baseline (previous day/week)
        baseline = self.store.get_latest_snapshot(keyword)
        if baseline and baseline.measured_at != snapshot.measured_at:
            baseline_alerts = self._alert_checker.compare_to_baseline(snapshot, baseline)
            alerts.extend(baseline_alerts)

        # Save alerts
        for alert in alerts:
            self.store.save_alert(alert)

        # Step 6: Detect shifts
        recent_snapshots = self.store.load_snapshots(
            keyword,
            start_date=(datetime.now(timezone.utc) - timedelta(days=30)).date(),
            limit=30,
        )
        shifts = self._shift_detector.detect_period_shifts(recent_snapshots)
        trend = calculate_trend(recent_snapshots)

        # Step 7: Identify opportunities
        opportunities = []
        if with_opportunities:
            opportunities = self._identify_opportunities(
                posts_df, comments_df, keyword, snapshot, with_llm
            )

            # Save opportunities
            for opp in opportunities[:10]:  # Limit saved opportunities
                self.store.save_opportunity(opp)

        # Calculate processing time
        took_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

        result = {
            "keyword": keyword,
            "status": "success",
            "volume": snapshot.volume,
            "snapshot": snapshot,
            "kpis": kpi_values,
            "alerts": alerts,
            "shifts": shifts,
            "trend": trend,
            "opportunities": opportunities,
            "took_ms": took_ms,
        }

        logger.info(
            f"Completed aggregation for '{keyword}': "
            f"{snapshot.volume} comments, {len(alerts)} alerts, "
            f"{len(opportunities)} opportunities in {took_ms}ms"
        )

        return result

    def _run_etl(
        self,
        keyword: str,
        days_back: int,
        limit: int,
        engine: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run ETL pipeline to collect and analyze data."""
        # Import ETL components
        try:
            from reddit_sentiment.data.collector import collect
            from reddit_sentiment.data.preprocess import apply_cleaning
            from reddit_sentiment.sentiment import analyze
        except ImportError:
            from ..data.collector import collect
            from ..data.preprocess import apply_cleaning
            from ..sentiment import analyze

        # Collect data
        posts_df, comments_df = collect(keyword, limit=limit, days_back=days_back)

        if comments_df.empty:
            return pd.DataFrame(), posts_df, comments_df

        # Clean comments
        comments_df = apply_cleaning(comments_df, column="body")

        # Run sentiment analysis
        sentiment_df = analyze(comments_df["body"].tolist(), engine=engine)

        # Merge results
        df = pd.concat(
            [comments_df.reset_index(drop=True), sentiment_df.reset_index(drop=True)],
            axis=1,
        )

        # Normalize prob column for VADER
        if "prob" not in df.columns and "compound" in df.columns:
            df["prob"] = df["compound"].abs()

        return df, posts_df, comments_df

    def _create_snapshot(
        self,
        df: pd.DataFrame,
        keyword: str,
        days_back: int,
        engine: str,
    ) -> SentimentSnapshot:
        """Create a sentiment snapshot from analyzed data."""
        now = datetime.now(timezone.utc)

        # Get prob column
        if "prob" in df.columns:
            probs = df["prob"]
        elif "compound" in df.columns:
            # Normalize VADER compound to 0-1
            probs = (df["compound"] + 1) / 2
        else:
            probs = pd.Series([0.5] * len(df))

        # Calculate sentiment distribution
        distribution = SentimentDistribution(
            mean=float(probs.mean()),
            median=float(probs.median()),
            std=float(probs.std()),
            min=float(probs.min()),
            max=float(probs.max()),
            p25=float(probs.quantile(0.25)),
            p75=float(probs.quantile(0.75)),
            positive_pct=self._calc_sentiment_pct(df, "positive"),
            neutral_pct=self._calc_sentiment_pct(df, "neutral"),
            negative_pct=self._calc_sentiment_pct(df, "negative"),
            sample_size=len(df),
        )

        # Get time range from data
        if "created" in df.columns:
            df["created"] = pd.to_datetime(df["created"], utc=True, errors="coerce")
            period_start = df["created"].min()
            period_end = df["created"].max()

            # Handle NaT
            if pd.isna(period_start):
                period_start = now - timedelta(days=days_back)
            if pd.isna(period_end):
                period_end = now
        else:
            period_start = now - timedelta(days=days_back)
            period_end = now

        # Ensure timezone-aware
        if period_start.tzinfo is None:
            period_start = period_start.replace(tzinfo=timezone.utc)
        if period_end.tzinfo is None:
            period_end = period_end.replace(tzinfo=timezone.utc)

        # Top subreddits
        if "subreddit" in df.columns:
            top_subs = df["subreddit"].value_counts().head(10).items()
            top_subreddits = [(str(s), int(c)) for s, c in top_subs]
        else:
            top_subreddits = []

        return SentimentSnapshot(
            keyword=keyword,
            measured_at=now,
            period_start=period_start,
            period_end=period_end,
            distribution=distribution,
            volume=len(df),
            unique_authors=df["author"].nunique() if "author" in df.columns else 0,
            unique_subreddits=df["subreddit"].nunique() if "subreddit" in df.columns else 0,
            top_subreddits=top_subreddits,
            engine=engine,
            metadata={"days_back": days_back},
        )

    def _calc_sentiment_pct(self, df: pd.DataFrame, sentiment: str) -> float:
        """Calculate percentage for a sentiment label."""
        if len(df) == 0:
            return 0.0

        if "sentiment" in df.columns:
            # HF engine
            count = (df["sentiment"].str.lower() == sentiment).sum()
        elif "compound" in df.columns:
            # VADER
            if sentiment == "positive":
                count = (df["compound"] > 0.05).sum()
            elif sentiment == "negative":
                count = (df["compound"] < -0.05).sum()
            else:
                count = ((df["compound"] >= -0.05) & (df["compound"] <= 0.05)).sum()
        else:
            return 33.33

        return float(count / len(df) * 100)

    def _create_metric_events(
        self,
        keyword: str,
        kpi_values: dict,
    ) -> list[MetricEvent]:
        """Create metric events from KPI values."""
        now = datetime.now(timezone.utc)
        events = []

        for metric_name, value in kpi_values.items():
            events.append(
                MetricEvent(
                    keyword=keyword,
                    metric_name=metric_name,
                    value=value,
                    measured_at=now,
                )
            )

        return events

    def _identify_opportunities(
        self,
        posts_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        keyword: str,
        snapshot: SentimentSnapshot,
        with_llm: bool,
    ) -> list[ContentOpportunity]:
        """Identify content opportunities."""
        llm_opportunities = None

        # Try LLM-based opportunity identification
        if with_llm and self.llm_client and not comments_df.empty:
            try:
                # Sort by score and get top comments
                if "score" in comments_df.columns:
                    top_comments = (
                        comments_df.nlargest(75, "score")["body"]
                        .dropna()
                        .tolist()
                    )
                else:
                    top_comments = comments_df["body"].dropna().head(75).tolist()

                llm_opportunities = self.llm_client.identify_opportunities(
                    comments=top_comments,
                    keyword=keyword,
                    positive_pct=snapshot.distribution.positive_pct,
                    neutral_pct=snapshot.distribution.neutral_pct,
                    negative_pct=snapshot.distribution.negative_pct,
                )
            except Exception as e:
                logger.warning(f"LLM opportunity identification failed: {e}")

        # Score opportunities (with LLM results if available)
        opportunities = self._opportunity_scorer.score_dataframe(
            posts_df=posts_df,
            comments_df=comments_df,
            keyword=keyword,
            llm_opportunities=llm_opportunities,
        )

        return opportunities

    def generate_insights(
        self,
        keyword: str,
        days_back: int = 7,
    ) -> Optional[str]:
        """
        Generate LLM-powered insights for a keyword.

        Args:
            keyword: Search keyword
            days_back: Number of days to analyze

        Returns:
            Insights text, or None if LLM not available
        """
        if not self.llm_client:
            return None

        # Get recent snapshot
        snapshot = self.store.get_latest_snapshot(keyword)
        if not snapshot:
            return None

        # Run ETL to get top comments
        df, _, _ = self._run_etl(keyword, days_back, limit=500, engine="vader")
        if df.empty:
            return None

        # Get top positive and negative comments
        if "score" in df.columns:
            df_sorted = df.sort_values("score", ascending=False)
        else:
            df_sorted = df

        if "prob" in df.columns:
            top_positive = (
                df_sorted.nlargest(10, "prob")["body"].dropna().tolist()
            )
            top_negative = (
                df_sorted.nsmallest(10, "prob")["body"].dropna().tolist()
            )
        else:
            top_positive = df_sorted.head(10)["body"].dropna().tolist()
            top_negative = df_sorted.tail(10)["body"].dropna().tolist()

        try:
            return self.llm_client.generate_insights(
                snapshot=snapshot,
                top_positive=top_positive,
                top_negative=top_negative,
            )
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return None


def run_daily_aggregation(
    keyword: str,
    days_back: int = 1,
    limit: int = 1000,
    engine: Literal["vader", "hf"] = "vader",
) -> dict:
    """
    Convenience function to run daily aggregation.

    This is the main entry point for scheduled jobs.

    Args:
        keyword: Search keyword
        days_back: Number of days to analyze
        limit: Maximum posts to fetch
        engine: Sentiment engine

    Returns:
        Aggregation results dict
    """
    aggregator = Aggregator()
    return aggregator.run_aggregation(
        keyword=keyword,
        days_back=days_back,
        limit=limit,
        engine=engine,
    )
