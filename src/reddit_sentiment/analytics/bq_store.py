# src/reddit_sentiment/analytics/bq_store.py
"""
BigQuery storage layer for sentiment analytics.

Handles persistence and querying of:
- Sentiment snapshots (daily aggregates)
- Metric events (time-series KPIs)
- Content opportunities
- Alerts
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date, timezone
from typing import Iterator, Optional

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from .models import (
    SentimentSnapshot,
    SentimentDistribution,
    MetricEvent,
    ContentOpportunity,
    Alert,
    TrackedKeyword,
    CompetitorSnapshot,
    CompetitorAnalysis,
)

logger = logging.getLogger(__name__)


# BigQuery table schemas
SNAPSHOTS_SCHEMA = [
    bigquery.SchemaField("keyword", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("measured_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("period_start", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("period_end", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("mean_sentiment", "FLOAT64"),
    bigquery.SchemaField("median_sentiment", "FLOAT64"),
    bigquery.SchemaField("std_sentiment", "FLOAT64"),
    bigquery.SchemaField("min_sentiment", "FLOAT64"),
    bigquery.SchemaField("max_sentiment", "FLOAT64"),
    bigquery.SchemaField("p25_sentiment", "FLOAT64"),
    bigquery.SchemaField("p75_sentiment", "FLOAT64"),
    bigquery.SchemaField("positive_pct", "FLOAT64"),
    bigquery.SchemaField("neutral_pct", "FLOAT64"),
    bigquery.SchemaField("negative_pct", "FLOAT64"),
    bigquery.SchemaField("volume", "INT64"),
    bigquery.SchemaField("unique_authors", "INT64"),
    bigquery.SchemaField("unique_subreddits", "INT64"),
    bigquery.SchemaField("top_subreddits", "STRING"),  # JSON array
    bigquery.SchemaField("engine", "STRING"),
    bigquery.SchemaField("metadata", "STRING"),  # JSON object
]

METRIC_EVENTS_SCHEMA = [
    bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("keyword", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("metric_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("value", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("measured_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("dimensions", "STRING"),  # JSON object
    bigquery.SchemaField("metadata", "STRING"),  # JSON object
]

OPPORTUNITIES_SCHEMA = [
    bigquery.SchemaField("keyword", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("topic", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("detected_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("engagement_score", "FLOAT64"),
    bigquery.SchemaField("competition_score", "FLOAT64"),
    bigquery.SchemaField("opportunity_score", "FLOAT64"),
    bigquery.SchemaField("evidence", "STRING"),  # JSON object
    bigquery.SchemaField("subreddits", "STRING"),  # JSON array
    bigquery.SchemaField("recommended_action", "STRING"),
]

ALERTS_SCHEMA = [
    bigquery.SchemaField("alert_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("keyword", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("metric_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("current_value", "FLOAT64"),
    bigquery.SchemaField("threshold_value", "FLOAT64"),
    bigquery.SchemaField("severity", "STRING"),
    bigquery.SchemaField("triggered_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("message", "STRING"),
    bigquery.SchemaField("acknowledged", "BOOL"),
]

TRACKED_KEYWORDS_SCHEMA = [
    bigquery.SchemaField("keyword", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("schedule", "STRING"),
    bigquery.SchemaField("last_run", "TIMESTAMP"),
    bigquery.SchemaField("next_run", "TIMESTAMP"),
    bigquery.SchemaField("enabled", "BOOL"),
    bigquery.SchemaField("config", "STRING"),  # JSON object
]

COMPETITOR_SNAPSHOTS_SCHEMA = [
    bigquery.SchemaField("snapshot_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("client_id", "STRING"),  # Multi-tenant support
    bigquery.SchemaField("primary_brand", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("measured_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("period_start", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("period_end", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("period_days", "INT64"),
    bigquery.SchemaField("posts_analyzed", "INT64"),
    bigquery.SchemaField("comments_analyzed", "INT64"),
    bigquery.SchemaField("primary_sentiment", "FLOAT64"),
    bigquery.SchemaField("primary_positive_pct", "FLOAT64"),
    bigquery.SchemaField("primary_neutral_pct", "FLOAT64"),
    bigquery.SchemaField("primary_negative_pct", "FLOAT64"),
    bigquery.SchemaField("competitors", "STRING"),  # JSON array of CompetitorAnalysis
    bigquery.SchemaField("total_competitor_mentions", "INT64"),
    bigquery.SchemaField("threats", "STRING"),  # JSON array
    bigquery.SchemaField("opportunities", "STRING"),  # JSON array
    bigquery.SchemaField("top_pain_points", "STRING"),  # JSON array
    # LLM-generated deep analysis fields
    bigquery.SchemaField("llm_executive_summary", "STRING"),
    bigquery.SchemaField("llm_themes", "STRING"),  # JSON array
    bigquery.SchemaField("llm_unanswered_questions", "STRING"),  # JSON array
    bigquery.SchemaField("llm_competitive_insights", "STRING"),  # JSON array
    bigquery.SchemaField("llm_recommendations", "STRING"),  # JSON array
    bigquery.SchemaField("llm_risk_signals", "STRING"),  # JSON array
    bigquery.SchemaField("metadata", "STRING"),  # JSON object
]


class BigQueryStore:
    """BigQuery storage for sentiment analytics data."""

    def __init__(
        self,
        project: str,
        dataset: str = "reddit_sentiment",
        location: str = "US",
    ):
        """
        Initialize BigQuery store.

        Args:
            project: GCP project ID
            dataset: BigQuery dataset name
            location: Dataset location (default: US)
        """
        self.project = project
        self.dataset_id = dataset
        self.location = location
        self.client = bigquery.Client(project=project)
        self._ensure_dataset()

    def _ensure_dataset(self) -> None:
        """Create dataset if it doesn't exist."""
        dataset_ref = f"{self.project}.{self.dataset_id}"
        try:
            self.client.get_dataset(dataset_ref)
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = self.location
            self.client.create_dataset(dataset)
            logger.info(f"Created dataset {dataset_ref}")

    def _ensure_table(self, table_name: str, schema: list) -> str:
        """Create table if it doesn't exist, return full table ID."""
        table_id = f"{self.project}.{self.dataset_id}.{table_name}"
        try:
            self.client.get_table(table_id)
        except NotFound:
            table = bigquery.Table(table_id, schema=schema)
            self.client.create_table(table)
            logger.info(f"Created table {table_id}")
        return table_id

    # ─────────────────────────────────────────────
    # Snapshots
    # ─────────────────────────────────────────────

    def save_snapshot(self, snapshot: SentimentSnapshot) -> None:
        """Save a sentiment snapshot to BigQuery."""
        table_id = self._ensure_table("snapshots", SNAPSHOTS_SCHEMA)

        row = {
            "keyword": snapshot.keyword,
            "measured_at": snapshot.measured_at.isoformat(),
            "period_start": snapshot.period_start.isoformat(),
            "period_end": snapshot.period_end.isoformat(),
            "mean_sentiment": snapshot.distribution.mean,
            "median_sentiment": snapshot.distribution.median,
            "std_sentiment": snapshot.distribution.std,
            "min_sentiment": snapshot.distribution.min,
            "max_sentiment": snapshot.distribution.max,
            "p25_sentiment": snapshot.distribution.p25,
            "p75_sentiment": snapshot.distribution.p75,
            "positive_pct": snapshot.distribution.positive_pct,
            "neutral_pct": snapshot.distribution.neutral_pct,
            "negative_pct": snapshot.distribution.negative_pct,
            "volume": snapshot.volume,
            "unique_authors": snapshot.unique_authors,
            "unique_subreddits": snapshot.unique_subreddits,
            "top_subreddits": json.dumps(snapshot.top_subreddits),
            "engine": snapshot.engine,
            "metadata": json.dumps(snapshot.metadata),
        }

        errors = self.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"Failed to insert snapshot: {errors}")

    def load_snapshots(
        self,
        keyword: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100,
    ) -> list[SentimentSnapshot]:
        """Load snapshots for a keyword within a date range."""
        table_id = self._ensure_table("snapshots", SNAPSHOTS_SCHEMA)

        query = f"""
            SELECT *
            FROM `{table_id}`
            WHERE keyword = @keyword
        """
        params = [bigquery.ScalarQueryParameter("keyword", "STRING", keyword)]

        if start_date:
            query += " AND DATE(measured_at) >= @start_date"
            params.append(
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date)
            )

        if end_date:
            query += " AND DATE(measured_at) <= @end_date"
            params.append(bigquery.ScalarQueryParameter("end_date", "DATE", end_date))

        query += " ORDER BY measured_at DESC LIMIT @limit"
        params.append(bigquery.ScalarQueryParameter("limit", "INT64", max(1, min(limit, 10000))))

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        results = self.client.query(query, job_config=job_config).result()

        snapshots = []
        for row in results:
            distribution = SentimentDistribution(
                mean=row.mean_sentiment or 0,
                median=row.median_sentiment or 0,
                std=row.std_sentiment or 0,
                min=row.min_sentiment or 0,
                max=row.max_sentiment or 1,
                p25=row.p25_sentiment or 0,
                p75=row.p75_sentiment or 0,
                positive_pct=row.positive_pct or 0,
                neutral_pct=row.neutral_pct or 0,
                negative_pct=row.negative_pct or 0,
                sample_size=row.volume or 0,
            )

            snapshot = SentimentSnapshot(
                keyword=row.keyword,
                measured_at=row.measured_at.replace(tzinfo=timezone.utc),
                period_start=row.period_start.replace(tzinfo=timezone.utc),
                period_end=row.period_end.replace(tzinfo=timezone.utc),
                distribution=distribution,
                volume=row.volume or 0,
                unique_authors=row.unique_authors or 0,
                unique_subreddits=row.unique_subreddits or 0,
                top_subreddits=json.loads(row.top_subreddits or "[]"),
                engine=row.engine,
                metadata=json.loads(row.metadata or "{}"),
            )
            snapshots.append(snapshot)

        return snapshots

    def get_latest_snapshot(self, keyword: str) -> Optional[SentimentSnapshot]:
        """Get the most recent snapshot for a keyword."""
        snapshots = self.load_snapshots(keyword, limit=1)
        return snapshots[0] if snapshots else None

    # ─────────────────────────────────────────────
    # Metric Events
    # ─────────────────────────────────────────────

    def save_metric_event(self, event: MetricEvent) -> None:
        """Save a metric event to BigQuery."""
        table_id = self._ensure_table("metric_events", METRIC_EVENTS_SCHEMA)

        row = {
            "event_id": event.event_id,
            "keyword": event.keyword,
            "metric_name": event.metric_name,
            "value": event.value,
            "measured_at": event.measured_at.isoformat(),
            "dimensions": json.dumps(event.dimensions),
            "metadata": json.dumps(event.metadata),
        }

        errors = self.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"Failed to insert metric event: {errors}")

    def save_metric_events_batch(self, events: list[MetricEvent]) -> None:
        """Save multiple metric events in a batch."""
        if not events:
            return

        table_id = self._ensure_table("metric_events", METRIC_EVENTS_SCHEMA)

        rows = [
            {
                "event_id": e.event_id,
                "keyword": e.keyword,
                "metric_name": e.metric_name,
                "value": e.value,
                "measured_at": e.measured_at.isoformat(),
                "dimensions": json.dumps(e.dimensions),
                "metadata": json.dumps(e.metadata),
            }
            for e in events
        ]

        errors = self.client.insert_rows_json(table_id, rows)
        if errors:
            raise RuntimeError(f"Failed to insert metric events: {errors}")

    def query_metrics(
        self,
        keyword: str,
        metric_name: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        granularity: str = "day",
    ) -> list[dict]:
        """
        Query aggregated metrics for a keyword.

        Args:
            keyword: Search keyword
            metric_name: Optional metric filter
            start_date: Start of date range
            end_date: End of date range
            granularity: Aggregation level (hour, day, week, month)

        Returns:
            List of aggregated metric records
        """
        table_id = self._ensure_table("metric_events", METRIC_EVENTS_SCHEMA)

        # Date truncation based on granularity
        trunc_map = {
            "hour": "HOUR",
            "day": "DAY",
            "week": "WEEK",
            "month": "MONTH",
        }
        trunc = trunc_map.get(granularity, "DAY")

        query = f"""
            SELECT
                metric_name,
                DATE_TRUNC(measured_at, {trunc}) as period,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                COUNT(*) as sample_count
            FROM `{table_id}`
            WHERE keyword = @keyword
        """
        params = [bigquery.ScalarQueryParameter("keyword", "STRING", keyword)]

        if metric_name:
            query += " AND metric_name = @metric_name"
            params.append(
                bigquery.ScalarQueryParameter("metric_name", "STRING", metric_name)
            )

        if start_date:
            query += " AND DATE(measured_at) >= @start_date"
            params.append(
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date)
            )

        if end_date:
            query += " AND DATE(measured_at) <= @end_date"
            params.append(bigquery.ScalarQueryParameter("end_date", "DATE", end_date))

        query += f" GROUP BY metric_name, period ORDER BY period DESC"

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        results = self.client.query(query, job_config=job_config).result()

        return [
            {
                "metric_name": row.metric_name,
                "period": row.period.isoformat() if row.period else None,
                "avg_value": row.avg_value,
                "min_value": row.min_value,
                "max_value": row.max_value,
                "sample_count": row.sample_count,
            }
            for row in results
        ]

    # ─────────────────────────────────────────────
    # Opportunities
    # ─────────────────────────────────────────────

    def save_opportunity(self, opportunity: ContentOpportunity) -> None:
        """Save a content opportunity to BigQuery."""
        table_id = self._ensure_table("opportunities", OPPORTUNITIES_SCHEMA)

        row = {
            "keyword": opportunity.keyword,
            "topic": opportunity.topic,
            "detected_at": opportunity.detected_at.isoformat(),
            "engagement_score": opportunity.engagement_score,
            "competition_score": opportunity.competition_score,
            "opportunity_score": opportunity.opportunity_score,
            "evidence": json.dumps(opportunity.evidence),
            "subreddits": json.dumps(opportunity.subreddits),
            "recommended_action": opportunity.recommended_action,
        }

        errors = self.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"Failed to insert opportunity: {errors}")

    def query_opportunities(
        self,
        keyword: Optional[str] = None,
        min_score: float = 0.5,
        days_back: int = 30,
        limit: int = 20,
    ) -> list[ContentOpportunity]:
        """Query recent content opportunities."""
        table_id = self._ensure_table("opportunities", OPPORTUNITIES_SCHEMA)

        query = f"""
            SELECT *
            FROM `{table_id}`
            WHERE opportunity_score >= @min_score
              AND detected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days_back DAY)
        """
        params = [
            bigquery.ScalarQueryParameter("min_score", "FLOAT64", min_score),
            bigquery.ScalarQueryParameter("days_back", "INT64", days_back),
        ]

        if keyword:
            query += " AND keyword = @keyword"
            params.append(bigquery.ScalarQueryParameter("keyword", "STRING", keyword))

        query += " ORDER BY opportunity_score DESC LIMIT @limit"
        params.append(bigquery.ScalarQueryParameter("limit", "INT64", max(1, min(limit, 1000))))

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        results = self.client.query(query, job_config=job_config).result()

        return [
            ContentOpportunity(
                keyword=row.keyword,
                topic=row.topic,
                detected_at=row.detected_at.replace(tzinfo=timezone.utc),
                engagement_score=row.engagement_score or 0,
                competition_score=row.competition_score or 0,
                opportunity_score=row.opportunity_score or 0,
                evidence=json.loads(row.evidence or "{}"),
                subreddits=json.loads(row.subreddits or "[]"),
                recommended_action=row.recommended_action or "",
            )
            for row in results
        ]

    # ─────────────────────────────────────────────
    # Alerts
    # ─────────────────────────────────────────────

    def save_alert(self, alert: Alert) -> None:
        """Save an alert to BigQuery."""
        table_id = self._ensure_table("alerts", ALERTS_SCHEMA)

        row = {
            "alert_id": alert.alert_id,
            "keyword": alert.keyword,
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "severity": alert.severity,
            "triggered_at": alert.triggered_at.isoformat(),
            "message": alert.message,
            "acknowledged": alert.acknowledged,
        }

        errors = self.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"Failed to insert alert: {errors}")

    def query_alerts(
        self,
        keyword: Optional[str] = None,
        severity: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        days_back: int = 7,
    ) -> list[Alert]:
        """Query recent alerts."""
        table_id = self._ensure_table("alerts", ALERTS_SCHEMA)

        query = f"""
            SELECT *
            FROM `{table_id}`
            WHERE triggered_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days_back DAY)
        """
        params = [bigquery.ScalarQueryParameter("days_back", "INT64", days_back)]

        if keyword:
            query += " AND keyword = @keyword"
            params.append(bigquery.ScalarQueryParameter("keyword", "STRING", keyword))

        if severity:
            query += " AND severity = @severity"
            params.append(bigquery.ScalarQueryParameter("severity", "STRING", severity))

        if acknowledged is not None:
            query += " AND acknowledged = @acknowledged"
            params.append(
                bigquery.ScalarQueryParameter("acknowledged", "BOOL", acknowledged)
            )

        query += " ORDER BY triggered_at DESC"

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        results = self.client.query(query, job_config=job_config).result()

        return [
            Alert(
                alert_id=row.alert_id,
                keyword=row.keyword,
                metric_name=row.metric_name,
                current_value=row.current_value or 0,
                threshold_value=row.threshold_value or 0,
                severity=row.severity,
                triggered_at=row.triggered_at.replace(tzinfo=timezone.utc),
                message=row.message or "",
                acknowledged=row.acknowledged or False,
            )
            for row in results
        ]

    # ─────────────────────────────────────────────
    # Tracked Keywords
    # ─────────────────────────────────────────────

    def save_tracked_keyword(self, tracked: TrackedKeyword) -> None:
        """Save or update a tracked keyword."""
        table_id = self._ensure_table("tracked_keywords", TRACKED_KEYWORDS_SCHEMA)

        # Delete existing entry first (upsert pattern)
        delete_query = f"""
            DELETE FROM `{table_id}`
            WHERE keyword = @keyword
        """
        params = [bigquery.ScalarQueryParameter("keyword", "STRING", tracked.keyword)]
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        self.client.query(delete_query, job_config=job_config).result()

        # Insert new entry
        row = {
            "keyword": tracked.keyword,
            "schedule": tracked.schedule,
            "last_run": tracked.last_run.isoformat() if tracked.last_run else None,
            "next_run": tracked.next_run.isoformat() if tracked.next_run else None,
            "enabled": tracked.enabled,
            "config": json.dumps(tracked.config),
        }

        errors = self.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"Failed to insert tracked keyword: {errors}")

    def get_tracked_keywords(self, enabled_only: bool = True) -> list[TrackedKeyword]:
        """Get all tracked keywords."""
        table_id = self._ensure_table("tracked_keywords", TRACKED_KEYWORDS_SCHEMA)

        query = f"SELECT * FROM `{table_id}`"
        if enabled_only:
            query += " WHERE enabled = TRUE"

        results = self.client.query(query).result()

        return [
            TrackedKeyword(
                keyword=row.keyword,
                schedule=row.schedule or "0 6 * * *",
                last_run=(
                    row.last_run.replace(tzinfo=timezone.utc) if row.last_run else None
                ),
                next_run=(
                    row.next_run.replace(tzinfo=timezone.utc) if row.next_run else None
                ),
                enabled=row.enabled if row.enabled is not None else True,
                config=json.loads(row.config or "{}"),
            )
            for row in results
        ]

    def delete_tracked_keyword(self, keyword: str) -> None:
        """Remove a keyword from tracking."""
        table_id = self._ensure_table("tracked_keywords", TRACKED_KEYWORDS_SCHEMA)

        query = f"""
            DELETE FROM `{table_id}`
            WHERE keyword = @keyword
        """
        params = [bigquery.ScalarQueryParameter("keyword", "STRING", keyword)]
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        self.client.query(query, job_config=job_config).result()

    # ─────────────────────────────────────────────
    # Summary / Dashboard
    # ─────────────────────────────────────────────

    def get_keywords_summary(self) -> list[dict]:
        """Get summary statistics for all tracked keywords."""
        snapshots_table = self._ensure_table("snapshots", SNAPSHOTS_SCHEMA)
        alerts_table = self._ensure_table("alerts", ALERTS_SCHEMA)

        query = f"""
            WITH latest_snapshots AS (
                SELECT
                    keyword,
                    mean_sentiment,
                    volume,
                    measured_at,
                    ROW_NUMBER() OVER (PARTITION BY keyword ORDER BY measured_at DESC) as rn
                FROM `{snapshots_table}`
            ),
            recent_alerts AS (
                SELECT
                    keyword,
                    COUNT(*) as alert_count
                FROM `{alerts_table}`
                WHERE acknowledged = FALSE
                  AND triggered_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
                GROUP BY keyword
            )
            SELECT
                ls.keyword,
                ls.mean_sentiment as latest_sentiment,
                ls.volume as latest_volume,
                ls.measured_at as last_updated,
                COALESCE(ra.alert_count, 0) as active_alerts
            FROM latest_snapshots ls
            LEFT JOIN recent_alerts ra ON ls.keyword = ra.keyword
            WHERE ls.rn = 1
            ORDER BY ls.keyword
        """

        results = self.client.query(query).result()

        return [
            {
                "keyword": row.keyword,
                "latest_sentiment": row.latest_sentiment,
                "latest_volume": row.latest_volume,
                "last_updated": (
                    row.last_updated.isoformat() if row.last_updated else None
                ),
                "active_alerts": row.active_alerts,
            }
            for row in results
        ]

    # ─────────────────────────────────────────────
    # Competitor Snapshots
    # ─────────────────────────────────────────────

    def save_competitor_snapshot(self, snapshot: CompetitorSnapshot) -> None:
        """Save a competitive intelligence snapshot to BigQuery."""
        table_id = self._ensure_table("competitor_snapshots", COMPETITOR_SNAPSHOTS_SCHEMA)

        # Convert competitors to JSON-serializable format
        competitors_data = [
            {
                "competitor": c.competitor,
                "mention_count": c.mention_count,
                "avg_sentiment": c.avg_sentiment,
                "positive_mentions": c.positive_mentions,
                "negative_mentions": c.negative_mentions,
                "neutral_mentions": c.neutral_mentions,
                "better_at": c.better_at,
                "worse_at": c.worse_at,
                "switch_to_count": c.switch_to_count,
                "switch_from_count": c.switch_from_count,
                "top_subreddits": c.top_subreddits,
                "sample_mentions": c.sample_mentions,
            }
            for c in snapshot.competitors
        ]

        row = {
            "snapshot_id": snapshot.snapshot_id,
            "client_id": snapshot.client_id,
            "primary_brand": snapshot.primary_brand,
            "measured_at": snapshot.measured_at.isoformat(),
            "period_start": snapshot.period_start.isoformat(),
            "period_end": snapshot.period_end.isoformat(),
            "period_days": snapshot.period_days,
            "posts_analyzed": snapshot.posts_analyzed,
            "comments_analyzed": snapshot.comments_analyzed,
            "primary_sentiment": snapshot.primary_sentiment,
            "primary_positive_pct": snapshot.primary_positive_pct,
            "primary_neutral_pct": snapshot.primary_neutral_pct,
            "primary_negative_pct": snapshot.primary_negative_pct,
            "competitors": json.dumps(competitors_data),
            "total_competitor_mentions": snapshot.total_competitor_mentions,
            "threats": json.dumps(snapshot.threats),
            "opportunities": json.dumps(snapshot.opportunities),
            "top_pain_points": json.dumps(snapshot.top_pain_points),
            # LLM-generated fields
            "llm_executive_summary": snapshot.llm_executive_summary,
            "llm_themes": json.dumps(snapshot.llm_themes),
            "llm_unanswered_questions": json.dumps(snapshot.llm_unanswered_questions),
            "llm_competitive_insights": json.dumps(snapshot.llm_competitive_insights),
            "llm_recommendations": json.dumps(snapshot.llm_recommendations),
            "llm_risk_signals": json.dumps(snapshot.llm_risk_signals),
            "metadata": json.dumps(snapshot.metadata),
        }

        errors = self.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"Failed to insert competitor snapshot: {errors}")

        client_info = f" (client: {snapshot.client_id})" if snapshot.client_id else ""
        logger.info(f"Saved competitor snapshot {snapshot.snapshot_id} for {snapshot.primary_brand}{client_info}")

    def load_competitor_snapshots(
        self,
        primary_brand: Optional[str] = None,
        client_id: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 12,  # Default to 12 months of monthly snapshots
    ) -> list[CompetitorSnapshot]:
        """
        Load competitor snapshots within a date range.

        Args:
            primary_brand: Filter by brand name (optional if client_id provided)
            client_id: Filter by client ID for multi-tenant queries
            start_date: Start of date range
            end_date: End of date range
            limit: Maximum snapshots to return

        Returns:
            List of CompetitorSnapshot objects
        """
        table_id = self._ensure_table("competitor_snapshots", COMPETITOR_SNAPSHOTS_SCHEMA)

        query = f"""
            SELECT *
            FROM `{table_id}`
            WHERE 1=1
        """
        params = []

        if client_id:
            query += " AND client_id = @client_id"
            params.append(bigquery.ScalarQueryParameter("client_id", "STRING", client_id))

        if primary_brand:
            query += " AND primary_brand = @primary_brand"
            params.append(bigquery.ScalarQueryParameter("primary_brand", "STRING", primary_brand))

        if start_date:
            query += " AND DATE(measured_at) >= @start_date"
            params.append(
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date)
            )

        if end_date:
            query += " AND DATE(measured_at) <= @end_date"
            params.append(bigquery.ScalarQueryParameter("end_date", "DATE", end_date))

        query += " ORDER BY measured_at DESC LIMIT @limit"
        params.append(bigquery.ScalarQueryParameter("limit", "INT64", max(1, min(limit, 10000))))

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        results = self.client.query(query, job_config=job_config).result()

        snapshots = []
        for row in results:
            # Parse competitors from JSON
            competitors_data = json.loads(row.competitors or "[]")
            competitors = [
                CompetitorAnalysis(
                    competitor=c["competitor"],
                    mention_count=c["mention_count"],
                    avg_sentiment=c["avg_sentiment"],
                    positive_mentions=c.get("positive_mentions", 0),
                    negative_mentions=c.get("negative_mentions", 0),
                    neutral_mentions=c.get("neutral_mentions", 0),
                    better_at=c.get("better_at", []),
                    worse_at=c.get("worse_at", []),
                    switch_to_count=c.get("switch_to_count", 0),
                    switch_from_count=c.get("switch_from_count", 0),
                    top_subreddits=c.get("top_subreddits", []),
                    sample_mentions=c.get("sample_mentions", []),
                )
                for c in competitors_data
            ]

            snapshot = CompetitorSnapshot(
                snapshot_id=row.snapshot_id,
                client_id=getattr(row, 'client_id', None),
                primary_brand=row.primary_brand,
                measured_at=row.measured_at.replace(tzinfo=timezone.utc),
                period_start=row.period_start.replace(tzinfo=timezone.utc),
                period_end=row.period_end.replace(tzinfo=timezone.utc),
                period_days=row.period_days or 30,
                posts_analyzed=row.posts_analyzed or 0,
                comments_analyzed=row.comments_analyzed or 0,
                primary_sentiment=row.primary_sentiment or 0,
                primary_positive_pct=row.primary_positive_pct or 0,
                primary_neutral_pct=row.primary_neutral_pct or 0,
                primary_negative_pct=row.primary_negative_pct or 0,
                competitors=competitors,
                total_competitor_mentions=row.total_competitor_mentions or 0,
                threats=json.loads(row.threats or "[]"),
                opportunities=json.loads(row.opportunities or "[]"),
                top_pain_points=json.loads(row.top_pain_points or "[]"),
                # LLM-generated fields
                llm_executive_summary=getattr(row, 'llm_executive_summary', None),
                llm_themes=json.loads(getattr(row, 'llm_themes', None) or "[]"),
                llm_unanswered_questions=json.loads(getattr(row, 'llm_unanswered_questions', None) or "[]"),
                llm_competitive_insights=json.loads(getattr(row, 'llm_competitive_insights', None) or "[]"),
                llm_recommendations=json.loads(getattr(row, 'llm_recommendations', None) or "[]"),
                llm_risk_signals=json.loads(getattr(row, 'llm_risk_signals', None) or "[]"),
                metadata=json.loads(row.metadata or "{}"),
            )
            snapshots.append(snapshot)

        return snapshots

    def get_latest_competitor_snapshot(
        self,
        primary_brand: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> Optional[CompetitorSnapshot]:
        """Get the most recent competitor snapshot for a brand or client."""
        snapshots = self.load_competitor_snapshots(
            primary_brand=primary_brand,
            client_id=client_id,
            limit=1,
        )
        return snapshots[0] if snapshots else None
