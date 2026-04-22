# src/reddit_sentiment/metrics/alerts.py
"""
Alert checking and triggering for KPI thresholds.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ..analytics.models import Alert, SentimentSnapshot
from .kpi_registry import KPIRegistry, KPIDefinition

logger = logging.getLogger(__name__)


class AlertChecker:
    """Checks KPIs against thresholds and generates alerts."""

    def __init__(self, custom_thresholds: Optional[dict] = None):
        """
        Initialize alert checker.

        Args:
            custom_thresholds: Optional override thresholds per KPI
                Format: {"kpi_name": {"warning": value, "critical": value}}
        """
        self.custom_thresholds = custom_thresholds or {}

    def _get_thresholds(self, kpi: KPIDefinition) -> dict:
        """Get thresholds for a KPI, with custom overrides."""
        base = kpi.thresholds.copy()
        if kpi.name in self.custom_thresholds:
            base.update(self.custom_thresholds[kpi.name])
        return base

    def check_value(
        self,
        kpi_name: str,
        value: float,
        keyword: str,
    ) -> Optional[Alert]:
        """
        Check a single KPI value against thresholds.

        Args:
            kpi_name: Name of the KPI
            value: Current value
            keyword: Associated keyword

        Returns:
            Alert if threshold exceeded, None otherwise
        """
        kpi = KPIRegistry.get(kpi_name)
        if not kpi:
            return None

        thresholds = self._get_thresholds(kpi)
        if not thresholds:
            return None

        warning = thresholds.get("warning")
        critical = thresholds.get("critical")

        # Determine if alert should trigger
        # For higher_is_better metrics, alert when value drops below threshold
        # For lower_is_better metrics, alert when value exceeds threshold
        severity = None
        threshold_value = None

        if kpi.higher_is_better:
            if critical is not None and value < critical:
                severity = "critical"
                threshold_value = critical
            elif warning is not None and value < warning:
                severity = "warning"
                threshold_value = warning
        else:
            if critical is not None and value > critical:
                severity = "critical"
                threshold_value = critical
            elif warning is not None and value > warning:
                severity = "warning"
                threshold_value = warning

        if severity is None:
            return None

        # Generate alert message
        direction = "below" if kpi.higher_is_better else "above"
        message = (
            f"{kpi.description} is {direction} {severity} threshold: "
            f"{value:.2f} vs {threshold_value:.2f}"
        )

        return Alert(
            keyword=keyword,
            metric_name=kpi_name,
            current_value=value,
            threshold_value=threshold_value,
            severity=severity,
            triggered_at=datetime.now(timezone.utc),
            message=message,
        )

    def check_dataframe(
        self,
        df: pd.DataFrame,
        keyword: str,
    ) -> list[Alert]:
        """
        Check all KPIs for a dataframe.

        Args:
            df: DataFrame with sentiment data
            keyword: Associated keyword

        Returns:
            List of triggered alerts
        """
        alerts = []

        for kpi in KPIRegistry.all():
            try:
                value = kpi.calculator(df)
                alert = self.check_value(kpi.name, value, keyword)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.warning(f"Failed to calculate KPI {kpi.name}: {e}")

        return alerts

    def check_snapshot(
        self,
        snapshot: SentimentSnapshot,
    ) -> list[Alert]:
        """
        Check KPIs from a sentiment snapshot.

        Args:
            snapshot: SentimentSnapshot to check

        Returns:
            List of triggered alerts
        """
        alerts = []
        dist = snapshot.distribution

        # Map snapshot fields to KPI values
        kpi_values = {
            "sentiment_mean": dist.mean,
            "sentiment_median": dist.median,
            "sentiment_volatility": dist.std,
            "volume": snapshot.volume,
            "positive_ratio": dist.positive_pct / 100,  # Convert % to ratio
            "negative_ratio": dist.negative_pct / 100,
            "neutral_ratio": dist.neutral_pct / 100,
            "author_diversity": (
                snapshot.unique_authors / snapshot.volume
                if snapshot.volume > 0
                else 0
            ),
            "subreddit_diversity": (
                snapshot.unique_subreddits / snapshot.volume
                if snapshot.volume > 0
                else 0
            ),
        }

        for kpi_name, value in kpi_values.items():
            alert = self.check_value(kpi_name, value, snapshot.keyword)
            if alert:
                alerts.append(alert)

        return alerts

    def compare_to_baseline(
        self,
        current: SentimentSnapshot,
        baseline: SentimentSnapshot,
        pct_threshold: float = 20.0,
    ) -> list[Alert]:
        """
        Generate alerts based on significant changes from baseline.

        Args:
            current: Current snapshot
            baseline: Baseline snapshot to compare against
            pct_threshold: Percentage change threshold for alerts

        Returns:
            List of change-based alerts
        """
        alerts = []

        # Compare key metrics
        comparisons = [
            ("sentiment_mean", current.distribution.mean, baseline.distribution.mean),
            ("volume", current.volume, baseline.volume),
            (
                "positive_ratio",
                current.distribution.positive_pct,
                baseline.distribution.positive_pct,
            ),
            (
                "negative_ratio",
                current.distribution.negative_pct,
                baseline.distribution.negative_pct,
            ),
        ]

        for metric_name, current_val, baseline_val in comparisons:
            if baseline_val == 0:
                continue

            pct_change = ((current_val - baseline_val) / abs(baseline_val)) * 100

            if abs(pct_change) >= pct_threshold:
                kpi = KPIRegistry.get(metric_name)
                direction = "increased" if pct_change > 0 else "decreased"

                # Determine if this is good or bad
                is_improvement = (kpi.higher_is_better and pct_change > 0) or (
                    not kpi.higher_is_better and pct_change < 0
                )

                severity = "warning" if not is_improvement else None

                if severity:
                    message = (
                        f"{kpi.description if kpi else metric_name} {direction} "
                        f"by {abs(pct_change):.1f}% from baseline "
                        f"({baseline_val:.2f} → {current_val:.2f})"
                    )

                    alerts.append(
                        Alert(
                            keyword=current.keyword,
                            metric_name=metric_name,
                            current_value=current_val,
                            threshold_value=baseline_val,
                            severity=severity,
                            triggered_at=datetime.now(timezone.utc),
                            message=message,
                        )
                    )

        return alerts


def get_alert_summary(alerts: list[Alert]) -> dict:
    """
    Generate summary statistics for a list of alerts.

    Args:
        alerts: List of alerts

    Returns:
        Summary dict with counts by severity and keyword
    """
    if not alerts:
        return {
            "total": 0,
            "by_severity": {"warning": 0, "critical": 0},
            "by_keyword": {},
            "unacknowledged": 0,
        }

    by_severity = {"warning": 0, "critical": 0}
    by_keyword: dict = {}

    for alert in alerts:
        by_severity[alert.severity] += 1

        if alert.keyword not in by_keyword:
            by_keyword[alert.keyword] = 0
        by_keyword[alert.keyword] += 1

    unacknowledged = sum(1 for a in alerts if not a.acknowledged)

    return {
        "total": len(alerts),
        "by_severity": by_severity,
        "by_keyword": by_keyword,
        "unacknowledged": unacknowledged,
    }
