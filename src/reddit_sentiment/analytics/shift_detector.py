# src/reddit_sentiment/analytics/shift_detector.py
"""
Sentiment shift detection using statistical analysis.

Detects statistically significant changes in sentiment between time periods.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy import stats

from .models import SentimentSnapshot, SentimentDistribution, ShiftResult

logger = logging.getLogger(__name__)


class ShiftDetector:
    """Detects statistically significant sentiment shifts."""

    # Statistical thresholds
    SIGNIFICANCE_THRESHOLD = 0.05  # p-value threshold for significance
    MIN_SAMPLE_SIZE = 30  # Minimum samples for reliable comparison
    MIN_CHANGE_PCT_DEFAULT = 5.0  # Minimum % change to consider "notable"

    def __init__(
        self,
        min_change_pct: float = MIN_CHANGE_PCT_DEFAULT,
        significance_level: float = SIGNIFICANCE_THRESHOLD,
    ):
        """
        Initialize shift detector.

        Args:
            min_change_pct: Minimum percentage change to consider notable
            significance_level: P-value threshold for statistical significance
        """
        self.min_change_pct = min_change_pct
        self.significance_level = significance_level

    def compare_snapshots(
        self,
        snapshot_a: SentimentSnapshot,
        snapshot_b: SentimentSnapshot,
    ) -> ShiftResult:
        """
        Compare two snapshots and detect significant shifts.

        Args:
            snapshot_a: Earlier snapshot (baseline)
            snapshot_b: Later snapshot (comparison)

        Returns:
            ShiftResult with comparison metrics and interpretation
        """
        if snapshot_a.keyword != snapshot_b.keyword:
            raise ValueError("Cannot compare snapshots for different keywords")

        dist_a = snapshot_a.distribution
        dist_b = snapshot_b.distribution

        # Calculate changes
        mean_change = dist_b.mean - dist_a.mean
        mean_change_pct = (
            (mean_change / abs(dist_a.mean)) * 100 if dist_a.mean != 0 else 0
        )

        volume_change = snapshot_b.volume - snapshot_a.volume
        volume_change_pct = (
            (volume_change / snapshot_a.volume) * 100 if snapshot_a.volume > 0 else 0
        )

        # Statistical significance test
        is_significant, p_value = self._test_significance(dist_a, dist_b)

        # Determine direction
        if abs(mean_change_pct) < self.min_change_pct:
            direction = "stable"
        elif mean_change > 0:
            direction = "up"
        else:
            direction = "down"

        # Generate interpretation
        interpretation = self._interpret_shift(
            mean_change_pct=mean_change_pct,
            volume_change_pct=volume_change_pct,
            direction=direction,
            is_significant=is_significant,
            dist_a=dist_a,
            dist_b=dist_b,
        )

        return ShiftResult(
            keyword=snapshot_a.keyword,
            period_a=(snapshot_a.period_start, snapshot_a.period_end),
            period_b=(snapshot_b.period_start, snapshot_b.period_end),
            mean_change=mean_change,
            mean_change_pct=mean_change_pct,
            volume_change=volume_change,
            volume_change_pct=volume_change_pct,
            is_significant=is_significant,
            p_value=p_value,
            confidence_level=1 - self.significance_level,
            shift_direction=direction,
            interpretation=interpretation,
        )

    def _test_significance(
        self,
        dist_a: SentimentDistribution,
        dist_b: SentimentDistribution,
    ) -> tuple[bool, float]:
        """
        Perform Welch's t-test for difference of means using summary statistics.

        This approximates the test when we only have summary stats (mean, std, n)
        rather than raw data.

        Args:
            dist_a: Distribution for period A
            dist_b: Distribution for period B

        Returns:
            Tuple of (is_significant, p_value)
        """
        n1 = dist_a.sample_size
        n2 = dist_b.sample_size
        m1 = dist_a.mean
        m2 = dist_b.mean
        s1 = dist_a.std
        s2 = dist_b.std

        # Check minimum sample size
        if n1 < self.MIN_SAMPLE_SIZE or n2 < self.MIN_SAMPLE_SIZE:
            logger.warning(
                f"Insufficient sample size for significance test "
                f"(n1={n1}, n2={n2}, min={self.MIN_SAMPLE_SIZE})"
            )
            return False, 1.0

        # Avoid division by zero
        if s1 == 0 and s2 == 0:
            # Both distributions have zero variance - can't determine significance
            return False, 1.0

        # Welch's t-statistic
        # t = (m1 - m2) / sqrt(s1^2/n1 + s2^2/n2)
        se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
        if se == 0:
            return False, 1.0

        t_stat = (m1 - m2) / se

        # Welch-Satterthwaite degrees of freedom
        # df = (s1^2/n1 + s2^2/n2)^2 / ((s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1))
        numerator = ((s1**2 / n1) + (s2**2 / n2)) ** 2
        denominator = (s1**4 / (n1**2 * (n1 - 1))) + (s2**4 / (n2**2 * (n2 - 1)))

        if denominator == 0:
            return False, 1.0

        df = numerator / denominator

        # Two-tailed p-value
        p_value = 2 * stats.t.sf(abs(t_stat), df)

        is_significant = p_value < self.significance_level
        return is_significant, float(p_value)

    def _interpret_shift(
        self,
        mean_change_pct: float,
        volume_change_pct: float,
        direction: str,
        is_significant: bool,
        dist_a: SentimentDistribution,
        dist_b: SentimentDistribution,
    ) -> str:
        """Generate human-readable interpretation of the shift."""
        if direction == "stable":
            base = f"Sentiment remained stable (change: {mean_change_pct:+.1f}%)"

            # Add context about volume changes
            if abs(volume_change_pct) > 50:
                vol_dir = "increased" if volume_change_pct > 0 else "decreased"
                base += f", though discussion volume {vol_dir} significantly ({volume_change_pct:+.0f}%)"

            return base

        # Significant vs slight
        significance = "significantly" if is_significant else "slightly"

        # Direction
        dir_word = "improved" if direction == "up" else "declined"

        # Base message
        message = f"Sentiment {significance} {dir_word} by {abs(mean_change_pct):.1f}%"

        # Add volume context
        if abs(volume_change_pct) > 20:
            vol_dir = "increased" if volume_change_pct > 0 else "decreased"
            message += f" with {vol_dir} engagement ({volume_change_pct:+.0f}%)"

        # Add distribution context
        if direction == "down" and dist_b.negative_pct > dist_a.negative_pct + 10:
            message += f". Negative sentiment rose from {dist_a.negative_pct:.0f}% to {dist_b.negative_pct:.0f}%"
        elif direction == "up" and dist_b.positive_pct > dist_a.positive_pct + 10:
            message += f". Positive sentiment rose from {dist_a.positive_pct:.0f}% to {dist_b.positive_pct:.0f}%"

        return message

    def detect_period_shifts(
        self,
        snapshots: list[SentimentSnapshot],
        period: str = "week",
    ) -> list[ShiftResult]:
        """
        Detect shifts between consecutive periods.

        Args:
            snapshots: List of snapshots (should be sorted by date)
            period: Period type ("day", "week", "month")

        Returns:
            List of ShiftResult for consecutive period comparisons
        """
        if len(snapshots) < 2:
            return []

        # Sort by period_end descending (most recent first)
        sorted_snapshots = sorted(
            snapshots, key=lambda s: s.period_end, reverse=True
        )

        shifts = []
        for i in range(len(sorted_snapshots) - 1):
            # Compare current to previous (more recent vs older)
            snapshot_b = sorted_snapshots[i]  # More recent
            snapshot_a = sorted_snapshots[i + 1]  # Older

            try:
                shift = self.compare_snapshots(snapshot_a, snapshot_b)
                shifts.append(shift)
            except Exception as e:
                logger.warning(f"Failed to compare snapshots: {e}")

        return shifts

    def aggregate_to_periods(
        self,
        snapshots: list[SentimentSnapshot],
        period: str = "week",
    ) -> list[SentimentSnapshot]:
        """
        Aggregate daily snapshots into weekly/monthly periods.

        Args:
            snapshots: List of daily snapshots
            period: Target period ("week" or "month")

        Returns:
            List of aggregated snapshots
        """
        if not snapshots or period == "day":
            return snapshots

        # Group snapshots by period
        from collections import defaultdict

        period_groups: dict = defaultdict(list)

        for snap in snapshots:
            if period == "week":
                # ISO week number
                key = snap.measured_at.strftime("%Y-W%W")
            else:  # month
                key = snap.measured_at.strftime("%Y-%m")

            period_groups[key].append(snap)

        # Aggregate each group
        aggregated = []
        for period_key, group in sorted(period_groups.items()):
            if not group:
                continue

            # Calculate weighted averages (weighted by volume)
            total_volume = sum(s.volume for s in group)
            if total_volume == 0:
                continue

            # Weighted mean sentiment
            weighted_mean = sum(
                s.distribution.mean * s.volume for s in group
            ) / total_volume

            # Combine distributions
            combined_dist = SentimentDistribution(
                mean=weighted_mean,
                median=np.median([s.distribution.median for s in group]),
                std=np.mean([s.distribution.std for s in group]),
                min=min(s.distribution.min for s in group),
                max=max(s.distribution.max for s in group),
                p25=np.percentile([s.distribution.p25 for s in group], 25),
                p75=np.percentile([s.distribution.p75 for s in group], 75),
                positive_pct=sum(
                    s.distribution.positive_pct * s.volume for s in group
                ) / total_volume,
                neutral_pct=sum(
                    s.distribution.neutral_pct * s.volume for s in group
                ) / total_volume,
                negative_pct=sum(
                    s.distribution.negative_pct * s.volume for s in group
                ) / total_volume,
                sample_size=total_volume,
            )

            # Combine top subreddits
            from collections import Counter

            sub_counts: Counter = Counter()
            for snap in group:
                for sub, count in snap.top_subreddits:
                    sub_counts[sub] += count

            aggregated.append(
                SentimentSnapshot(
                    keyword=group[0].keyword,
                    measured_at=max(s.measured_at for s in group),
                    period_start=min(s.period_start for s in group),
                    period_end=max(s.period_end for s in group),
                    distribution=combined_dist,
                    volume=total_volume,
                    unique_authors=sum(s.unique_authors for s in group),  # May have overlap
                    unique_subreddits=len(sub_counts),
                    top_subreddits=sub_counts.most_common(10),
                    engine=group[0].engine,
                    metadata={"aggregated_from": len(group), "period": period},
                )
            )

        return sorted(aggregated, key=lambda s: s.period_end, reverse=True)


def calculate_trend(
    snapshots: list[SentimentSnapshot],
    window: int = 7,
) -> str:
    """
    Calculate overall trend direction from recent snapshots.

    Args:
        snapshots: List of snapshots (most recent first)
        window: Number of snapshots to consider

    Returns:
        Trend direction: "rising", "falling", or "stable"
    """
    if len(snapshots) < 2:
        return "stable"

    # Get recent snapshots
    recent = sorted(snapshots, key=lambda s: s.measured_at, reverse=True)[:window]

    if len(recent) < 2:
        return "stable"

    # Calculate linear regression slope
    x = np.arange(len(recent))
    y = np.array([s.distribution.mean for s in recent])

    # Reverse so older is first (x=0 is oldest)
    y = y[::-1]

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Determine trend based on slope significance
    if abs(slope) < 0.01:
        return "stable"
    elif slope > 0:
        return "rising"
    else:
        return "falling"
