# src/reddit_sentiment/metrics/kpi_registry.py
"""
KPI definitions and registry for sentiment metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any, Optional

import pandas as pd


class AggregationType(Enum):
    """Aggregation methods for KPIs."""

    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    LAST = "last"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


@dataclass
class KPIDefinition:
    """Definition of a Key Performance Indicator."""

    name: str
    description: str
    unit: str
    aggregation: AggregationType
    calculator: Callable[[pd.DataFrame], float]
    thresholds: dict = field(default_factory=dict)  # {"warning": 0.7, "critical": 0.5}
    dimensions: list[str] = field(default_factory=list)  # Dimensional breakdowns
    higher_is_better: bool = True  # For determining alert direction


class KPIRegistry:
    """Registry of all tracked KPIs."""

    _kpis: dict[str, KPIDefinition] = {}

    @classmethod
    def register(cls, kpi: KPIDefinition) -> None:
        """Register a KPI definition."""
        cls._kpis[kpi.name] = kpi

    @classmethod
    def get(cls, name: str) -> Optional[KPIDefinition]:
        """Get a KPI by name."""
        return cls._kpis.get(name)

    @classmethod
    def all(cls) -> list[KPIDefinition]:
        """Get all registered KPIs."""
        return list(cls._kpis.values())

    @classmethod
    def names(cls) -> list[str]:
        """Get all KPI names."""
        return list(cls._kpis.keys())

    @classmethod
    def calculate(cls, name: str, df: pd.DataFrame) -> Optional[float]:
        """Calculate a KPI value from a dataframe."""
        kpi = cls.get(name)
        if not kpi:
            return None
        try:
            return kpi.calculator(df)
        except Exception:
            return None

    @classmethod
    def calculate_all(cls, df: pd.DataFrame) -> dict[str, float]:
        """Calculate all KPIs from a dataframe."""
        results = {}
        for name in cls.names():
            value = cls.calculate(name, df)
            if value is not None:
                results[name] = value
        return results


# ─────────────────────────────────────────────
# KPI Calculator Functions
# ─────────────────────────────────────────────


def _calc_sentiment_mean(df: pd.DataFrame) -> float:
    """Calculate mean sentiment score."""
    if "prob" in df.columns:
        return float(df["prob"].mean())
    elif "compound" in df.columns:
        # VADER compound is -1 to 1, normalize to 0-1
        return float((df["compound"].mean() + 1) / 2)
    return 0.5


def _calc_sentiment_median(df: pd.DataFrame) -> float:
    """Calculate median sentiment score."""
    if "prob" in df.columns:
        return float(df["prob"].median())
    elif "compound" in df.columns:
        return float((df["compound"].median() + 1) / 2)
    return 0.5


def _calc_sentiment_std(df: pd.DataFrame) -> float:
    """Calculate sentiment standard deviation (volatility)."""
    if "prob" in df.columns:
        return float(df["prob"].std())
    elif "compound" in df.columns:
        return float(df["compound"].std())
    return 0.0


def _calc_volume(df: pd.DataFrame) -> float:
    """Calculate total comment volume."""
    return float(len(df))


def _calc_positive_ratio(df: pd.DataFrame) -> float:
    """Calculate percentage of positive sentiment."""
    if "sentiment" in df.columns:
        # HF engine output
        positive = (df["sentiment"].str.lower() == "positive").sum()
        return float(positive / len(df)) if len(df) > 0 else 0.0
    elif "compound" in df.columns:
        # VADER output
        positive = (df["compound"] > 0.05).sum()
        return float(positive / len(df)) if len(df) > 0 else 0.0
    return 0.0


def _calc_negative_ratio(df: pd.DataFrame) -> float:
    """Calculate percentage of negative sentiment."""
    if "sentiment" in df.columns:
        negative = (df["sentiment"].str.lower() == "negative").sum()
        return float(negative / len(df)) if len(df) > 0 else 0.0
    elif "compound" in df.columns:
        negative = (df["compound"] < -0.05).sum()
        return float(negative / len(df)) if len(df) > 0 else 0.0
    return 0.0


def _calc_neutral_ratio(df: pd.DataFrame) -> float:
    """Calculate percentage of neutral sentiment."""
    if "sentiment" in df.columns:
        neutral = (df["sentiment"].str.lower() == "neutral").sum()
        return float(neutral / len(df)) if len(df) > 0 else 0.0
    elif "compound" in df.columns:
        neutral = ((df["compound"] >= -0.05) & (df["compound"] <= 0.05)).sum()
        return float(neutral / len(df)) if len(df) > 0 else 0.0
    return 0.0


def _calc_engagement_velocity(df: pd.DataFrame) -> float:
    """Calculate comments per day."""
    if "created" not in df.columns or len(df) < 2:
        return float(len(df))

    try:
        df_sorted = df.sort_values("created")
        time_range = (df_sorted["created"].max() - df_sorted["created"].min()).days
        if time_range <= 0:
            return float(len(df))
        return float(len(df) / time_range)
    except Exception:
        return float(len(df))


def _calc_author_diversity(df: pd.DataFrame) -> float:
    """Calculate unique authors / total comments ratio."""
    if "author" not in df.columns or len(df) == 0:
        return 0.0
    unique = df["author"].nunique()
    return float(unique / len(df))


def _calc_subreddit_diversity(df: pd.DataFrame) -> float:
    """Calculate unique subreddits / total comments ratio."""
    if "subreddit" not in df.columns or len(df) == 0:
        return 0.0
    unique = df["subreddit"].nunique()
    return float(unique / len(df))


def _calc_avg_score(df: pd.DataFrame) -> float:
    """Calculate average Reddit score (upvotes)."""
    if "score" not in df.columns:
        return 0.0
    return float(df["score"].mean())


def _calc_engagement_quality(df: pd.DataFrame) -> float:
    """
    Calculate engagement quality score.
    Combines author diversity, score, and sentiment positivity.
    """
    author_div = _calc_author_diversity(df)
    positive_ratio = _calc_positive_ratio(df)

    # Normalize score (log scale for large values)
    avg_score = _calc_avg_score(df)
    score_factor = min(1.0, (avg_score + 1) / 100) if avg_score > 0 else 0.5

    # Weighted combination
    return float(0.3 * author_div + 0.3 * positive_ratio + 0.4 * score_factor)


# ─────────────────────────────────────────────
# Register Core KPIs
# ─────────────────────────────────────────────

KPIRegistry.register(
    KPIDefinition(
        name="sentiment_mean",
        description="Average sentiment score (0-1 scale)",
        unit="score",
        aggregation=AggregationType.MEAN,
        calculator=_calc_sentiment_mean,
        thresholds={"warning": 0.4, "critical": 0.3},
        higher_is_better=True,
    )
)

KPIRegistry.register(
    KPIDefinition(
        name="sentiment_median",
        description="Median sentiment score (0-1 scale)",
        unit="score",
        aggregation=AggregationType.MEAN,
        calculator=_calc_sentiment_median,
        thresholds={"warning": 0.4, "critical": 0.3},
        higher_is_better=True,
    )
)

KPIRegistry.register(
    KPIDefinition(
        name="sentiment_volatility",
        description="Standard deviation of sentiment scores",
        unit="score",
        aggregation=AggregationType.MEAN,
        calculator=_calc_sentiment_std,
        thresholds={"warning": 0.35, "critical": 0.4},
        higher_is_better=False,  # Lower volatility is better
    )
)

KPIRegistry.register(
    KPIDefinition(
        name="volume",
        description="Total number of comments analyzed",
        unit="count",
        aggregation=AggregationType.SUM,
        calculator=_calc_volume,
        thresholds={"warning": 10, "critical": 5},
        higher_is_better=True,
    )
)

KPIRegistry.register(
    KPIDefinition(
        name="positive_ratio",
        description="Percentage of positive sentiment comments",
        unit="percent",
        aggregation=AggregationType.MEAN,
        calculator=_calc_positive_ratio,
        thresholds={"warning": 0.3, "critical": 0.2},
        higher_is_better=True,
    )
)

KPIRegistry.register(
    KPIDefinition(
        name="negative_ratio",
        description="Percentage of negative sentiment comments",
        unit="percent",
        aggregation=AggregationType.MEAN,
        calculator=_calc_negative_ratio,
        thresholds={"warning": 0.4, "critical": 0.5},
        higher_is_better=False,  # Lower negative is better
    )
)

KPIRegistry.register(
    KPIDefinition(
        name="neutral_ratio",
        description="Percentage of neutral sentiment comments",
        unit="percent",
        aggregation=AggregationType.MEAN,
        calculator=_calc_neutral_ratio,
        higher_is_better=True,  # Neutral can be good (less polarization)
    )
)

KPIRegistry.register(
    KPIDefinition(
        name="engagement_velocity",
        description="Comments per day",
        unit="comments/day",
        aggregation=AggregationType.MEAN,
        calculator=_calc_engagement_velocity,
        dimensions=["subreddit"],
        higher_is_better=True,
    )
)

KPIRegistry.register(
    KPIDefinition(
        name="author_diversity",
        description="Unique authors / total comments ratio",
        unit="ratio",
        aggregation=AggregationType.MEAN,
        calculator=_calc_author_diversity,
        thresholds={"warning": 0.3, "critical": 0.2},
        higher_is_better=True,
    )
)

KPIRegistry.register(
    KPIDefinition(
        name="subreddit_diversity",
        description="Unique subreddits / total comments ratio",
        unit="ratio",
        aggregation=AggregationType.MEAN,
        calculator=_calc_subreddit_diversity,
        higher_is_better=True,
    )
)

KPIRegistry.register(
    KPIDefinition(
        name="avg_score",
        description="Average Reddit score (upvotes)",
        unit="points",
        aggregation=AggregationType.MEAN,
        calculator=_calc_avg_score,
        higher_is_better=True,
    )
)

KPIRegistry.register(
    KPIDefinition(
        name="engagement_quality",
        description="Combined quality score (diversity, positivity, engagement)",
        unit="score",
        aggregation=AggregationType.MEAN,
        calculator=_calc_engagement_quality,
        thresholds={"warning": 0.4, "critical": 0.3},
        higher_is_better=True,
    )
)
