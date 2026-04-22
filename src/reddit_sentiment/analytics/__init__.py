# src/reddit_sentiment/analytics/__init__.py
"""
Analytics module for social search intelligence.

Provides:
- Sentiment shift detection
- Content opportunity scoring
- KPI metrics and aggregation
"""

from .models import (
    SentimentDistribution,
    SentimentSnapshot,
    ShiftResult,
    ContentOpportunity,
    MetricEvent,
    Theme,
)

__all__ = [
    "SentimentDistribution",
    "SentimentSnapshot",
    "ShiftResult",
    "ContentOpportunity",
    "MetricEvent",
    "Theme",
]
