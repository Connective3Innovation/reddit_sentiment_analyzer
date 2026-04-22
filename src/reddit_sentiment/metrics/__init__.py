# src/reddit_sentiment/metrics/__init__.py
"""
Metrics module for KPI tracking and alerting.

Provides:
- KPI definitions and registry
- KPI calculators
- Alert thresholds and triggers
"""

from .kpi_registry import KPIRegistry, KPIDefinition, AggregationType
from .alerts import AlertChecker

__all__ = [
    "KPIRegistry",
    "KPIDefinition",
    "AggregationType",
    "AlertChecker",
]
