# src/reddit_sentiment/analytics/models.py
"""
Pydantic models for social search intelligence analytics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SentimentDistribution(BaseModel):
    """Statistical distribution summary for sentiment measurements."""

    mean: float = Field(..., description="Mean sentiment score")
    median: float = Field(..., description="Median sentiment score")
    std: float = Field(..., description="Standard deviation")
    min: float = Field(..., description="Minimum score")
    max: float = Field(..., description="Maximum score")
    p25: float = Field(..., description="25th percentile")
    p75: float = Field(..., description="75th percentile")
    positive_pct: float = Field(..., description="Percentage of positive sentiment")
    neutral_pct: float = Field(..., description="Percentage of neutral sentiment")
    negative_pct: float = Field(..., description="Percentage of negative sentiment")
    sample_size: int = Field(..., description="Number of samples in distribution")


class SentimentSnapshot(BaseModel):
    """Point-in-time sentiment measurement for a keyword."""

    keyword: str = Field(..., description="Search keyword")
    measured_at: datetime = Field(..., description="When snapshot was taken")
    period_start: datetime = Field(..., description="Start of measurement window")
    period_end: datetime = Field(..., description="End of measurement window")
    distribution: SentimentDistribution = Field(..., description="Sentiment statistics")
    volume: int = Field(..., description="Total comments analyzed")
    unique_authors: int = Field(..., description="Distinct comment authors")
    unique_subreddits: int = Field(..., description="Distinct subreddits")
    top_subreddits: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Top subreddits by volume (name, count)"
    )
    engine: Literal["hf", "vader"] = Field(..., description="Sentiment engine used")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ShiftResult(BaseModel):
    """Result of comparing sentiment between two time periods."""

    keyword: str = Field(..., description="Search keyword")
    period_a: tuple[datetime, datetime] = Field(..., description="First period (start, end)")
    period_b: tuple[datetime, datetime] = Field(..., description="Second period (start, end)")
    mean_change: float = Field(..., description="Absolute change in mean sentiment")
    mean_change_pct: float = Field(..., description="Percentage change in mean sentiment")
    volume_change: int = Field(..., description="Absolute change in volume")
    volume_change_pct: float = Field(..., description="Percentage change in volume")
    is_significant: bool = Field(..., description="Whether change is statistically significant")
    p_value: float = Field(..., description="P-value from statistical test")
    confidence_level: float = Field(default=0.95, description="Confidence level used")
    shift_direction: Literal["up", "down", "stable"] = Field(
        ...,
        description="Direction of sentiment shift"
    )
    interpretation: str = Field(..., description="Human-readable interpretation")


class Theme(BaseModel):
    """A discussion theme extracted from comments."""

    name: str = Field(..., description="Theme name or label")
    description: str = Field(default="", description="What users say about this theme")
    keywords: list[str] = Field(default_factory=list, description="Associated keywords")
    sentiment: str = Field(default="mixed", description="Theme sentiment: positive, negative, mixed")
    comment_count: int = Field(default=0, description="Number of comments in theme")
    sentiment_mean: float = Field(default=0.0, description="Average sentiment for theme")
    sample_comments: list[str] = Field(
        default_factory=list,
        description="Representative comment samples"
    )
    subreddits: list[str] = Field(
        default_factory=list,
        description="Subreddits where theme appears"
    )


class ContentOpportunity(BaseModel):
    """A detected content opportunity with scoring."""

    keyword: str = Field(..., description="Parent search keyword")
    topic: str = Field(..., description="Specific topic or question")
    detected_at: datetime = Field(..., description="When opportunity was detected")
    engagement_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Normalized engagement score (0-1)"
    )
    competition_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Competition level (lower = less competition)"
    )
    opportunity_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Combined opportunity score (0-1)"
    )
    evidence: dict = Field(
        default_factory=dict,
        description="Supporting data (samples, counts)"
    )
    subreddits: list[str] = Field(
        default_factory=list,
        description="Relevant subreddits"
    )
    recommended_action: str = Field(
        default="",
        description="Suggested action to take"
    )
    themes: list[Theme] = Field(
        default_factory=list,
        description="Related themes"
    )


class MetricEvent(BaseModel):
    """Generic KPI measurement event for time-series tracking."""

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique event identifier"
    )
    keyword: str = Field(..., description="Search keyword")
    metric_name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Metric value")
    measured_at: datetime = Field(..., description="When metric was measured")
    dimensions: dict = Field(
        default_factory=dict,
        description="Dimensional breakdowns (e.g., subreddit)"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional context"
    )


class Alert(BaseModel):
    """A triggered alert when KPI exceeds threshold."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique alert identifier"
    )
    keyword: str = Field(..., description="Affected keyword")
    metric_name: str = Field(..., description="Metric that triggered alert")
    current_value: float = Field(..., description="Current metric value")
    threshold_value: float = Field(..., description="Threshold that was exceeded")
    severity: Literal["warning", "critical"] = Field(..., description="Alert severity")
    triggered_at: datetime = Field(..., description="When alert was triggered")
    message: str = Field(..., description="Alert message")
    acknowledged: bool = Field(default=False, description="Whether alert was acknowledged")


class TrackedKeyword(BaseModel):
    """A keyword being tracked for scheduled monitoring."""

    keyword: str = Field(..., description="Keyword to track")
    schedule: str = Field(
        default="0 6 * * *",
        description="Cron expression for collection schedule"
    )
    last_run: Optional[datetime] = Field(
        default=None,
        description="Last successful run timestamp"
    )
    next_run: Optional[datetime] = Field(
        default=None,
        description="Next scheduled run timestamp"
    )
    enabled: bool = Field(default=True, description="Whether tracking is enabled")
    config: dict = Field(
        default_factory=dict,
        description="Additional configuration (days_back, limit, engine)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Competitive Intelligence Models
# ─────────────────────────────────────────────────────────────────────────────

class CompetitorAnalysis(BaseModel):
    """Analysis summary for a single competitor."""

    competitor: str = Field(..., description="Competitor name")
    mention_count: int = Field(..., description="Total mentions found")
    avg_sentiment: float = Field(..., description="Average sentiment of mentions (generic)")
    brand_sentiment_score: float = Field(
        default=0.0,
        description="ABSA sentiment TOWARD this competitor (-1 to +1)"
    )
    positive_mentions: int = Field(default=0, description="Count of positive mentions")
    negative_mentions: int = Field(default=0, description="Count of negative mentions")
    neutral_mentions: int = Field(default=0, description="Count of neutral mentions")
    better_at: list[str] = Field(
        default_factory=list,
        description="Areas where competitor excels"
    )
    worse_at: list[str] = Field(
        default_factory=list,
        description="Areas where competitor underperforms"
    )
    switch_to_count: int = Field(
        default=0,
        description="Users mentioning switching TO this competitor"
    )
    switch_from_count: int = Field(
        default=0,
        description="Users mentioning switching FROM this competitor"
    )
    top_subreddits: list[str] = Field(
        default_factory=list,
        description="Subreddits with most mentions"
    )
    sample_mentions: list[str] = Field(
        default_factory=list,
        description="Sample mention excerpts"
    )


class CompetitorSnapshot(BaseModel):
    """Point-in-time competitive intelligence snapshot."""

    snapshot_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique snapshot identifier"
    )
    client_id: Optional[str] = Field(
        default=None,
        description="Client identifier for multi-tenant support"
    )
    primary_brand: str = Field(..., description="Brand being analyzed")
    measured_at: datetime = Field(..., description="When analysis was performed")
    period_start: datetime = Field(..., description="Start of analysis window")
    period_end: datetime = Field(..., description="End of analysis window")
    period_days: int = Field(default=30, description="Number of days analyzed")
    posts_analyzed: int = Field(default=0, description="Number of posts fetched")
    comments_analyzed: int = Field(default=0, description="Total comments analyzed")

    # Primary brand metrics
    primary_sentiment: float = Field(..., description="Primary brand avg sentiment (generic)")
    primary_brand_sentiment: float = Field(
        default=0.0,
        description="ABSA sentiment TOWARD primary brand (-1 to +1)"
    )
    primary_positive_pct: float = Field(default=0, description="% positive comments")
    primary_neutral_pct: float = Field(default=0, description="% neutral comments")
    primary_negative_pct: float = Field(default=0, description="% negative comments")

    # Competitor data
    competitors: list[CompetitorAnalysis] = Field(
        default_factory=list,
        description="Analysis for each competitor"
    )
    total_competitor_mentions: int = Field(
        default=0,
        description="Total competitor mentions found"
    )

    # Strategic insights
    threats: list[str] = Field(
        default_factory=list,
        description="Identified competitive threats"
    )
    opportunities: list[str] = Field(
        default_factory=list,
        description="Identified opportunities"
    )

    # Pain points from negative comments
    top_pain_points: list[str] = Field(
        default_factory=list,
        description="Most negative comment excerpts"
    )

    # LLM-generated deep analysis (optional, requires OPENROUTER_API_KEY)
    llm_executive_summary: Optional[str] = Field(
        default=None,
        description="LLM-generated executive summary"
    )
    llm_themes: list[dict] = Field(
        default_factory=list,
        description="LLM-extracted discussion themes"
    )
    llm_unanswered_questions: list[dict] = Field(
        default_factory=list,
        description="Unanswered questions identified by LLM"
    )
    llm_competitive_insights: list[dict] = Field(
        default_factory=list,
        description="Deep competitive insights from LLM"
    )
    llm_recommendations: list[dict] = Field(
        default_factory=list,
        description="LLM actionable recommendations"
    )
    llm_risk_signals: list[dict] = Field(
        default_factory=list,
        description="Risk signals identified by LLM"
    )
    llm_brand_perception: dict = Field(
        default_factory=dict,
        description="LLM-extracted brand perception analysis"
    )

    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata"
    )
