# src/reddit_sentiment/export/deck_export.py
"""
Deck Export Module.

Exports CompetitorSnapshot data to presentation-ready CSV/text files
for use in PowerPoint decks. All files are bundled into a ZIP download.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from ..analytics.models import CompetitorSnapshot


def export_executive_summary(snapshot: "CompetitorSnapshot") -> str:
    """
    Export executive summary as plain text.

    Args:
        snapshot: CompetitorSnapshot with analysis results

    Returns:
        Plain text summary for copy-paste into deck
    """
    lines = []
    lines.append(f"Reddit Analysis: {snapshot.primary_brand}")
    lines.append(f"Analysis Period: {snapshot.period_start.strftime('%Y-%m-%d')} to {snapshot.period_end.strftime('%Y-%m-%d')}")
    lines.append(f"Posts Analyzed: {snapshot.posts_analyzed:,}")
    lines.append(f"Comments Analyzed: {snapshot.comments_analyzed:,}")
    lines.append("")

    # LLM executive summary if available
    if snapshot.llm_executive_summary:
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(snapshot.llm_executive_summary)
        lines.append("")

    # Overall sentiment
    lines.append("SENTIMENT OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Positive: {snapshot.primary_positive_pct:.1f}%")
    lines.append(f"Neutral: {snapshot.primary_neutral_pct:.1f}%")
    lines.append(f"Negative: {snapshot.primary_negative_pct:.1f}%")
    lines.append(f"Mean Sentiment Score: {snapshot.primary_sentiment:.3f}")
    lines.append("")

    # Competitor mentions
    if snapshot.competitors:
        lines.append("TOP COMPETITORS MENTIONED")
        lines.append("-" * 40)
        for comp in sorted(snapshot.competitors, key=lambda x: x.mention_count, reverse=True)[:5]:
            lines.append(f"- {comp.competitor}: {comp.mention_count} mentions (avg sentiment: {comp.avg_sentiment:.2f})")

    return "\n".join(lines)


def export_sentiment_breakdown(snapshot: "CompetitorSnapshot") -> pd.DataFrame:
    """
    Export sentiment breakdown for pie/bar chart.

    Args:
        snapshot: CompetitorSnapshot with analysis results

    Returns:
        DataFrame with Sentiment, Percentage, Count columns
    """
    total = snapshot.comments_analyzed or 1  # Avoid division by zero

    data = [
        {
            "Sentiment": "Positive",
            "Percentage": round(snapshot.primary_positive_pct, 1),
            "Count": int(total * snapshot.primary_positive_pct / 100),
        },
        {
            "Sentiment": "Neutral",
            "Percentage": round(snapshot.primary_neutral_pct, 1),
            "Count": int(total * snapshot.primary_neutral_pct / 100),
        },
        {
            "Sentiment": "Negative",
            "Percentage": round(snapshot.primary_negative_pct, 1),
            "Count": int(total * snapshot.primary_negative_pct / 100),
        },
    ]

    return pd.DataFrame(data)


def export_competitor_table(snapshot: "CompetitorSnapshot") -> pd.DataFrame:
    """
    Export competitor comparison table.

    Args:
        snapshot: CompetitorSnapshot with analysis results

    Returns:
        DataFrame with competitor comparison data
    """
    if not snapshot.competitors:
        return pd.DataFrame(columns=[
            "Competitor", "Mentions", "Avg Sentiment", "Positive", "Neutral",
            "Negative", "Switch To", "Switch From", "Better At", "Worse At"
        ])

    rows = []
    for comp in sorted(snapshot.competitors, key=lambda x: x.mention_count, reverse=True):
        rows.append({
            "Competitor": comp.competitor.title(),
            "Mentions": comp.mention_count,
            "Avg Sentiment": round(comp.avg_sentiment, 3),
            "Positive": comp.positive_mentions,
            "Neutral": comp.neutral_mentions,
            "Negative": comp.negative_mentions,
            "Switch To": comp.switch_to_count,
            "Switch From": comp.switch_from_count,
            "Better At": ", ".join(comp.better_at[:3]) if comp.better_at else "",
            "Worse At": ", ".join(comp.worse_at[:3]) if comp.worse_at else "",
        })

    return pd.DataFrame(rows)


def export_themes(snapshot: "CompetitorSnapshot") -> pd.DataFrame:
    """
    Export discussion themes from LLM analysis.

    Args:
        snapshot: CompetitorSnapshot with LLM themes

    Returns:
        DataFrame with theme data
    """
    if not snapshot.llm_themes:
        return pd.DataFrame(columns=["Theme", "Sentiment", "Description", "Keywords"])

    rows = []
    for theme in snapshot.llm_themes:
        rows.append({
            "Theme": theme.get("name", theme.get("theme", "Unknown")),
            "Sentiment": theme.get("sentiment", "mixed"),
            "Description": theme.get("description", theme.get("summary", "")),
            "Keywords": ", ".join(theme.get("keywords", [])[:5]),
        })

    return pd.DataFrame(rows)


def export_threats_opportunities(snapshot: "CompetitorSnapshot") -> pd.DataFrame:
    """
    Export threats and opportunities as two-column table.

    Args:
        snapshot: CompetitorSnapshot with strategic insights

    Returns:
        DataFrame with Threats and Opportunities columns
    """
    max_len = max(len(snapshot.threats), len(snapshot.opportunities), 1)

    threats = snapshot.threats + [""] * (max_len - len(snapshot.threats))
    opportunities = snapshot.opportunities + [""] * (max_len - len(snapshot.opportunities))

    return pd.DataFrame({
        "Threats": threats,
        "Opportunities": opportunities,
    })


def export_recommendations(snapshot: "CompetitorSnapshot") -> pd.DataFrame:
    """
    Export LLM recommendations with priority.

    Args:
        snapshot: CompetitorSnapshot with LLM recommendations

    Returns:
        DataFrame with Priority, Recommendation, Rationale columns
    """
    if not snapshot.llm_recommendations:
        return pd.DataFrame(columns=["Priority", "Recommendation", "Rationale"])

    rows = []
    for rec in snapshot.llm_recommendations:
        rows.append({
            "Priority": rec.get("priority", "medium").title(),
            "Recommendation": rec.get("recommendation", rec.get("action", "")),
            "Rationale": rec.get("rationale", rec.get("reason", "")),
        })

    # Sort by priority: High > Medium > Low
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    rows.sort(key=lambda x: priority_order.get(x["Priority"], 99))

    return pd.DataFrame(rows)


def export_risk_signals(snapshot: "CompetitorSnapshot") -> pd.DataFrame:
    """
    Export risk signals from LLM analysis.

    Args:
        snapshot: CompetitorSnapshot with LLM risk signals

    Returns:
        DataFrame with Severity, Signal, Evidence columns
    """
    if not snapshot.llm_risk_signals:
        return pd.DataFrame(columns=["Severity", "Signal", "Evidence"])

    rows = []
    for risk in snapshot.llm_risk_signals:
        rows.append({
            "Severity": risk.get("severity", "medium").title(),
            "Signal": risk.get("signal", risk.get("risk", "")),
            "Evidence": risk.get("evidence", risk.get("details", "")),
        })

    # Sort by severity: High > Medium > Low
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    rows.sort(key=lambda x: severity_order.get(x["Severity"], 99))

    return pd.DataFrame(rows)


def export_key_quotes(
    snapshot: "CompetitorSnapshot",
    df: Optional[pd.DataFrame] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Export key quotes (top positive and negative comments).

    Args:
        snapshot: CompetitorSnapshot with pain points
        df: Optional DataFrame with all comments (for extracting positive quotes)
        top_n: Number of quotes per category

    Returns:
        DataFrame with Type, Comment, Subreddit, Sentiment columns
    """
    rows = []

    # Add negative quotes from pain points
    for i, pain_point in enumerate(snapshot.top_pain_points[:top_n]):
        rows.append({
            "Type": "Negative",
            "Comment": pain_point[:500],  # Truncate long comments
            "Subreddit": "",  # Not available in pain points
            "Sentiment": "negative",
        })

    # Add positive quotes from DataFrame if available
    if df is not None and "sentiment_label" in df.columns:
        positive_df = df[df["sentiment_label"] == "positive"].nlargest(top_n, "sentiment_score")
        for _, row in positive_df.iterrows():
            body = row.get("body", row.get("text", ""))
            rows.append({
                "Type": "Positive",
                "Comment": str(body)[:500],
                "Subreddit": row.get("subreddit", ""),
                "Sentiment": "positive",
            })

    return pd.DataFrame(rows)


def export_subreddit_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Export subreddit volume breakdown for chart.

    Args:
        df: DataFrame with subreddit column

    Returns:
        DataFrame with Subreddit, Count columns sorted by count
    """
    if df is None or df.empty or "subreddit" not in df.columns:
        return pd.DataFrame(columns=["Subreddit", "Count"])

    counts = df["subreddit"].value_counts().head(15)

    return pd.DataFrame({
        "Subreddit": counts.index.tolist(),
        "Count": counts.values.tolist(),
    })


def create_deck_export_zip(
    snapshot: "CompetitorSnapshot",
    df: Optional[pd.DataFrame] = None,
    brand_name: Optional[str] = None,
) -> io.BytesIO:
    """
    Create ZIP file with all deck export files.

    Args:
        snapshot: CompetitorSnapshot with analysis results
        df: Optional DataFrame with raw comment data
        brand_name: Optional brand name for filename

    Returns:
        BytesIO containing the ZIP file
    """
    brand = brand_name or snapshot.primary_brand
    date_str = snapshot.measured_at.strftime("%Y%m%d")

    # Create in-memory ZIP
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Executive summary (text)
        summary = export_executive_summary(snapshot)
        zf.writestr("01_executive_summary.txt", summary)

        # 2. Sentiment breakdown (CSV)
        sentiment_df = export_sentiment_breakdown(snapshot)
        zf.writestr("02_sentiment_breakdown.csv", sentiment_df.to_csv(index=False))

        # 3. Competitor comparison (CSV)
        competitor_df = export_competitor_table(snapshot)
        zf.writestr("03_competitor_comparison.csv", competitor_df.to_csv(index=False))

        # 4. Themes (CSV)
        themes_df = export_themes(snapshot)
        zf.writestr("04_themes.csv", themes_df.to_csv(index=False))

        # 5. Threats & Opportunities (CSV)
        threats_opps_df = export_threats_opportunities(snapshot)
        zf.writestr("05_threats_opportunities.csv", threats_opps_df.to_csv(index=False))

        # 6. Recommendations (CSV)
        recommendations_df = export_recommendations(snapshot)
        zf.writestr("06_recommendations.csv", recommendations_df.to_csv(index=False))

        # 7. Key quotes (CSV)
        quotes_df = export_key_quotes(snapshot, df)
        zf.writestr("07_key_quotes.csv", quotes_df.to_csv(index=False))

        # 8. Subreddit breakdown (CSV)
        if df is not None:
            subreddit_df = export_subreddit_breakdown(df)
            zf.writestr("08_subreddit_breakdown.csv", subreddit_df.to_csv(index=False))

        # 9. Risk signals (CSV)
        risk_df = export_risk_signals(snapshot)
        zf.writestr("09_risk_signals.csv", risk_df.to_csv(index=False))

    zip_buffer.seek(0)
    return zip_buffer
