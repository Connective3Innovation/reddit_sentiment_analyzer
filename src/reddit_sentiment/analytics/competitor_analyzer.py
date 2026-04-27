# src/reddit_sentiment/analytics/competitor_analyzer.py
"""
Competitive Intelligence Analyzer.

Identifies competitor mentions in Reddit discussions and extracts
what competitors are doing better (or worse).

Supports multi-client configuration via ClientConfig.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, TYPE_CHECKING
from uuid import uuid4

import pandas as pd

from .models import CompetitorSnapshot, CompetitorAnalysis as CompetitorAnalysisModel

if TYPE_CHECKING:
    from ..config.clients import ClientConfig


# Industry-specific topic patterns
# Common patterns applicable to all industries
_COMMON_PATTERNS: dict[str, str] = {
    "customer service": r"(customer service|support|help|representative|agent|chat)",
    "mobile app": r"(app|mobile|interface|ui|ux)",
    "website": r"(website|site|online|portal|login)",
    "pricing": r"(price|pricing|cost|expensive|cheap|afford)",
    "quality": r"(quality|reliable|broken|buggy|works|doesn't work)",
}

_INDUSTRY_PATTERNS: dict[str, dict[str, str]] = {
    "financial_services": {
        "interest rates": r"(interest|apr|rate|yield|apy)",
        "rewards": r"(reward|points|cashback|cash back|miles|bonus)",
        "fees": r"(fee|charge|cost|free|no fee|annual fee)",
        "credit limit": r"(credit limit|limit increase|cli)",
        "approval": r"(approv|denied|reject|accept|hard pull|soft pull)",
        "transfer": r"(transfer|balance transfer|bt)",
        "fraud protection": r"(fraud|security|protect|safe|unauthorized)",
    },
    "tech": {
        "performance": r"(fast|slow|lag|performance|speed|crash)",
        "features": r"(feature|functionality|capability|update)",
        "integration": r"(integrat|api|connect|sync|compatible)",
        "security": r"(security|privacy|encrypt|breach|hack)",
        "reliability": r"(uptime|downtime|outage|reliable|stable)",
    },
    "retail": {
        "shipping": r"(ship|delivery|arrived|package|tracking)",
        "returns": r"(return|refund|exchange|warranty)",
        "product quality": r"(quality|defect|broken|damaged|fake)",
        "selection": r"(selection|variety|stock|available|out of stock)",
        "value": r"(value|worth|deal|overpriced|bargain)",
    },
    "saas": {
        "onboarding": r"(onboard|setup|getting started|learning curve)",
        "features": r"(feature|functionality|capability|roadmap)",
        "integration": r"(integrat|api|connect|sync|zapier|webhook)",
        "pricing model": r"(pricing|tier|plan|subscription|per seat|per user)",
        "support response": r"(response time|ticket|wait|resolved)",
    },
}


def get_topic_patterns(industry: str = "financial_services") -> dict[str, str]:
    """
    Get topic patterns for a specific industry.

    Args:
        industry: Industry identifier (financial_services, tech, retail, saas)

    Returns:
        Dict of topic name -> regex pattern
    """
    patterns = _COMMON_PATTERNS.copy()
    industry_specific = _INDUSTRY_PATTERNS.get(industry.lower(), {})
    patterns.update(industry_specific)
    return patterns


# Default patterns for backwards compatibility
TOPIC_PATTERNS: dict[str, str] = get_topic_patterns("financial_services")


@dataclass
class CompetitorMention:
    """A single competitor mention in a comment."""
    competitor: str
    comment_id: str
    body: str
    sentiment_score: float
    sentiment_label: str
    subreddit: str
    created_utc: datetime
    context: str  # Extracted sentence/phrase mentioning competitor
    comparison_type: Optional[str] = None  # "better", "worse", "same", "switch_to", "switch_from"


@dataclass
class CompetitorAnalysisResult:
    """Analysis summary for a single competitor (internal use)."""
    competitor: str
    mention_count: int
    avg_sentiment: float
    positive_mentions: int
    negative_mentions: int
    neutral_mentions: int
    # Competitive positioning
    better_at: list[str] = field(default_factory=list)  # What they do better
    worse_at: list[str] = field(default_factory=list)   # What they do worse
    switch_to_count: int = 0    # People switching TO this competitor
    switch_from_count: int = 0  # People switching FROM this competitor
    top_subreddits: list[str] = field(default_factory=list)
    sample_mentions: list[CompetitorMention] = field(default_factory=list)


# Default competitors (used when no client config provided)
# Import from clients module if available, otherwise use minimal defaults
def _get_default_competitors() -> dict[str, list[str]]:
    """Get default competitors, falling back to minimal set if clients module unavailable."""
    try:
        from ..config.clients import FINANCIAL_SERVICES_COMPETITORS
        return FINANCIAL_SERVICES_COMPETITORS
    except ImportError:
        return {
            "chase": ["chase", "jp morgan", "jpmorgan"],
            "bank of america": ["bank of america", "bofa", "boa"],
            "wells fargo": ["wells fargo", "wellsfargo"],
            "amex": ["amex", "american express"],
        }


class CompetitorAnalyzer:
    """Analyzes competitor mentions in Reddit data."""

    def __init__(
        self,
        primary_brand: str = "capital one",
        competitors: Optional[dict[str, list[str]]] = None,
        client_config: Optional["ClientConfig"] = None,
    ):
        """
        Initialize analyzer.

        Args:
            primary_brand: The brand being analyzed (ignored if client_config provided)
            competitors: Dict mapping competitor name to list of aliases (ignored if client_config provided)
            client_config: Full client configuration (preferred way to initialize)
        """
        # Store client_config for later use (e.g., search_keywords in pain points filter)
        self.client_config = client_config

        if client_config:
            self.primary_brand = client_config.primary_brand.lower()
            self.competitors = client_config.competitors
            self.client_id = client_config.client_id
            self.industry = client_config.industry
        else:
            self.primary_brand = primary_brand.lower()
            self.competitors = competitors or _get_default_competitors()
            self.client_id = None
            self.industry = "financial_services"  # Default for backwards compatibility

        # Get industry-specific topic patterns
        self._topic_patterns = get_topic_patterns(self.industry)

        # Build regex patterns for each competitor
        self._patterns = {}
        for name, aliases in self.competitors.items():
            pattern = r'\b(' + '|'.join(re.escape(a) for a in aliases) + r')\b'
            self._patterns[name] = re.compile(pattern, re.IGNORECASE)

        # Comparison patterns
        self._better_pattern = re.compile(
            r'(better|superior|prefer|love|great|excellent|amazing|best|recommend)',
            re.IGNORECASE
        )
        self._worse_pattern = re.compile(
            r'(worse|terrible|awful|hate|bad|horrible|worst|avoid|sucks|garbage)',
            re.IGNORECASE
        )
        self._switch_to_pattern = re.compile(
            r'(switch(?:ed|ing)?\s+to|moved?\s+to|changed?\s+to|went\s+(?:with|to))',
            re.IGNORECASE
        )
        self._switch_from_pattern = re.compile(
            r'(switch(?:ed|ing)?\s+from|left|leaving|moved?\s+(?:away\s+)?from|ditched)',
            re.IGNORECASE
        )

    def find_competitor_mentions(
        self,
        df: pd.DataFrame,
        text_col: str = "body",
    ) -> list[CompetitorMention]:
        """
        Find all competitor mentions in a DataFrame of comments.

        Args:
            df: DataFrame with Reddit comments
            text_col: Column containing comment text

        Returns:
            List of CompetitorMention objects
        """
        if df.empty or text_col not in df.columns:
            return []

        mentions = []

        # Use itertuples for ~10x speedup over iterrows
        # Pre-fetch column indices for faster access
        cols = df.columns.tolist()
        text_idx = cols.index(text_col)
        comment_id_idx = cols.index("comment_id") if "comment_id" in cols else None
        sentiment_score_idx = cols.index("sentiment_score") if "sentiment_score" in cols else None
        sentiment_label_idx = cols.index("sentiment_label") if "sentiment_label" in cols else None
        subreddit_idx = cols.index("subreddit") if "subreddit" in cols else None
        created_utc_idx = cols.index("created_utc") if "created_utc" in cols else None

        for row in df.itertuples(index=False):
            text = str(row[text_idx]) if row[text_idx] else ""
            if not text:
                continue

            for competitor, pattern in self._patterns.items():
                match = pattern.search(text)
                if match:
                    # Extract context (sentence containing mention)
                    context = self._extract_context(text, match.start(), match.end())

                    # Determine comparison type
                    comparison_type = self._classify_comparison(context, competitor)

                    mention = CompetitorMention(
                        competitor=competitor,
                        comment_id=str(row[comment_id_idx]) if comment_id_idx is not None else "",
                        body=text[:500],  # Truncate for storage
                        sentiment_score=float(row[sentiment_score_idx]) if sentiment_score_idx is not None and row[sentiment_score_idx] is not None else 0.0,
                        sentiment_label=str(row[sentiment_label_idx]) if sentiment_label_idx is not None else "neutral",
                        subreddit=str(row[subreddit_idx]) if subreddit_idx is not None else "",
                        created_utc=row[created_utc_idx] if created_utc_idx is not None else datetime.now(timezone.utc),
                        context=context,
                        comparison_type=comparison_type,
                    )
                    mentions.append(mention)

        return mentions

    def _extract_context(self, text: str, start: int, end: int, window: int = 150) -> str:
        """Extract context around a match."""
        # Find sentence boundaries
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)

        # Try to find sentence boundaries
        if context_start > 0:
            period = text.rfind('.', context_start - 50, start)
            if period != -1:
                context_start = period + 1

        if context_end < len(text):
            period = text.find('.', end, context_end + 50)
            if period != -1:
                context_end = period + 1

        return text[context_start:context_end].strip()

    def _classify_comparison(self, context: str, competitor: str) -> Optional[str]:
        """Classify the type of comparison being made."""
        context_lower = context.lower()

        # Check for switching behavior
        if self._switch_to_pattern.search(context):
            # Check if switching TO this competitor
            match = self._switch_to_pattern.search(context)
            if match:
                after_switch = context[match.end():match.end() + 50].lower()
                if competitor.lower() in after_switch or any(
                    alias in after_switch
                    for alias in self.competitors.get(competitor, [])
                ):
                    return "switch_to"

        if self._switch_from_pattern.search(context):
            match = self._switch_from_pattern.search(context)
            if match:
                after_switch = context[match.end():match.end() + 50].lower()
                if competitor.lower() in after_switch or any(
                    alias in after_switch
                    for alias in self.competitors.get(competitor, [])
                ):
                    return "switch_from"

        # Check for better/worse sentiment
        if self._better_pattern.search(context):
            return "better"
        if self._worse_pattern.search(context):
            return "worse"

        return None

    def analyze_competitors(
        self,
        df: pd.DataFrame,
        text_col: str = "body",
        top_n: int = 10,
        period_days: int = 30,
        posts_analyzed: int = 0,
    ) -> CompetitorSnapshot:
        """
        Generate full competitive intelligence report as a storable snapshot.

        Args:
            df: DataFrame with Reddit comments (must have sentiment scores)
            text_col: Column containing comment text
            top_n: Number of top competitors to include
            period_days: Number of days this analysis covers
            posts_analyzed: Number of posts fetched

        Returns:
            CompetitorSnapshot for storage in BigQuery
        """
        now = datetime.now(timezone.utc)

        # Find all mentions
        mentions = self.find_competitor_mentions(df, text_col)

        # Group by competitor
        competitor_data: dict[str, list[CompetitorMention]] = {}
        for mention in mentions:
            if mention.competitor not in competitor_data:
                competitor_data[mention.competitor] = []
            competitor_data[mention.competitor].append(mention)

        # Analyze each competitor
        analyses: list[CompetitorAnalysisResult] = []
        for competitor, comp_mentions in competitor_data.items():
            if not comp_mentions:
                continue

            # Filter out NaN sentiment scores for average calculation
            valid_sentiments = [m.sentiment_score for m in comp_mentions
                               if m.sentiment_score is not None and not pd.isna(m.sentiment_score)]
            labels = [m.sentiment_label for m in comp_mentions]
            subreddits = [m.subreddit for m in comp_mentions if m.subreddit]

            # Use BOTH comparison_type AND sentiment_score for better/worse classification
            # This gives more accurate results than just word matching
            better_mentions = [m for m in comp_mentions
                              if m.comparison_type == "better" or
                              (m.comparison_type is None and m.sentiment_label == "positive")]
            worse_mentions = [m for m in comp_mentions
                            if m.comparison_type == "worse" or
                            (m.comparison_type is None and m.sentiment_label == "negative")]
            switch_to = sum(1 for m in comp_mentions if m.comparison_type == "switch_to")
            switch_from = sum(1 for m in comp_mentions if m.comparison_type == "switch_from")

            # Extract what they're better/worse at with sentiment filtering to avoid duplicates
            better_at = self._extract_topics(better_mentions, sentiment_filter="positive")
            worse_at = self._extract_topics(worse_mentions, sentiment_filter="negative")

            # Remove topics that appear in both lists (keep in list with more mentions)
            better_at, worse_at = self._deduplicate_topics(better_mentions, worse_mentions, better_at, worse_at)

            # Get top subreddits
            subreddit_counts = pd.Series(subreddits).value_counts()
            top_subs = subreddit_counts.head(5).index.tolist()

            analysis = CompetitorAnalysisResult(
                competitor=competitor,
                mention_count=len(comp_mentions),
                avg_sentiment=sum(valid_sentiments) / len(valid_sentiments) if valid_sentiments else 0.0,
                positive_mentions=labels.count("positive"),
                negative_mentions=labels.count("negative"),
                neutral_mentions=labels.count("neutral"),
                better_at=better_at,
                worse_at=worse_at,
                switch_to_count=switch_to,
                switch_from_count=switch_from,
                top_subreddits=top_subs,
                sample_mentions=sorted(comp_mentions, key=lambda m: abs(m.sentiment_score), reverse=True)[:5],
            )
            analyses.append(analysis)

        # Sort by mention count
        analyses.sort(key=lambda a: a.mention_count, reverse=True)
        analyses = analyses[:top_n]

        # Calculate primary brand metrics
        primary_sentiment = df["sentiment_score"].mean() if "sentiment_score" in df.columns else 0
        primary_volume = len(df)

        # Calculate sentiment distribution
        if "sentiment_label" in df.columns:
            pos_pct = (df["sentiment_label"] == "positive").sum() / len(df) * 100 if len(df) > 0 else 0
            neu_pct = (df["sentiment_label"] == "neutral").sum() / len(df) * 100 if len(df) > 0 else 0
            neg_pct = (df["sentiment_label"] == "negative").sum() / len(df) * 100 if len(df) > 0 else 0
        else:
            pos_pct = neu_pct = neg_pct = 0

        # Get top pain points (actual complaints about the brand)
        pain_points = []
        if "sentiment_score" in df.columns and "body" in df.columns:
            neg_df = df[df.get("sentiment_label", pd.Series()) == "negative"].copy()
            if len(neg_df) > 0:
                # Filter to only comments that mention the primary brand
                brand_pattern = "|".join(
                    re.escape(kw.lower()) for kw in [self.primary_brand] +
                    (self.client_config.search_keywords if self.client_config else [])
                )
                neg_df = neg_df[neg_df["body"].str.lower().str.contains(brand_pattern, regex=True, na=False)]

                # ALSO filter to comments that contain actual complaint language
                complaint_patterns = (
                    r"(?:hate|terrible|awful|worst|horrible|frustrated|annoying|disappointed|"
                    r"problem|issue|broken|doesn\'t work|won\'t work|can\'t|cannot|refused|"
                    r"denied|rejected|scam|ripoff|rip off|avoid|never again|waste|"
                    r"sucks|garbage|useless|incompetent|ridiculous|unacceptable|pathetic)"
                )
                neg_df = neg_df[neg_df["body"].str.lower().str.contains(complaint_patterns, regex=True, na=False)]

                if len(neg_df) > 0:
                    top_neg = neg_df.nsmallest(5, "sentiment_score")
                    pain_points = [
                        row["body"][:200] + "..." if len(row["body"]) > 200 else row["body"]
                        for _, row in top_neg.iterrows()
                    ]

        # Identify threats and opportunities
        threats = self._identify_threats(analyses)
        opportunities = self._identify_opportunities(analyses)

        # Convert to Pydantic models for storage
        competitor_models = [
            CompetitorAnalysisModel(
                competitor=a.competitor,
                mention_count=a.mention_count,
                avg_sentiment=a.avg_sentiment,
                positive_mentions=a.positive_mentions,
                negative_mentions=a.negative_mentions,
                neutral_mentions=a.neutral_mentions,
                better_at=a.better_at,
                worse_at=a.worse_at,
                switch_to_count=a.switch_to_count,
                switch_from_count=a.switch_from_count,
                top_subreddits=a.top_subreddits,
                sample_mentions=[m.context[:300] for m in a.sample_mentions],
            )
            for a in analyses
        ]

        return CompetitorSnapshot(
            snapshot_id=str(uuid4()),
            client_id=self.client_id,
            primary_brand=self.primary_brand,
            measured_at=now,
            period_start=now - timedelta(days=period_days),
            period_end=now,
            period_days=period_days,
            posts_analyzed=posts_analyzed,
            comments_analyzed=primary_volume,
            primary_sentiment=float(primary_sentiment),
            primary_positive_pct=float(pos_pct),
            primary_neutral_pct=float(neu_pct),
            primary_negative_pct=float(neg_pct),
            competitors=competitor_models,
            total_competitor_mentions=len(mentions),
            threats=threats,
            opportunities=opportunities,
            top_pain_points=pain_points,
        )

    def _extract_topics(
        self,
        mentions: list[CompetitorMention],
        top_n: int = 5,
        sentiment_filter: Optional[str] = None,
    ) -> list[str]:
        """
        Extract common topics from mentions.

        Args:
            mentions: List of competitor mentions
            top_n: Max topics to return
            sentiment_filter: Optional filter - "positive" or "negative"
        """
        topic_counts = {topic: 0 for topic in self._topic_patterns}

        for mention in mentions:
            # Apply sentiment filter if specified
            if sentiment_filter == "positive" and mention.sentiment_label != "positive":
                continue
            if sentiment_filter == "negative" and mention.sentiment_label != "negative":
                continue

            context_lower = mention.context.lower()
            for topic, pattern in self._topic_patterns.items():
                if re.search(pattern, context_lower):
                    topic_counts[topic] += 1

        # Return top topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:top_n] if count > 0]

    def _deduplicate_topics(
        self,
        better_mentions: list[CompetitorMention],
        worse_mentions: list[CompetitorMention],
        better_at: list[str],
        worse_at: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Remove duplicate topics from better_at/worse_at lists.

        If a topic appears in both, keep it only in the list where it has
        more mentions with matching sentiment.
        """
        # Find topics that appear in both lists
        duplicates = set(better_at) & set(worse_at)

        if not duplicates:
            return better_at, worse_at

        # Count sentiment-aligned mentions for each duplicate topic
        for topic in duplicates:
            pattern = self._topic_patterns.get(topic, topic)

            # Count positive sentiment mentions for this topic
            pos_count = sum(
                1 for m in better_mentions
                if m.sentiment_label == "positive" and re.search(pattern, m.context.lower())
            )

            # Count negative sentiment mentions for this topic
            neg_count = sum(
                1 for m in worse_mentions
                if m.sentiment_label == "negative" and re.search(pattern, m.context.lower())
            )

            # Keep in the list with more aligned mentions
            if pos_count > neg_count:
                worse_at = [t for t in worse_at if t != topic]
            elif neg_count > pos_count:
                better_at = [t for t in better_at if t != topic]
            else:
                # Equal - remove from both (ambiguous)
                better_at = [t for t in better_at if t != topic]
                worse_at = [t for t in worse_at if t != topic]

        return better_at, worse_at

    def _identify_threats(self, analyses: list[CompetitorAnalysisResult]) -> list[str]:
        """Identify competitive threats."""
        threats = []

        for analysis in analyses:
            # High switch-to rate is a threat
            if analysis.switch_to_count > 2:
                threats.append(
                    f"Users switching to {analysis.competitor} "
                    f"({analysis.switch_to_count} mentions)"
                )

            # Competitor with better sentiment
            if analysis.avg_sentiment > 0.3 and analysis.mention_count > 5:
                if analysis.better_at:
                    threats.append(
                        f"{analysis.competitor} praised for: {', '.join(analysis.better_at[:3])}"
                    )

        return threats[:5]

    def _identify_opportunities(self, analyses: list[CompetitorAnalysisResult]) -> list[str]:
        """Identify improvement opportunities based on competitor weaknesses."""
        opportunities = []

        for analysis in analyses:
            # Competitor weaknesses are our opportunities
            if analysis.worse_at and analysis.avg_sentiment < 0:
                opportunities.append(
                    f"Opportunity vs {analysis.competitor}: "
                    f"improve {', '.join(analysis.worse_at[:2])}"
                )

            # High switch-from rate means competitor losing customers
            if analysis.switch_from_count > 2:
                opportunities.append(
                    f"Capture users leaving {analysis.competitor} "
                    f"({analysis.switch_from_count} mentions)"
                )

        return opportunities[:5]


def run_competitive_analysis(
    df: pd.DataFrame,
    primary_brand: str = "capital one",
    competitors: Optional[dict[str, list[str]]] = None,
    client_config: Optional["ClientConfig"] = None,
    period_days: int = 30,
    posts_analyzed: int = 0,
) -> CompetitorSnapshot:
    """
    Run competitive analysis on a DataFrame.

    Args:
        df: DataFrame with sentiment-scored comments
        primary_brand: Brand being analyzed (ignored if client_config provided)
        competitors: Optional dict of competitor names to aliases (ignored if client_config provided)
        client_config: Full client configuration (preferred)
        period_days: Number of days this analysis covers
        posts_analyzed: Number of posts fetched

    Returns:
        CompetitorSnapshot ready for BigQuery storage
    """
    analyzer = CompetitorAnalyzer(
        primary_brand=primary_brand,
        competitors=competitors,
        client_config=client_config,
    )
    return analyzer.analyze_competitors(
        df,
        period_days=period_days,
        posts_analyzed=posts_analyzed,
    )
