# src/reddit_sentiment/analytics/competitor_analyzer.py
"""
Competitive Intelligence Analyzer.

Identifies competitor mentions in Reddit discussions and extracts
what competitors are doing better (or worse).

Supports multi-client configuration via ClientConfig.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Final, Optional, TYPE_CHECKING
from uuid import uuid4

import pandas as pd

_LOGGER: Final = logging.getLogger(__name__)

from .models import CompetitorSnapshot, CompetitorAnalysis as CompetitorAnalysisModel

if TYPE_CHECKING:
    from ..config.clients import ClientConfig

# Try to import ABSA sentiment (optional - graceful fallback)
try:
    from ..sentiment import analyze_toward_brand
    ABSA_AVAILABLE = True
except ImportError:
    ABSA_AVAILABLE = False
    analyze_toward_brand = None  # type: ignore


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

        # Flag for ABSA sentiment
        self._use_absa = ABSA_AVAILABLE

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

    def _calculate_brand_sentiment(
        self,
        df: pd.DataFrame,
        brand: str,
        text_col: str = "body",
    ) -> float:
        """
        Calculate ABSA sentiment toward a specific brand.

        Returns average brand sentiment score (-1 to +1).
        Falls back to generic sentiment if ABSA unavailable.
        """
        if not self._use_absa or analyze_toward_brand is None:
            # Fallback: use generic sentiment filtered by brand mentions
            return 0.0

        if df.empty or text_col not in df.columns:
            return 0.0

        texts = df[text_col].fillna("").tolist()
        if not texts:
            return 0.0

        try:
            results = analyze_toward_brand(texts, brand)
            if "brand_sentiment_score" in results.columns:
                return float(results["brand_sentiment_score"].mean())
            return 0.0
        except Exception:
            # Graceful fallback on any error
            return 0.0

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
        _LOGGER.info(
            "COMPETITOR: Starting analysis - brand='%s', comments=%d, competitors=%d, industry=%s",
            self.primary_brand, len(df), len(self.competitors), self.industry
        )
        analysis_start = time.time()

        now = datetime.now(timezone.utc)

        # Find all mentions
        _LOGGER.info("COMPETITOR: Scanning for competitor mentions...")
        mention_start = time.time()
        mentions = self.find_competitor_mentions(df, text_col)
        _LOGGER.info(
            "COMPETITOR: Found %d competitor mentions in %.2fs",
            len(mentions), time.time() - mention_start
        )

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

            # Prioritize sample mentions that compare competitor TO primary brand
            # These provide better context for the analysis (not just general competitor comments)
            primary_pattern = re.compile(
                r'\b(' + '|'.join(re.escape(self.primary_brand)) + r')\b',
                re.IGNORECASE
            )
            # Split mentions: those comparing to primary brand vs general competitor mentions
            comparison_mentions = [m for m in comp_mentions if primary_pattern.search(m.body)]
            other_mentions = [m for m in comp_mentions if not primary_pattern.search(m.body)]

            # Prefer comparison mentions, fall back to general mentions
            sorted_comparison = sorted(comparison_mentions, key=lambda m: abs(m.sentiment_score), reverse=True)
            sorted_other = sorted(other_mentions, key=lambda m: abs(m.sentiment_score), reverse=True)

            # Take comparison mentions first, then fill with other mentions
            best_samples = sorted_comparison[:5]
            if len(best_samples) < 5:
                best_samples.extend(sorted_other[:5 - len(best_samples)])

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
                sample_mentions=best_samples,
            )
            analyses.append(analysis)

        # Sort by mention count
        analyses.sort(key=lambda a: a.mention_count, reverse=True)
        analyses = analyses[:top_n]

        # Calculate primary brand metrics
        # Note: sentiment_score now contains ABSA brand-directed sentiment from the pipeline
        _LOGGER.info("COMPETITOR: Calculating primary brand metrics...")
        primary_volume = len(df)

        # Use existing ABSA scores from pipeline (already brand-directed)
        if "sentiment_score" in df.columns:
            primary_brand_sentiment = float(df["sentiment_score"].mean())
            _LOGGER.info(
                "COMPETITOR: Using pipeline ABSA brand sentiment=%.3f",
                primary_brand_sentiment
            )
        else:
            # Fallback: run ABSA if sentiment_score not in DataFrame
            _LOGGER.info("COMPETITOR: Running ABSA sentiment for primary brand '%s'...", self.primary_brand)
            absa_start = time.time()
            primary_brand_sentiment = self._calculate_brand_sentiment(df, self.primary_brand)
            _LOGGER.info(
                "COMPETITOR: ABSA primary brand sentiment=%.3f (took %.2fs)",
                primary_brand_sentiment, time.time() - absa_start
            )

        # primary_sentiment = primary_brand_sentiment (now both use ABSA)
        primary_sentiment = primary_brand_sentiment

        # Calculate sentiment distribution
        # Ensure percentages sum to 100% by handling NaN and other labels
        if "sentiment_label" in df.columns and len(df) > 0:
            # Normalize labels to lowercase and fill NaN
            labels = df["sentiment_label"].fillna("neutral").str.lower()
            pos_count = (labels == "positive").sum()
            neu_count = (labels == "neutral").sum()
            neg_count = (labels == "negative").sum()
            other_count = len(df) - pos_count - neu_count - neg_count

            # If there are "other" labels (mixed, NaN, etc.), redistribute to neutral
            # This ensures percentages always sum to 100%
            if other_count > 0:
                _LOGGER.debug(
                    "COMPETITOR: %d comments with non-standard labels redistributed to neutral",
                    other_count
                )
                neu_count += other_count

            total = len(df)
            pos_pct = pos_count / total * 100
            neu_pct = neu_count / total * 100
            neg_pct = neg_count / total * 100
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

        # Calculate ABSA brand sentiment for each competitor
        competitor_brand_sentiments = {}
        if self._use_absa and analyses:
            _LOGGER.info("COMPETITOR: Running ABSA sentiment for %d competitors...", len(analyses))
            comp_absa_start = time.time()
            for analysis in analyses:
                # Get comments that mention this competitor
                comp_comments = df[
                    df["body"].str.lower().str.contains(
                        "|".join(re.escape(a) for a in self.competitors.get(analysis.competitor, [analysis.competitor])),
                        regex=True,
                        na=False
                    )
                ]
                if len(comp_comments) > 0:
                    comp_score = self._calculate_brand_sentiment(comp_comments, analysis.competitor)
                    competitor_brand_sentiments[analysis.competitor] = comp_score
                    _LOGGER.debug(
                        "COMPETITOR: ABSA for '%s' = %.3f (%d comments)",
                        analysis.competitor, comp_score, len(comp_comments)
                    )
            _LOGGER.info(
                "COMPETITOR: Competitor ABSA complete in %.2fs",
                time.time() - comp_absa_start
            )

        # Convert to Pydantic models for storage
        competitor_models = [
            CompetitorAnalysisModel(
                competitor=a.competitor,
                mention_count=a.mention_count,
                avg_sentiment=a.avg_sentiment,
                brand_sentiment_score=competitor_brand_sentiments.get(a.competitor, 0.0),
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

        total_elapsed = time.time() - analysis_start
        _LOGGER.info(
            "COMPETITOR: Analysis complete in %.2fs | "
            "brand='%s' sentiment=%.3f (ABSA=%.3f) | "
            "%d competitor mentions | %d threats, %d opportunities | %d pain points",
            total_elapsed,
            self.primary_brand, primary_sentiment, primary_brand_sentiment,
            len(mentions), len(threats), len(opportunities), len(pain_points)
        )

        # Log top competitors
        if competitor_models:
            top_comp = competitor_models[0]
            _LOGGER.info(
                "COMPETITOR: Top competitor: '%s' (%d mentions, ABSA=%.3f)",
                top_comp.competitor, top_comp.mention_count, top_comp.brand_sentiment_score
            )

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
            primary_brand_sentiment=float(primary_brand_sentiment),
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
            # Any switch-to mentions are notable (lowered from >2 to >0)
            if analysis.switch_to_count > 0:
                threats.append(
                    f"Users switching to {analysis.competitor} "
                    f"({analysis.switch_to_count} mention{'s' if analysis.switch_to_count > 1 else ''})"
                )

            # Competitor with positive sentiment (lowered thresholds)
            if analysis.avg_sentiment > 0.1 and analysis.mention_count > 2:
                if analysis.better_at:
                    threats.append(
                        f"{analysis.competitor} praised for: {', '.join(analysis.better_at[:3])}"
                    )
                elif analysis.positive_mentions > analysis.negative_mentions:
                    threats.append(
                        f"{analysis.competitor} has positive sentiment "
                        f"({analysis.positive_mentions} positive vs {analysis.negative_mentions} negative)"
                    )

        return threats[:5]

    def _identify_opportunities(self, analyses: list[CompetitorAnalysisResult]) -> list[str]:
        """Identify improvement opportunities based on competitor weaknesses."""
        opportunities = []

        for analysis in analyses:
            # Competitor weaknesses are opportunities (even with neutral sentiment)
            if analysis.worse_at:
                opportunities.append(
                    f"Opportunity vs {analysis.competitor}: "
                    f"improve {', '.join(analysis.worse_at[:2])}"
                )

            # Any switch-from mentions (lowered from >2 to >0)
            if analysis.switch_from_count > 0:
                opportunities.append(
                    f"Capture users leaving {analysis.competitor} "
                    f"({analysis.switch_from_count} mention{'s' if analysis.switch_from_count > 1 else ''})"
                )

            # Competitor with negative sentiment but no specific worse_at topics
            if analysis.avg_sentiment < -0.1 and analysis.negative_mentions > analysis.positive_mentions:
                if not analysis.worse_at:  # Only add if we didn't already add worse_at
                    opportunities.append(
                        f"{analysis.competitor} has negative sentiment "
                        f"({analysis.negative_mentions} negative mentions)"
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
