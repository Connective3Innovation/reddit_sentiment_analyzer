# src/reddit_sentiment/analytics/opportunity_scorer.py
"""
Content opportunity scoring for identifying high-value topics.

Combines engagement metrics, competition analysis, and LLM-powered theme extraction
to surface untapped content opportunities.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .models import ContentOpportunity, Theme

logger = logging.getLogger(__name__)


class OpportunityScorer:
    """Identifies and scores content opportunities from Reddit data."""

    # Scoring weights
    WEIGHTS = {
        "engagement": 0.40,
        "competition": 0.35,
        "trend": 0.25,
    }

    # Thresholds
    HIGH_ENGAGEMENT_THRESHOLD = 50  # comments
    QUALITY_POST_SCORE = 100  # minimum score for "quality"
    MIN_OPPORTUNITY_SCORE = 0.5

    def __init__(
        self,
        engagement_weight: float = 0.40,
        competition_weight: float = 0.35,
        trend_weight: float = 0.25,
    ):
        """
        Initialize opportunity scorer.

        Args:
            engagement_weight: Weight for engagement score (0-1)
            competition_weight: Weight for competition score (0-1)
            trend_weight: Weight for trend score (0-1)
        """
        total = engagement_weight + competition_weight + trend_weight
        self.weights = {
            "engagement": engagement_weight / total,
            "competition": competition_weight / total,
            "trend": trend_weight / total,
        }

    def score_dataframe(
        self,
        posts_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        keyword: str,
        llm_opportunities: Optional[list[ContentOpportunity]] = None,
    ) -> list[ContentOpportunity]:
        """
        Score content opportunities from posts and comments data.

        Args:
            posts_df: DataFrame with post data
            comments_df: DataFrame with comment data
            keyword: Search keyword
            llm_opportunities: Optional LLM-identified opportunities to enrich

        Returns:
            List of scored ContentOpportunity objects
        """
        # Calculate overall metrics
        engagement = self._calculate_engagement_metrics(comments_df)
        competition = self._calculate_competition_metrics(posts_df)
        trend = self._calculate_trend_metrics(comments_df)

        opportunities = []

        # If LLM opportunities provided, enrich them with scores
        if llm_opportunities:
            for opp in llm_opportunities:
                # Calculate combined score
                opp_score = (
                    self.weights["engagement"] * opp.engagement_score
                    + self.weights["competition"] * (1 - opp.competition_score)
                    + self.weights["trend"] * 0.7  # Default trend for LLM items
                )
                opp.opportunity_score = opp_score
                if opp_score >= self.MIN_OPPORTUNITY_SCORE:
                    opportunities.append(opp)

        # Also identify opportunities from data patterns
        data_opportunities = self._identify_from_data(
            posts_df, comments_df, keyword, engagement, competition, trend
        )
        opportunities.extend(data_opportunities)

        # Sort by score descending
        opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)

        return opportunities

    def _calculate_engagement_metrics(self, comments_df: pd.DataFrame) -> dict:
        """Calculate engagement metrics from comments."""
        if comments_df.empty:
            return {
                "total_comments": 0,
                "avg_score": 0,
                "unique_authors": 0,
                "discussion_ratio": 0,
                "score": 0,
            }

        total = len(comments_df)
        avg_score = comments_df["score"].mean() if "score" in comments_df.columns else 0
        unique_authors = (
            comments_df["author"].nunique() if "author" in comments_df.columns else 0
        )

        # Discussion ratio: approximate from post_id grouping
        if "post_id" in comments_df.columns:
            comments_per_post = comments_df.groupby("post_id").size()
            avg_comments_per_post = comments_per_post.mean()
        else:
            avg_comments_per_post = total

        # Normalize engagement score (0-1)
        # Higher score if many comments, high avg score, many authors
        score = min(
            1.0,
            (
                (total / 500)  # Volume component
                + (avg_score / 50)  # Quality component
                + (unique_authors / total if total > 0 else 0)  # Diversity
            )
            / 3,
        )

        return {
            "total_comments": total,
            "avg_score": float(avg_score),
            "unique_authors": unique_authors,
            "avg_comments_per_post": float(avg_comments_per_post),
            "score": score,
        }

    def _calculate_competition_metrics(self, posts_df: pd.DataFrame) -> dict:
        """Calculate competition metrics from posts."""
        if posts_df.empty:
            return {
                "post_count": 0,
                "avg_post_score": 0,
                "high_quality_posts": 0,
                "author_concentration": 0,
                "score": 0,
            }

        post_count = len(posts_df)
        avg_score = posts_df["score"].mean() if "score" in posts_df.columns else 0

        # Count high-quality posts
        high_quality = (
            len(posts_df[posts_df["score"] >= self.QUALITY_POST_SCORE])
            if "score" in posts_df.columns
            else 0
        )

        # Author concentration (Herfindahl-Hirschman Index)
        if "author" in posts_df.columns:
            author_shares = posts_df["author"].value_counts(normalize=True)
            hhi = float((author_shares**2).sum())
        else:
            hhi = 1.0

        # Competition score (0-1, higher = more competition)
        # Low competition = fewer quality posts, lower HHI (more diverse)
        score = min(
            1.0,
            (
                (high_quality / max(post_count, 1))  # Quality post ratio
                + hhi  # Concentration
            )
            / 2,
        )

        return {
            "post_count": post_count,
            "avg_post_score": float(avg_score),
            "high_quality_posts": high_quality,
            "author_concentration": hhi,
            "score": score,
        }

    def _calculate_trend_metrics(self, comments_df: pd.DataFrame) -> dict:
        """Calculate trend/momentum metrics."""
        if comments_df.empty or "created" not in comments_df.columns:
            return {"velocity": 0, "acceleration": 0, "recency": 0, "score": 0.5}

        try:
            comments_df = comments_df.copy()
            comments_df["created"] = pd.to_datetime(comments_df["created"], utc=True)

            # Sort by time
            sorted_df = comments_df.sort_values("created")

            # Calculate daily volumes
            daily = sorted_df.set_index("created").resample("D").size()

            if len(daily) < 2:
                return {"velocity": float(len(comments_df)), "acceleration": 0, "recency": 1, "score": 0.7}

            # Velocity: recent daily average
            recent_days = min(7, len(daily))
            velocity = float(daily.tail(recent_days).mean())

            # Acceleration: change in velocity
            if len(daily) >= 14:
                recent_vel = daily.tail(7).mean()
                prev_vel = daily.iloc[-14:-7].mean()
                acceleration = float(recent_vel - prev_vel)
            else:
                acceleration = 0

            # Recency: how recent is the most recent comment
            most_recent = sorted_df["created"].max()
            now = datetime.now(timezone.utc)
            days_since = (now - most_recent).days
            recency = max(0, 1 - (days_since / 30))  # Decay over 30 days

            # Trend score
            # Higher if: high velocity, positive acceleration, recent activity
            score = min(
                1.0,
                (
                    min(1, velocity / 50)  # Velocity component
                    + (0.5 + min(0.5, max(-0.5, acceleration / 20)))  # Acceleration
                    + recency  # Recency
                )
                / 3,
            )

            return {
                "velocity": velocity,
                "acceleration": acceleration,
                "recency": recency,
                "score": score,
            }

        except Exception as e:
            logger.warning(f"Failed to calculate trend metrics: {e}")
            return {"velocity": 0, "acceleration": 0, "recency": 0, "score": 0.5}

    def _identify_from_data(
        self,
        posts_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        keyword: str,
        engagement: dict,
        competition: dict,
        trend: dict,
    ) -> list[ContentOpportunity]:
        """Identify opportunities from data patterns."""
        opportunities = []

        # Pattern 1: High engagement but few quality posts
        if (
            engagement["total_comments"] >= self.HIGH_ENGAGEMENT_THRESHOLD
            and competition["high_quality_posts"] < 3
        ):
            opp_score = (
                self.weights["engagement"] * engagement["score"]
                + self.weights["competition"] * (1 - competition["score"])
                + self.weights["trend"] * trend["score"]
            )

            if opp_score >= self.MIN_OPPORTUNITY_SCORE:
                opportunities.append(
                    ContentOpportunity(
                        keyword=keyword,
                        topic=f"General discussion about {keyword}",
                        detected_at=datetime.now(timezone.utc),
                        engagement_score=engagement["score"],
                        competition_score=competition["score"],
                        opportunity_score=opp_score,
                        evidence={
                            "total_comments": engagement["total_comments"],
                            "high_quality_posts": competition["high_quality_posts"],
                            "trend_velocity": trend["velocity"],
                        },
                        subreddits=self._get_top_subreddits(comments_df),
                        recommended_action=(
                            f"Create comprehensive content about {keyword}. "
                            f"High discussion volume ({engagement['total_comments']} comments) "
                            f"but only {competition['high_quality_posts']} quality posts."
                        ),
                    )
                )

        # Pattern 2: Rising trend with low competition
        if trend["acceleration"] > 5 and competition["score"] < 0.4:
            opp_score = (
                self.weights["engagement"] * 0.6  # Moderate engagement assumed
                + self.weights["competition"] * (1 - competition["score"])
                + self.weights["trend"] * min(1, trend["score"] * 1.2)  # Boost trend
            )

            if opp_score >= self.MIN_OPPORTUNITY_SCORE:
                opportunities.append(
                    ContentOpportunity(
                        keyword=keyword,
                        topic=f"Trending topic: {keyword}",
                        detected_at=datetime.now(timezone.utc),
                        engagement_score=0.6,
                        competition_score=competition["score"],
                        opportunity_score=opp_score,
                        evidence={
                            "trend_acceleration": trend["acceleration"],
                            "velocity": trend["velocity"],
                            "competition_score": competition["score"],
                        },
                        subreddits=self._get_top_subreddits(comments_df),
                        recommended_action=(
                            f"Capitalize on rising interest in {keyword}. "
                            f"Discussion velocity increasing by {trend['acceleration']:.1f} "
                            f"comments/day with low competition."
                        ),
                    )
                )

        return opportunities

    def _get_top_subreddits(
        self, comments_df: pd.DataFrame, limit: int = 5
    ) -> list[str]:
        """Get top subreddits from comments."""
        if "subreddit" not in comments_df.columns:
            return []

        return comments_df["subreddit"].value_counts().head(limit).index.tolist()

    def extract_themes_simple(
        self,
        comments_df: pd.DataFrame,
        min_occurrences: int = 3,
    ) -> list[Theme]:
        """
        Simple theme extraction using n-gram analysis.

        This is a fallback when LLM is not available.

        Args:
            comments_df: DataFrame with comment data
            min_occurrences: Minimum occurrences to be considered a theme

        Returns:
            List of Theme objects
        """
        if "body" not in comments_df.columns:
            return []

        # Simple word frequency approach
        from collections import Counter
        import re

        # Common stop words to ignore
        stop_words = {
            "the", "a", "an", "is", "it", "to", "and", "of", "in", "for",
            "on", "that", "this", "with", "as", "be", "at", "by", "from",
            "or", "are", "was", "were", "been", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should",
            "i", "you", "he", "she", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his", "its", "our", "their",
            "what", "which", "who", "whom", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more", "most", "other",
            "some", "such", "no", "not", "only", "same", "so", "than",
            "too", "very", "just", "also", "now", "here", "there",
        }

        # Extract words and bigrams
        word_counter: Counter = Counter()
        bigram_counter: Counter = Counter()

        for text in comments_df["body"].dropna():
            # Clean and tokenize
            words = re.findall(r'\b[a-z]{3,15}\b', text.lower())
            words = [w for w in words if w not in stop_words]

            word_counter.update(words)

            # Bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigram_counter.update([bigram])

        # Combine top words and bigrams as themes
        themes = []

        # Top bigrams as themes
        for phrase, count in bigram_counter.most_common(5):
            if count >= min_occurrences:
                themes.append(
                    Theme(
                        name=phrase.title(),
                        keywords=phrase.split(),
                        comment_count=count,
                        sentiment_mean=0.5,  # Would need sentiment data
                        sample_comments=[],
                        subreddits=[],
                    )
                )

        # Top single words as additional themes
        for word, count in word_counter.most_common(10):
            if count >= min_occurrences * 2:  # Higher threshold for single words
                # Skip if already covered by bigram
                if any(word in t.name.lower() for t in themes):
                    continue
                themes.append(
                    Theme(
                        name=word.title(),
                        keywords=[word],
                        comment_count=count,
                        sentiment_mean=0.5,
                        sample_comments=[],
                        subreddits=[],
                    )
                )

        return themes[:7]  # Limit to top 7 themes
