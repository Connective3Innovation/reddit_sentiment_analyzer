# src/reddit_sentiment/analytics/theme_extractor.py
"""
Fast theme extraction from comments using TF-IDF and keyword clustering.
No LLM required - runs locally for free.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ExtractedTheme:
    """A theme extracted from comments."""
    name: str
    keywords: list[str]
    comment_count: int
    sentiment_mean: float
    sentiment_label: str  # positive, negative, neutral, mixed
    sample_comments: list[str] = field(default_factory=list)
    subreddits: list[str] = field(default_factory=list)


# Common financial services themes to detect
THEME_PATTERNS = {
    "customer_service": {
        "keywords": ["customer service", "support", "representative", "agent", "phone", "call", "wait", "hold", "chat"],
        "display_name": "Customer Service",
    },
    "mobile_app": {
        "keywords": ["app", "mobile", "interface", "ui", "ux", "glitch", "bug", "update", "login", "crash"],
        "display_name": "Mobile App Experience",
    },
    "rewards": {
        "keywords": ["reward", "points", "cashback", "cash back", "miles", "bonus", "redeem", "earn"],
        "display_name": "Rewards & Points",
    },
    "interest_rates": {
        "keywords": ["interest", "apr", "rate", "yield", "apy", "percentage"],
        "display_name": "Interest Rates & APR",
    },
    "fees": {
        "keywords": ["fee", "charge", "cost", "annual fee", "foreign transaction", "late fee", "penalty"],
        "display_name": "Fees & Charges",
    },
    "credit_limit": {
        "keywords": ["credit limit", "limit increase", "cli", "credit line", "spending limit"],
        "display_name": "Credit Limits",
    },
    "approval": {
        "keywords": ["approv", "denied", "reject", "application", "hard pull", "soft pull", "credit check"],
        "display_name": "Application & Approval",
    },
    "fraud_security": {
        "keywords": ["fraud", "security", "unauthorized", "stolen", "hack", "scam", "protect", "alert"],
        "display_name": "Fraud & Security",
    },
    "website": {
        "keywords": ["website", "site", "online", "portal", "browser", "login"],
        "display_name": "Website Experience",
    },
    "transfer": {
        "keywords": ["transfer", "balance transfer", "bt", "move money", "wire"],
        "display_name": "Balance Transfers",
    },
    "signup_bonus": {
        "keywords": ["signup bonus", "sign up bonus", "sub", "welcome bonus", "new account"],
        "display_name": "Sign-up Bonuses",
    },
    "travel": {
        "keywords": ["travel", "lounge", "airport", "tsa", "priority pass", "global entry"],
        "display_name": "Travel Benefits",
    },
}


class ThemeExtractor:
    """Extract themes from comments without LLM."""

    def __init__(self, custom_themes: Optional[dict] = None):
        """
        Initialize theme extractor.

        Args:
            custom_themes: Optional dict of custom theme patterns to add
        """
        self.themes = THEME_PATTERNS.copy()
        if custom_themes:
            self.themes.update(custom_themes)

    def extract_themes(
        self,
        df: pd.DataFrame,
        text_col: str = "body",
        sentiment_col: str = "sentiment_score",
        min_mentions: int = 3,
        top_n: int = 8,
    ) -> list[ExtractedTheme]:
        """
        Extract themes from a DataFrame of comments.

        Args:
            df: DataFrame with comments
            text_col: Column containing comment text
            sentiment_col: Column containing sentiment scores
            min_mentions: Minimum mentions to include a theme
            top_n: Maximum themes to return

        Returns:
            List of ExtractedTheme objects sorted by comment count
        """
        if df.empty or text_col not in df.columns:
            return []

        # Prepare text for matching
        texts = df[text_col].fillna("").str.lower().tolist()

        theme_results = []

        for theme_id, theme_config in self.themes.items():
            keywords = theme_config["keywords"]
            display_name = theme_config["display_name"]

            # Find comments matching this theme
            matching_indices = []
            matched_keywords = Counter()

            for idx, text in enumerate(texts):
                for kw in keywords:
                    if kw.lower() in text:
                        matching_indices.append(idx)
                        matched_keywords[kw] += 1
                        break  # Only count each comment once per theme

            if len(matching_indices) < min_mentions:
                continue

            # Get matching rows
            matching_df = df.iloc[matching_indices]

            # Calculate sentiment stats
            if sentiment_col in matching_df.columns:
                sentiment_mean = matching_df[sentiment_col].mean()
                pos_count = (matching_df[sentiment_col] > 0.05).sum()
                neg_count = (matching_df[sentiment_col] < -0.05).sum()

                if pos_count > neg_count * 1.5:
                    sentiment_label = "positive"
                elif neg_count > pos_count * 1.5:
                    sentiment_label = "negative"
                elif pos_count > 0 and neg_count > 0:
                    sentiment_label = "mixed"
                else:
                    sentiment_label = "neutral"
            else:
                sentiment_mean = 0.0
                sentiment_label = "neutral"

            # Get sample comments (highest engagement if score available)
            if "score" in matching_df.columns:
                samples = matching_df.nlargest(3, "score")[text_col].tolist()
            else:
                samples = matching_df[text_col].head(3).tolist()

            # Truncate samples
            samples = [s[:200] + "..." if len(s) > 200 else s for s in samples]

            # Get subreddits
            subreddits = []
            if "subreddit" in matching_df.columns:
                subreddits = matching_df["subreddit"].value_counts().head(3).index.tolist()

            # Get top matched keywords
            top_keywords = [kw for kw, _ in matched_keywords.most_common(5)]

            theme_results.append(ExtractedTheme(
                name=display_name,
                keywords=top_keywords,
                comment_count=len(matching_indices),
                sentiment_mean=float(sentiment_mean),
                sentiment_label=sentiment_label,
                sample_comments=samples,
                subreddits=subreddits,
            ))

        # Sort by comment count and return top N
        theme_results.sort(key=lambda t: t.comment_count, reverse=True)
        return theme_results[:top_n]

    def extract_trending_keywords(
        self,
        df: pd.DataFrame,
        text_col: str = "body",
        top_n: int = 20,
        min_word_length: int = 4,
    ) -> list[tuple[str, int]]:
        """
        Extract trending keywords using simple word frequency.

        Args:
            df: DataFrame with comments
            text_col: Column containing text
            top_n: Number of keywords to return
            min_word_length: Minimum word length to consider

        Returns:
            List of (keyword, count) tuples
        """
        if df.empty or text_col not in df.columns:
            return []

        # Common stopwords to exclude
        stopwords = {
            "the", "and", "that", "have", "for", "not", "with", "you", "this",
            "but", "his", "from", "they", "been", "have", "were", "said", "each",
            "which", "their", "will", "other", "about", "many", "then", "them",
            "these", "would", "make", "like", "just", "over", "such", "also",
            "into", "year", "some", "could", "than", "first", "been", "its",
            "after", "only", "think", "know", "get", "got", "can", "dont",
            "really", "even", "going", "want", "way", "because", "when", "what",
            "your", "there", "out", "all", "more", "one", "very", "much",
        }

        # Extract words
        word_counts = Counter()

        for text in df[text_col].fillna(""):
            # Simple tokenization
            words = re.findall(r'\b[a-zA-Z]{%d,}\b' % min_word_length, text.lower())
            for word in words:
                if word not in stopwords:
                    word_counts[word] += 1

        return word_counts.most_common(top_n)


def extract_themes_from_comments(
    df: pd.DataFrame,
    text_col: str = "body",
    sentiment_col: str = "sentiment_score",
) -> list[ExtractedTheme]:
    """
    Convenience function to extract themes from a DataFrame.

    Args:
        df: DataFrame with comments
        text_col: Column containing comment text
        sentiment_col: Column containing sentiment scores

    Returns:
        List of ExtractedTheme objects
    """
    extractor = ThemeExtractor()
    return extractor.extract_themes(df, text_col, sentiment_col)
