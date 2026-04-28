# src/reddit_sentiment/llm/openrouter_client.py
"""
OpenRouter API client for LLM-powered content analysis.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from ..analytics.models import Theme, ContentOpportunity, SentimentSnapshot
from .prompts import PROMPTS

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default models by use case
DEFAULT_MODELS = {
    "theme_extraction": "openai/gpt-4o-mini",
    "opportunity_identification": "openai/gpt-4o",
    "insight_generation": "openai/gpt-4o",
    "shift_explanation": "openai/gpt-4o-mini",
}


class OpenRouterClient:
    """Client for OpenRouter API to access various LLMs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        max_tokens: int = 4000,
        timeout: float = 90.0,
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            default_model: Default model to use (defaults to OPENROUTER_MODEL env var)
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var."
            )

        self.default_model = (
            default_model or os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o"
        )
        self.max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", max_tokens))
        self.timeout = timeout

        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/reddit-sentiment",
                "X-Title": "Reddit Sentiment Analytics",
            },
        )

    def _call_api(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: str = "You are an expert analyst. Respond only with valid JSON.",
        temperature: float = 0.7,
    ) -> str:
        """Make API call to OpenRouter."""
        payload = {
            "model": model or self.default_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": temperature,
        }

        try:
            response = self._client.post(OPENROUTER_API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            # Include response body for better debugging
            error_body = ""
            try:
                error_body = e.response.text[:500]
            except Exception:
                pass
            logger.error(
                f"OpenRouter API error: {e.response.status_code} - {error_body}"
            )
            raise
        except httpx.TimeoutException as e:
            logger.error(f"OpenRouter request timed out after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"OpenRouter request failed: {type(e).__name__}: {e}")
            raise

    def _repair_json(self, text: str) -> str:
        """Attempt to repair common JSON issues from LLM output."""
        import re

        # Fix missing commas between fields (common LLM error)
        # Pattern: "..." followed by whitespace/newlines then "key":
        # Without a comma between them
        text = re.sub(
            r'("\s*)\n(\s*"[^"]+"\s*:)',
            r'\1,\n\2',
            text
        )

        # Fix missing commas after } or ] followed by "key":
        text = re.sub(
            r'(\}|\])\s*\n(\s*"[^"]+"\s*:)',
            r'\1,\n\2',
            text
        )

        # Fix trailing commas before } or ]
        text = re.sub(r',(\s*[\}\]])', r'\1', text)

        return text

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling various markdown formats."""
        text = response.strip()

        # Handle various markdown code block formats
        # ```json\n{...}\n``` or ```\n{...}\n```
        if text.startswith("```"):
            # Find the end of the opening fence
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            else:
                text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Try parsing as-is first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from text (handles LLM adding extra text)
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            extracted = json_match.group()
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                # Try repairing the JSON
                repaired = self._repair_json(extracted)
                try:
                    logger.info("JSON repair attempted")
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

        # Log the raw response for debugging and raise
        logger.error(f"Failed to parse JSON from LLM response: {text[:500]}...")
        print(f"Failed to parse JSON from LLM response: {text[:500]}...", flush=True)
        raise json.JSONDecodeError("Could not extract valid JSON from response", text, 0)

    def extract_themes(
        self,
        comments: list[str],
        keyword: str,
        model: Optional[str] = None,
    ) -> list[Theme]:
        """
        Extract discussion themes from comments.

        Args:
            comments: List of comment texts
            keyword: Search keyword for context
            model: LLM model to use (defaults to theme extraction model)

        Returns:
            List of Theme objects
        """
        # Limit to top 50 comments to stay within context limits
        sample = comments[:50]
        comments_text = "\n---\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(sample))

        prompt = PROMPTS["theme_extraction"].format(
            keyword=keyword,
            comments=comments_text,
        )

        model = model or DEFAULT_MODELS["theme_extraction"]
        response = self._call_api(prompt, model=model)

        try:
            data = self._parse_json_response(response)
            themes = []
            for t in data.get("themes", []):
                # Map sentiment string to float
                sentiment_str = t.get("sentiment", "mixed").lower()
                sentiment_map = {"positive": 0.5, "negative": -0.5, "mixed": 0.0}
                sentiment_mean = sentiment_map.get(sentiment_str, 0.0)

                themes.append(
                    Theme(
                        name=t.get("name", "Unknown"),
                        description=t.get("description", ""),
                        keywords=t.get("keywords", []),
                        sentiment=sentiment_str,
                        comment_count=len(sample),  # Approximate
                        sentiment_mean=sentiment_mean,
                        sample_comments=t.get("sample_excerpts", []),
                        subreddits=[],
                    )
                )
            return themes
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse theme extraction response: {e}")
            return []

    def identify_opportunities(
        self,
        comments: list[str],
        keyword: str,
        positive_pct: float = 33.0,
        neutral_pct: float = 34.0,
        negative_pct: float = 33.0,
        model: Optional[str] = None,
    ) -> list[ContentOpportunity]:
        """
        Identify content opportunities from comments.

        Args:
            comments: List of comment texts (ideally sorted by engagement)
            keyword: Search keyword
            positive_pct: Percentage of positive sentiment
            neutral_pct: Percentage of neutral sentiment
            negative_pct: Percentage of negative sentiment
            model: LLM model to use

        Returns:
            List of ContentOpportunity objects
        """
        # Limit to top 75 comments
        sample = comments[:75]
        comments_text = "\n---\n".join(f"[{i+1}] {c[:400]}" for i, c in enumerate(sample))

        prompt = PROMPTS["opportunity_identification"].format(
            keyword=keyword,
            comments=comments_text,
            positive_pct=f"{positive_pct:.1f}",
            neutral_pct=f"{neutral_pct:.1f}",
            negative_pct=f"{negative_pct:.1f}",
        )

        model = model or DEFAULT_MODELS["opportunity_identification"]
        response = self._call_api(prompt, model=model)

        try:
            data = self._parse_json_response(response)
            opportunities = []
            for opp in data.get("opportunities", []):
                # Map difficulty to competition score
                difficulty_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
                difficulty = opp.get("difficulty", "medium").lower()
                competition = difficulty_map.get(difficulty, 0.5)

                opportunities.append(
                    ContentOpportunity(
                        keyword=keyword,
                        topic=opp.get("topic", "Unknown"),
                        detected_at=datetime.now(timezone.utc),
                        engagement_score=0.7,  # Default high since LLM selected it
                        competition_score=competition,
                        opportunity_score=(0.7 + (1 - competition)) / 2,
                        evidence={
                            "reason": opp.get("evidence", ""),
                            "target_audience": opp.get("target_audience", ""),
                            "sample_comments": opp.get("sample_comments", []),
                        },
                        subreddits=[],
                        recommended_action=opp.get("recommended_approach", ""),
                    )
                )
            return opportunities
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse opportunity response: {e}")
            return []

    def generate_insights(
        self,
        snapshot: SentimentSnapshot,
        top_positive: list[str],
        top_negative: list[str],
        model: Optional[str] = None,
    ) -> str:
        """
        Generate executive insights summary.

        Args:
            snapshot: SentimentSnapshot with metrics
            top_positive: List of top positive comments
            top_negative: List of top negative comments
            model: LLM model to use

        Returns:
            Insights summary text
        """
        # Format top subreddits
        top_subs_text = "\n".join(
            f"- r/{sub}: {count} comments"
            for sub, count in snapshot.top_subreddits[:10]
        )

        # Format comment samples
        pos_text = "\n".join(f"- {c[:300]}" for c in top_positive[:5])
        neg_text = "\n".join(f"- {c[:300]}" for c in top_negative[:5])

        prompt = PROMPTS["insight_generation"].format(
            keyword=snapshot.keyword,
            period_start=snapshot.period_start.strftime("%Y-%m-%d"),
            period_end=snapshot.period_end.strftime("%Y-%m-%d"),
            mean_sentiment=snapshot.distribution.mean,
            positive_pct=snapshot.distribution.positive_pct,
            neutral_pct=snapshot.distribution.neutral_pct,
            negative_pct=snapshot.distribution.negative_pct,
            volume=snapshot.volume,
            unique_authors=snapshot.unique_authors,
            top_subreddits=top_subs_text or "No subreddit data available",
            top_positive=pos_text or "No positive comments available",
            top_negative=neg_text or "No negative comments available",
        )

        model = model or DEFAULT_MODELS["insight_generation"]
        response = self._call_api(
            prompt,
            model=model,
            system_prompt="You are a professional analyst writing executive summaries. Be concise and actionable.",
            temperature=0.5,
        )

        return response

    def explain_shift(
        self,
        keyword: str,
        period_a: tuple[datetime, datetime],
        period_b: tuple[datetime, datetime],
        mean_a: float,
        mean_b: float,
        positive_a: float,
        positive_b: float,
        volume_a: int,
        volume_b: int,
        sample_comments: list[str],
        model: Optional[str] = None,
    ) -> str:
        """
        Generate explanation for a sentiment shift.

        Args:
            keyword: Search keyword
            period_a: First period (start, end)
            period_b: Second period (start, end)
            mean_a: Mean sentiment for period A
            mean_b: Mean sentiment for period B
            positive_a: Positive % for period A
            positive_b: Positive % for period B
            volume_a: Volume for period A
            volume_b: Volume for period B
            sample_comments: Sample comments from period B
            model: LLM model to use

        Returns:
            Shift explanation text
        """
        shift_pct = abs(mean_b - mean_a) / max(abs(mean_a), 0.01) * 100
        shift_direction = "improved" if mean_b > mean_a else "declined"

        comments_text = "\n".join(f"- {c[:300]}" for c in sample_comments[:10])

        prompt = PROMPTS["shift_explanation"].format(
            keyword=keyword,
            period_a_start=period_a[0].strftime("%Y-%m-%d"),
            period_a_end=period_a[1].strftime("%Y-%m-%d"),
            period_b_start=period_b[0].strftime("%Y-%m-%d"),
            period_b_end=period_b[1].strftime("%Y-%m-%d"),
            mean_a=mean_a,
            mean_b=mean_b,
            positive_a=positive_a,
            positive_b=positive_b,
            volume_a=volume_a,
            volume_b=volume_b,
            shift_direction=shift_direction,
            shift_pct=f"{shift_pct:.1f}",
            sample_comments=comments_text or "No sample comments available",
        )

        model = model or DEFAULT_MODELS["shift_explanation"]
        response = self._call_api(
            prompt,
            model=model,
            system_prompt="You are an analyst explaining sentiment changes. Be specific and evidence-based.",
            temperature=0.5,
        )

        return response

    def analyze_competitive_intelligence(
        self,
        primary_brand: str,
        industry: str,
        period_days: int,
        comments_analyzed: int,
        mean_sentiment: float,
        positive_pct: float,
        neutral_pct: float,
        negative_pct: float,
        competitor_summary: str,
        pain_points: list[str],
        sample_comments: list[str],
        model: Optional[str] = None,
    ) -> dict:
        """
        Generate comprehensive competitive intelligence analysis using LLM.

        Args:
            primary_brand: Brand being analyzed
            industry: Industry sector
            period_days: Days of data analyzed
            comments_analyzed: Total comments
            mean_sentiment: Overall sentiment score
            positive_pct: % positive comments
            neutral_pct: % neutral comments
            negative_pct: % negative comments
            competitor_summary: Formatted competitor mentions (deprecated, kept for compatibility)
            pain_points: Top negative comments
            sample_comments: High-engagement comment samples - PRIMARY DATA SOURCE
            model: LLM model to use

        Returns:
            Dict with brand_perception, competitors_mentioned, key_themes,
            unanswered_questions, actionable_recommendations, risk_signals
        """
        # Combine sample comments and pain points for richer analysis
        # Send more comments to LLM since it's now identifying competitors
        all_comments = sample_comments[:30] + pain_points[:10]
        samples_text = "\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(all_comments[:40]))

        prompt = PROMPTS["competitive_intelligence"].format(
            primary_brand=primary_brand,
            industry=industry,
            period_days=period_days,
            comments_analyzed=comments_analyzed,
            mean_sentiment=mean_sentiment,
            positive_pct=positive_pct,
            neutral_pct=neutral_pct,
            negative_pct=negative_pct,
            sample_comments=samples_text or "No sample comments available",
        )

        model = model or DEFAULT_MODELS.get("opportunity_identification", "openai/gpt-4o")

        try:
            response = self._call_api(
                prompt,
                model=model,
                system_prompt="You are a senior competitive intelligence analyst. Respond only with valid JSON.",
                temperature=0.6,
            )
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"Competitive intelligence analysis failed: {e}")
            return {
                "executive_summary": f"Analysis failed: {str(e)}",
                "themes": [],
                "unanswered_questions": [],
                "competitive_insights": [],
                "actionable_recommendations": [],
                "risk_signals": [],
            }

    def run_unified_analysis(
        self,
        primary_brand: str,
        industry: str,
        period_days: int,
        comments_df,  # pandas DataFrame with sentiment_score, body, score columns
        model: Optional[str] = None,
    ) -> dict:
        """
        Run unified analysis with stratified sampling (RECOMMENDED - single API call).

        This method replaces the need for separate calls to:
        - extract_themes()
        - analyze_competitive_intelligence()
        - identify_opportunities()

        Cost: ~$0.03-0.10 (vs ~$0.15-0.30 for 3 separate calls)

        Args:
            primary_brand: Brand being analyzed
            industry: Industry sector
            period_days: Days of data analyzed
            comments_df: DataFrame with columns: body, sentiment_score, score (optional)
            model: LLM model to use (defaults to gpt-4o)

        Returns:
            Comprehensive analysis dict with all insights in one response
        """
        from .prompts_v2 import UNIFIED_ANALYSIS_PROMPT, build_stratified_samples

        # Build brand terms for filtering (include common variations)
        brand_terms = [primary_brand.lower()]
        if "capital one" in primary_brand.lower():
            brand_terms.extend(["capitalone", "cap one", "capital one's"])

        # Build stratified samples for balanced coverage
        # IMPORTANT: Pass brand terms to filter out non-brand-specific comments
        samples = build_stratified_samples(
            comments_df,
            primary_brand=primary_brand,
            brand_terms=brand_terms,
        )

        # Calculate metrics from DataFrame
        comments_analyzed = len(comments_df)
        mean_sentiment = float(comments_df["sentiment_score"].mean()) if "sentiment_score" in comments_df.columns else 0.0

        if "sentiment_label" in comments_df.columns and comments_analyzed > 0:
            # Normalize labels and ensure they sum to 100%
            labels = comments_df["sentiment_label"].fillna("neutral").str.lower()
            pos_count = (labels == "positive").sum()
            neu_count = (labels == "neutral").sum()
            neg_count = (labels == "negative").sum()
            other_count = comments_analyzed - pos_count - neu_count - neg_count

            # Redistribute "other" labels (mixed, NaN) to neutral
            neu_count += other_count

            positive_pct = pos_count / comments_analyzed * 100
            neutral_pct = neu_count / comments_analyzed * 100
            negative_pct = neg_count / comments_analyzed * 100
        else:
            positive_pct = neutral_pct = negative_pct = 33.3

        prompt = UNIFIED_ANALYSIS_PROMPT.format(
            primary_brand=primary_brand,
            industry=industry,
            period_days=period_days,
            comments_analyzed=comments_analyzed,
            mean_sentiment=mean_sentiment,
            positive_pct=positive_pct,
            neutral_pct=neutral_pct,
            negative_pct=negative_pct,
            positive_samples=samples["positive_samples"],
            negative_samples=samples["negative_samples"],
            neutral_samples=samples.get("neutral_samples", "No neutral comments available."),
            engagement_samples=samples["engagement_samples"],
        )

        model = model or "openai/gpt-4o"

        try:
            response = self._call_api(
                prompt,
                model=model,
                system_prompt="You are a senior competitive intelligence analyst. Respond only with valid JSON. Be specific and evidence-based.",
                temperature=0.5,
            )
            result = self._parse_json_response(response)

            # Normalize field names for compatibility with existing code
            if "verified_pain_points" in result and "top_pain_points" not in result:
                result["top_pain_points"] = [
                    p.get("issue", "") for p in result["verified_pain_points"]
                ]
            if "content_opportunities" in result and "opportunities" not in result:
                result["opportunities"] = result["content_opportunities"]

            # Include cluster topics from sampling (for UI display)
            if "cluster_topics" in samples:
                result["cluster_topics"] = samples["cluster_topics"]

            return result
        except Exception as e:
            logger.error(f"Unified analysis failed: {e}")
            return {
                "executive_summary": f"Analysis failed: {str(e)}",
                "brand_perception": {},
                "verified_pain_points": [],
                "competitors_mentioned": [],
                "key_themes": [],
                "content_opportunities": [],
                "actionable_recommendations": [],
                "risk_signals": [],
                "competitive_position": {},
            }

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
