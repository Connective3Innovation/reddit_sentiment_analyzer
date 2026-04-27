# src/reddit_sentiment/llm/prompts_v2.py
"""
Improved LLM prompt templates - consolidated for cost efficiency.

Key improvements over v1:
1. Single comprehensive prompt instead of 3 separate calls (3x cost reduction)
2. Stratified sampling instructions for better coverage
3. Industry-agnostic (works for any brand/industry)
4. Pain point verification built-in
"""

UNIFIED_ANALYSIS_PROMPT = """You are a competitive intelligence analyst reviewing Reddit discussions about "{primary_brand}" in the {industry} industry.

BRAND: {primary_brand}
INDUSTRY: {industry}
PERIOD: Last {period_days} days
TOTAL COMMENTS: {comments_analyzed}

SENTIMENT METRICS (from ML classifier):
- Mean Score: {mean_sentiment:.3f} (scale: -1 to +1)
- Positive: {positive_pct:.1f}%
- Neutral: {neutral_pct:.1f}%
- Negative: {negative_pct:.1f}%

---
SAMPLE COMMENTS - POSITIVE SENTIMENT:
{positive_samples}

---
SAMPLE COMMENTS - NEGATIVE SENTIMENT:
{negative_samples}

---
SAMPLE COMMENTS - NEUTRAL (questions, comparisons, factual):
{neutral_samples}

---
SAMPLE COMMENTS - HIGH ENGAGEMENT:
{engagement_samples}

---

Analyze ALL the comments above and provide a comprehensive analysis in JSON format.
Pay special attention to NEUTRAL comments - they often contain unanswered questions and content opportunities.

CRITICAL INSTRUCTIONS:
1. ONLY analyze comments that are ACTUALLY about {primary_brand}
2. Identify competitors mentioned by users (don't rely on predefined lists)
3. Extract REAL pain points - specific complaints with evidence
4. Be specific - cite actual user concerns, not generic statements
5. Identify content opportunities where users have questions without good answers

{{
  "executive_summary": "3-4 sentence summary of Reddit sentiment about {primary_brand}. Include main strengths, weaknesses, and overall perception.",

  "brand_perception": {{
    "overall_sentiment": "positive|negative|mixed",
    "sentiment_drivers": {{
      "positive": ["specific thing users praise - with evidence"],
      "negative": ["specific complaint - with evidence"]
    }},
    "brand_strengths": ["what {primary_brand} does well according to users"],
    "brand_weaknesses": ["where {primary_brand} falls short"]
  }},

  "verified_pain_points": [
    {{
      "issue": "Specific customer issue/complaint",
      "severity": "high|medium|low",
      "frequency": "How often this appears in comments",
      "sample_quote": "Direct quote showing this issue",
      "business_impact": "Why this matters for the brand"
    }}
  ],

  "competitors_mentioned": [
    {{
      "competitor": "Competitor name found in comments",
      "mention_context": "switching_to|switching_from|comparison|recommendation",
      "user_perception": "How users compare them to {primary_brand}",
      "why_mentioned": "Specific reason users brought them up"
    }}
  ],

  "key_themes": [
    {{
      "theme": "Theme name (2-5 words, brand-specific)",
      "description": "What users are saying",
      "sentiment": "positive|negative|mixed",
      "comment_evidence": ["quote 1", "quote 2"]
    }}
  ],

  "content_opportunities": [
    {{
      "topic": "Specific question/topic users discuss",
      "opportunity_type": "unanswered_question|confusion|high_interest|emerging_trend",
      "target_audience": "Who is asking/discussing this",
      "recommended_content": "What type of content would help",
      "evidence": "Why this is an opportunity"
    }}
  ],

  "actionable_recommendations": [
    {{
      "priority": "high|medium|low",
      "category": "product|service|marketing|support|pricing",
      "action": "Specific action to take",
      "expected_impact": "What improvement this would drive",
      "evidence": "What comments support this"
    }}
  ],

  "risk_signals": [
    {{
      "signal": "Concerning pattern or trend",
      "severity": "high|medium|low",
      "trend": "growing|stable|declining",
      "evidence": "Specific comments showing this"
    }}
  ],

  "competitive_position": {{
    "market_perception": "How users see {primary_brand} vs alternatives",
    "switching_risk": "low|medium|high",
    "loyalty_indicators": ["What keeps users with {primary_brand}"],
    "churn_indicators": ["What makes users consider leaving"]
  }}
}}"""


def build_stratified_samples(
    comments_df,
    positive_count: int | None = None,
    negative_count: int | None = None,
    neutral_count: int | None = None,
    engagement_count: int | None = None,
    max_chars_per_comment: int = 500,
) -> dict[str, str]:
    """
    Build stratified samples for LLM analysis.

    We sample from 4 categories:
    - Top positive sentiment comments
    - Top negative sentiment comments
    - Neutral comments (questions, comparisons, factual discussions)
    - Top engagement (score) comments regardless of sentiment

    Sample sizes scale with data:
    - <200 comments: 12/12/10/16 = 50 total
    - 200-500 comments: 15/15/15/25 = 70 total
    - 500-1000 comments: 20/20/20/30 = 90 total
    - 1000+ comments: 25/25/25/35 = 110 total

    This gives the LLM a balanced view including neutral discussions
    which often contain unanswered questions and content opportunities.
    """
    # Auto-scale sample sizes based on data volume
    total_comments = len(comments_df)
    if positive_count is None or negative_count is None or neutral_count is None or engagement_count is None:
        if total_comments >= 1000:
            positive_count = positive_count or 25
            negative_count = negative_count or 25
            neutral_count = neutral_count or 25
            engagement_count = engagement_count or 35
        elif total_comments >= 500:
            positive_count = positive_count or 20
            negative_count = negative_count or 20
            neutral_count = neutral_count or 20
            engagement_count = engagement_count or 30
        elif total_comments >= 200:
            positive_count = positive_count or 15
            negative_count = negative_count or 15
            neutral_count = neutral_count or 15
            engagement_count = engagement_count or 25
        else:
            positive_count = positive_count or 12
            negative_count = negative_count or 12
            neutral_count = neutral_count or 10
            engagement_count = engagement_count or 16
    samples = {}

    # Positive samples (most confident positive)
    if "sentiment_score" in comments_df.columns:
        pos_df = comments_df[comments_df["sentiment_score"] > 0.2].copy()
        if len(pos_df) > 0:
            pos_df = pos_df.nlargest(positive_count, "sentiment_score")
            samples["positive_samples"] = "\n---\n".join(
                f"[+{row['sentiment_score']:.2f}] {row['body'][:max_chars_per_comment]}"
                for _, row in pos_df.iterrows()
            )
        else:
            samples["positive_samples"] = "No positive comments found."
    else:
        samples["positive_samples"] = "Sentiment scores not available."

    # Negative samples (most confident negative)
    if "sentiment_score" in comments_df.columns:
        neg_df = comments_df[comments_df["sentiment_score"] < -0.2].copy()
        if len(neg_df) > 0:
            neg_df = neg_df.nsmallest(negative_count, "sentiment_score")
            samples["negative_samples"] = "\n---\n".join(
                f"[{row['sentiment_score']:.2f}] {row['body'][:max_chars_per_comment]}"
                for _, row in neg_df.iterrows()
            )
        else:
            samples["negative_samples"] = "No negative comments found."
    else:
        samples["negative_samples"] = "Sentiment scores not available."

    # Neutral samples (questions, comparisons, factual discussions)
    # Use sentiment_label for 3-class model, or score threshold for binary
    if "sentiment_label" in comments_df.columns:
        neu_df = comments_df[comments_df["sentiment_label"] == "neutral"].copy()
    elif "sentiment_score" in comments_df.columns:
        neu_df = comments_df[
            (comments_df["sentiment_score"] >= -0.2) &
            (comments_df["sentiment_score"] <= 0.2)
        ].copy()
    else:
        neu_df = comments_df.copy()

    if len(neu_df) > 0:
        # Prioritize neutral comments with higher engagement (questions get upvotes)
        if "score" in neu_df.columns:
            neu_df = neu_df.nlargest(neutral_count, "score")
        else:
            neu_df = neu_df.head(neutral_count)
        samples["neutral_samples"] = "\n---\n".join(
            f"[neutral, score={row.get('score', 0)}] {row['body'][:max_chars_per_comment]}"
            for _, row in neu_df.iterrows()
        )
    else:
        samples["neutral_samples"] = "No neutral comments found."

    # High engagement samples (by Reddit score)
    if "score" in comments_df.columns:
        eng_df = comments_df.nlargest(engagement_count, "score")
        samples["engagement_samples"] = "\n---\n".join(
            f"[score={row['score']}] {row['body'][:max_chars_per_comment]}"
            for _, row in eng_df.iterrows()
        )
    else:
        # Fall back to random sample
        eng_df = comments_df.head(engagement_count)
        samples["engagement_samples"] = "\n---\n".join(
            f"{row['body'][:max_chars_per_comment]}"
            for _, row in eng_df.iterrows()
        )

    return samples


PROMPTS_V2 = {
    "unified_analysis": UNIFIED_ANALYSIS_PROMPT,
}
