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

FOCUS AREAS:
- NEUTRAL comments often contain unanswered questions and content opportunities
- HIGH ENGAGEMENT comments show what topics resonate with users
- Questions that repeat across comments = content gap (opportunity!)
- Comparisons users make frequently = need for comparison guide content

CRITICAL INSTRUCTIONS FOR EVIDENCE QUALITY:
1. ONLY cite quotes from comments that EXPLICITLY MENTION "{primary_brand}" by name
2. DO NOT cite generic financial advice comments as evidence (e.g., "try to build credit first")
3. DO NOT cite AutoMod messages, bot responses, or sidebar redirects
4. Each "evidence_quotes" field MUST contain direct user complaints/praise ABOUT {primary_brand} specifically
5. If a quote doesn't explicitly mention {primary_brand}, DO NOT use it as evidence

FOR PAIN POINTS:
- Extract 3-5 REAL pain points from NEGATIVE comments with direct quotes
- Each quote MUST mention {primary_brand} and describe a specific issue
- For each pain point, verify the quote actually supports the claimed issue
- Look for: app/website issues, customer service complaints, fee frustrations, feature requests, approval/denial issues

FOR CONTENT OPPORTUNITIES:
- Find 5-7 content opportunities from the comments
- Prioritize: unanswered questions, confusion about features, comparison requests
- Each opportunity MUST have evidence quotes that show real user need

FOR RECOMMENDATIONS:
- Provide 5-7 specific, actionable recommendations (not generic advice)
- Each recommendation must cite evidence from the comments
- Include industry-specific recommendations relevant to {industry}

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
      "issue": "Specific customer issue/complaint (be specific, not generic)",
      "severity": "high|medium|low",
      "frequency": "many|some|few (estimate based on how often you see this)",
      "evidence_quotes": [
        "Direct quote 1 from a comment showing this issue",
        "Direct quote 2 from another comment (different user)",
        "Direct quote 3 if available"
      ],
      "business_impact": "Why this matters for {primary_brand}"
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
      "topic": "Specific question/topic users discuss (be specific, not generic)",
      "opportunity_type": "unanswered_question|confusion|high_interest|emerging_trend|comparison_gap",
      "target_audience": "Who is asking/discussing this (e.g., 'new customers', 'people switching from X')",
      "recommended_content": "What type of content would help (blog post, FAQ, video, comparison guide)",
      "evidence": "Quote or paraphrase from comments showing this is a real need"
    }}
  ],

  "actionable_recommendations": [
    {{
      "priority": "high|medium|low",
      "category": "product|service|marketing|support|pricing|content",
      "action": "Specific action to take (be concrete, not generic)",
      "expected_impact": "What improvement this would drive and estimated scope",
      "evidence": "Direct quote from comments supporting this (MUST mention {primary_brand})",
      "target_segment": "Who this helps (e.g., 'small business owners', 'new cardholders')"
    }}
  ],

  "NOTE": "Provide 5-7 actionable_recommendations minimum. Generic advice like 'improve customer service' is not acceptable - be specific about WHAT to improve based on the evidence.",

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
    use_clustering: bool = True,
    primary_brand: str | None = None,
    brand_terms: list[str] | None = None,
) -> dict[str, str]:
    """
    Build stratified samples for LLM analysis.

    We sample from 4 categories:
    - Top positive sentiment comments
    - Top negative sentiment comments
    - Neutral comments (questions, comparisons, factual discussions)
    - Top engagement (score) comments regardless of sentiment

    When use_clustering=True, comments within each group are clustered by topic
    to ensure diverse representation rather than potentially similar comments.

    IMPORTANT: When primary_brand/brand_terms are provided, we ONLY sample from
    comments that explicitly mention the brand. This ensures LLM evidence is
    relevant and specific to the brand being analyzed.

    Sample sizes scale with data:
    - <200 comments: 12/12/10/16 = 50 total
    - 200-500 comments: 15/15/15/25 = 70 total
    - 500-1000 comments: 20/20/20/30 = 90 total
    - 1000+ comments: 25/25/25/35 = 110 total

    This gives the LLM a balanced view including neutral discussions
    which often contain unanswered questions and content opportunities.
    """
    import re

    # CRITICAL: Filter to brand-mentioning comments ONLY for evidence quality
    # Without this filter, generic advice comments pollute the LLM samples
    if primary_brand or brand_terms:
        terms = brand_terms or []
        if primary_brand and primary_brand.lower() not in [t.lower() for t in terms]:
            terms.append(primary_brand)

        if terms and "body" in comments_df.columns:
            # Build pattern for brand mentions
            brand_pattern = "|".join(re.escape(t.lower()) for t in terms)
            original_count = len(comments_df)
            comments_df = comments_df[
                comments_df["body"].str.lower().str.contains(brand_pattern, regex=True, na=False)
            ].copy()
            filtered_count = original_count - len(comments_df)
            if filtered_count > 0:
                print(f"[LLM SAMPLING] Filtered {filtered_count}/{original_count} comments that don't mention brand - keeping {len(comments_df)} brand-specific comments", flush=True)
    # Initialize clusterer if clustering is enabled
    clusterer = None
    if use_clustering:
        try:
            from .comment_clustering import CommentClusterer, is_clustering_available
            if is_clustering_available():
                clusterer = CommentClusterer()
                print("[CLUSTERING] Clustering enabled for diverse sampling", flush=True)
                try:
                    import streamlit as st
                    st.toast("🔀 Clustering enabled for diverse sampling")
                except:
                    pass
            else:
                print("[CLUSTERING] sklearn not available, using standard sampling", flush=True)
        except ImportError:
            print("[CLUSTERING] Import failed, using standard sampling", flush=True)

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
    all_cluster_names = set()  # Track cluster names across all sample types

    # Positive samples (most confident positive)
    if "sentiment_score" in comments_df.columns:
        pos_df = comments_df[comments_df["sentiment_score"] > 0.2].copy()
        if len(pos_df) > 0:
            # Use clustering for diverse topic selection
            if clusterer and len(pos_df) >= 10:
                pos_df = clusterer.get_diverse_samples(pos_df, n_samples=positive_count)
            else:
                pos_df = pos_df.nlargest(positive_count, "sentiment_score")
            samples["positive_samples"] = "\n---\n".join(
                f"[+{row['sentiment_score']:.2f}] [{row.get('cluster_name', 'unclustered')}] {row['body'][:max_chars_per_comment]}"
                for _, row in pos_df.iterrows()
            )
            if 'cluster_name' in pos_df.columns:
                all_cluster_names.update(pos_df['cluster_name'].unique())
        else:
            samples["positive_samples"] = "No positive comments found."
    else:
        samples["positive_samples"] = "Sentiment scores not available."

    # Negative samples (most confident negative)
    if "sentiment_score" in comments_df.columns:
        neg_df = comments_df[comments_df["sentiment_score"] < -0.2].copy()
        if len(neg_df) > 0:
            # Use clustering for diverse topic selection
            if clusterer and len(neg_df) >= 10:
                neg_df = clusterer.get_diverse_samples(neg_df, n_samples=negative_count)
            else:
                neg_df = neg_df.nsmallest(negative_count, "sentiment_score")
            samples["negative_samples"] = "\n---\n".join(
                f"[{row['sentiment_score']:.2f}] [{row.get('cluster_name', 'unclustered')}] {row['body'][:max_chars_per_comment]}"
                for _, row in neg_df.iterrows()
            )
            if 'cluster_name' in neg_df.columns:
                all_cluster_names.update(neg_df['cluster_name'].unique())
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
        # Use clustering for diverse topic selection
        if clusterer and len(neu_df) >= 10:
            neu_df = clusterer.get_diverse_samples(neu_df, n_samples=neutral_count)
        elif "score" in neu_df.columns:
            # Prioritize neutral comments with higher engagement (questions get upvotes)
            neu_df = neu_df.nlargest(neutral_count, "score")
        else:
            neu_df = neu_df.head(neutral_count)
        samples["neutral_samples"] = "\n---\n".join(
            f"[neutral] [{row.get('cluster_name', 'unclustered')}] {row['body'][:max_chars_per_comment]}"
            for _, row in neu_df.iterrows()
        )
        if 'cluster_name' in neu_df.columns:
            all_cluster_names.update(neu_df['cluster_name'].unique())
    else:
        samples["neutral_samples"] = "No neutral comments found."

    # High engagement samples (by Reddit score)
    if "score" in comments_df.columns:
        # Use clustering for diverse topic selection among high-engagement comments
        if clusterer:
            # First get top engagement comments, then cluster within them
            eng_pool = comments_df.nlargest(min(engagement_count * 3, len(comments_df)), "score")
            if len(eng_pool) >= 10:
                eng_df = clusterer.get_diverse_samples(eng_pool, n_samples=engagement_count)
            else:
                eng_df = eng_pool.head(engagement_count)
        else:
            eng_df = comments_df.nlargest(engagement_count, "score")
        samples["engagement_samples"] = "\n---\n".join(
            f"[score={row['score']}] [{row.get('cluster_name', 'unclustered')}] {row['body'][:max_chars_per_comment]}"
            for _, row in eng_df.iterrows()
        )
        if 'cluster_name' in eng_df.columns:
            all_cluster_names.update(eng_df['cluster_name'].unique())
    else:
        # Fall back to random sample
        eng_df = comments_df.head(engagement_count)
        samples["engagement_samples"] = "\n---\n".join(
            f"{row['body'][:max_chars_per_comment]}"
            for _, row in eng_df.iterrows()
        )

    # Remove placeholder values from cluster names
    all_cluster_names.discard('Unclustered')
    all_cluster_names.discard('unclustered')
    if all_cluster_names:
        samples["cluster_topics"] = sorted(all_cluster_names)
        print(f"[CLUSTERING] Identified topics: {sorted(all_cluster_names)}", flush=True)

    return samples


PROMPTS_V2 = {
    "unified_analysis": UNIFIED_ANALYSIS_PROMPT,
}
