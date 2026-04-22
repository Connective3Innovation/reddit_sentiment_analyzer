# src/reddit_sentiment/llm/prompts.py
"""
LLM prompt templates for content analysis.
"""

THEME_EXTRACTION_PROMPT = """Analyze Reddit comments about "{keyword}" and extract what users are saying SPECIFICALLY ABOUT THIS BRAND.

BRAND TO ANALYZE: {keyword}

COMMENTS:
{comments}

---

IMPORTANT: Extract themes that are SPECIFICALLY about "{keyword}" (the brand/company), NOT general industry topics.

BAD example themes (too generic):
- "Credit Card Advice" (generic topic)
- "Financial Planning" (not about the brand)
- "Cashback Strategies" (general topic)

GOOD example themes (brand-specific):
- "{keyword} App Issues" (specific to this brand's app)
- "{keyword} Customer Service" (about this brand's service)
- "{keyword} Approval Process" (about this brand's process)
- "Switching from {keyword}" (about this brand specifically)

Identify 3-7 themes about what users think/say about {keyword} specifically. For each theme:
1. A descriptive name that includes or relates to {keyword} (2-5 words)
2. What users are specifically saying about {keyword} on this topic
3. Key phrases users use when discussing this
4. Overall sentiment (positive, negative, or mixed)
5. 1-2 actual quotes from the comments

Respond in JSON format:
{{
  "themes": [
    {{
      "name": "Theme Name (brand-specific)",
      "keywords": ["keyword1", "keyword2"],
      "sentiment": "positive|negative|mixed",
      "description": "What users say about {keyword} on this topic",
      "sample_excerpts": ["actual quote from comments"]
    }}
  ]
}}"""

OPPORTUNITY_IDENTIFICATION_PROMPT = """You are a content strategist analyzing Reddit discussions about "{keyword}".

COMMENTS (sorted by engagement):
{comments}

SENTIMENT DISTRIBUTION:
- Positive: {positive_pct}%
- Neutral: {neutral_pct}%
- Negative: {negative_pct}%

---

Identify untapped content opportunities where:
1. There are many questions being asked but few good answers
2. People express frustration or confusion about specific topics
3. There's high engagement on topics with little quality content addressing them
4. Emerging trends or new topics with growing interest

For each opportunity found, provide:
1. The specific topic/question (be specific, not generic)
2. Why it's an opportunity (evidence from comments)
3. The target audience
4. A recommended content approach
5. Estimated difficulty (low/medium/high)

Respond in JSON format:
{{
  "opportunities": [
    {{
      "topic": "Specific topic or question",
      "evidence": "Why this is an opportunity",
      "target_audience": "Who would benefit",
      "recommended_approach": "What type of content to create",
      "difficulty": "low|medium|high",
      "sample_comments": ["comment supporting this opportunity"]
    }}
  ]
}}"""

INSIGHT_GENERATION_PROMPT = """You are an analyst providing insights on Reddit sentiment data.

KEYWORD: {keyword}
TIME PERIOD: {period_start} to {period_end}

SENTIMENT METRICS:
- Mean Sentiment: {mean_sentiment:.2f}
- Positive: {positive_pct}%
- Neutral: {neutral_pct}%
- Negative: {negative_pct}%
- Total Volume: {volume} comments
- Unique Authors: {unique_authors}

TOP SUBREDDITS:
{top_subreddits}

TOP POSITIVE COMMENTS:
{top_positive}

TOP NEGATIVE COMMENTS:
{top_negative}

---

Provide a concise executive summary (3-4 paragraphs) covering:
1. Overall sentiment health and what's driving it
2. Key topics and themes in discussions
3. Notable trends or shifts
4. Actionable recommendations

Write in a professional but accessible tone. Be specific and cite evidence from the data."""

SHIFT_EXPLANATION_PROMPT = """Explain why sentiment might have shifted for "{keyword}".

BEFORE PERIOD ({period_a_start} to {period_a_end}):
- Mean Sentiment: {mean_a:.2f}
- Positive: {positive_a}%
- Volume: {volume_a} comments

AFTER PERIOD ({period_b_start} to {period_b_end}):
- Mean Sentiment: {mean_b:.2f}
- Positive: {positive_b}%
- Volume: {volume_b} comments

SHIFT: {shift_direction} by {shift_pct}%

SAMPLE COMMENTS FROM AFTER PERIOD:
{sample_comments}

---

Provide a brief analysis (2-3 paragraphs) explaining:
1. What likely caused the sentiment shift
2. Key events or topics that may have influenced the change
3. Whether this appears to be a temporary or sustained shift

Be specific and reference the comment samples where relevant."""

COMPETITIVE_INTELLIGENCE_PROMPT = """You are a competitive intelligence analyst reviewing Reddit discussions about "{primary_brand}" in the {industry} industry.

BRAND: {primary_brand}
INDUSTRY: {industry}
PERIOD: Last {period_days} days
COMMENTS ANALYZED: {comments_analyzed}

SENTIMENT METRICS:
- Mean Score: {mean_sentiment:.3f} (scale: -1 to +1)
- Positive: {positive_pct:.1f}%
- Neutral: {neutral_pct:.1f}%
- Negative: {negative_pct:.1f}%

SAMPLE REDDIT COMMENTS (analyze these carefully):
{sample_comments}

---

Analyze the comments and provide competitive intelligence in JSON format.

IMPORTANT INSTRUCTIONS:
1. ONLY analyze comments that are ACTUALLY about {primary_brand} (the company/brand)
2. Identify ANY competitors mentioned by users (don't rely on a predefined list)
3. Extract REAL pros/cons that users are discussing
4. Be specific - cite actual user concerns, not generic statements

{{
  "executive_summary": "2-3 sentence summary of what Reddit users think about {primary_brand}. Be specific about the main praise and complaints.",

  "brand_perception": {{
    "overall_sentiment": "positive|negative|mixed",
    "what_users_love": ["specific thing 1", "specific thing 2"],
    "what_users_hate": ["specific complaint 1", "specific complaint 2"],
    "common_use_cases": ["how users typically use the product/service"]
  }},

  "competitors_mentioned": [
    {{
      "competitor": "Name of competitor mentioned in comments",
      "context": "How/why users mentioned this competitor",
      "compared_to_brand": "How users compare them to {primary_brand}",
      "competitor_pros": ["advantages users cite"],
      "competitor_cons": ["disadvantages users cite"]
    }}
  ],

  "key_themes": [
    {{
      "theme": "Theme name (2-4 words)",
      "description": "What users are saying about this",
      "sentiment": "positive|negative|mixed",
      "sample_quote": "Brief quote or paraphrase from comments"
    }}
  ],

  "unanswered_questions": [
    {{
      "question": "Question users are asking that isn't being answered",
      "frequency": "low|medium|high",
      "opportunity": "How {primary_brand} could address this"
    }}
  ],

  "actionable_recommendations": [
    {{
      "priority": "high|medium|low",
      "action": "Specific action {primary_brand} should take",
      "evidence": "What in the comments supports this recommendation"
    }}
  ],

  "risk_signals": [
    {{
      "signal": "Concerning pattern observed",
      "severity": "high|medium|low",
      "evidence": "What comments showed this"
    }}
  ]
}}"""

PROMPTS = {
    "theme_extraction": THEME_EXTRACTION_PROMPT,
    "opportunity_identification": OPPORTUNITY_IDENTIFICATION_PROMPT,
    "insight_generation": INSIGHT_GENERATION_PROMPT,
    "shift_explanation": SHIFT_EXPLANATION_PROMPT,
    "competitive_intelligence": COMPETITIVE_INTELLIGENCE_PROMPT,
}
