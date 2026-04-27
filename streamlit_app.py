"""
Streamlit UI for Reddit Competitive Intelligence Analysis.

Run with: streamlit run streamlit_app.py
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path for module imports
src_path = Path(__file__).parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

# Support Streamlit Cloud secrets - merge into environment
try:
    import streamlit as _st_secrets
    if hasattr(_st_secrets, 'secrets'):
        for key in ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT',
                    'REDDIT_USERNAME', 'REDDIT_PASSWORD',
                    'OPENROUTER_API_KEY', 'REDDIT_GCP_PROJECT', 'GCP_SERVICE_ACCOUNT']:
            if key in _st_secrets.secrets and not os.getenv(key):
                # Strip whitespace and quotes that might be in TOML values
                value = str(_st_secrets.secrets[key]).strip().strip('"').strip("'")
                os.environ[key] = value
except Exception:
    pass  # Running locally without secrets

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import re
from collections import Counter

# Optional: ECharts for word cloud (graceful fallback if not installed)
try:
    from streamlit_echarts import st_echarts
    ECHARTS_AVAILABLE = True
except ImportError:
    ECHARTS_AVAILABLE = False
    st_echarts = None

# Page config
st.set_page_config(
    page_title="Reddit Competitive Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI - Light theme for better readability
st.markdown("""
<style>
    /* Light/clean background */
    .stApp {
        background: #f8fafc;
    }

    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.25);
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        padding: 0.75rem 1.25rem;
        margin: 1.5rem 0 1rem 0;
        border-radius: 10px;
    }

    .section-header h2 {
        color: #ffffff;
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }

    /* Make sidebar text darker */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] span {
        color: #1e293b !important;
    }

    /* Expander styling - better contrast */
    .streamlit-expanderHeader {
        background: #ffffff;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        color: #1e293b !important;
    }

    .streamlit-expanderHeader:hover {
        background: #f1f5f9;
        border-color: #6366f1;
    }

    .streamlit-expanderHeader p {
        color: #1e293b !important;
        font-weight: 500;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 0.5rem;
        gap: 0.5rem;
        border: 1px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #64748b;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"] {
        color: #1e293b !important;
    }

    [data-testid="stMetricDelta"] {
        color: #10b981 !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 10px;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    /* Chart containers */
    [data-testid="stVegaLiteChart"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }

    /* Health score badge */
    .health-badge {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem;
        border-radius: 50%;
        width: 130px;
        height: 130px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
    }

    .health-good {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.35);
    }

    .health-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff;
        box-shadow: 0 8px 30px rgba(245, 158, 11, 0.35);
    }

    .health-critical {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff;
        box-shadow: 0 8px 30px rgba(239, 68, 68, 0.35);
    }

    /* General text colors */
    .stMarkdown, .stText, p, span, label {
        color: #334155;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #1e293b;
    }

    /* Alert boxes */
    .stAlert {
        border-radius: 10px;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: #e2e8f0;
        margin: 2rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Feature cards on welcome screen */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
        border-color: #6366f1;
        transform: translateY(-4px);
    }

    .feature-card h3 {
        color: #1e293b;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }

    .feature-card p {
        color: #64748b;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None
if "running" not in st.session_state:
    st.session_state.running = False
if "posts_df" not in st.session_state:
    st.session_state.posts_df = None
if "comments_df" not in st.session_state:
    st.session_state.comments_df = None
if "opportunities" not in st.session_state:
    st.session_state.opportunities = None
if "kpi_metrics" not in st.session_state:
    st.session_state.kpi_metrics = None
if "llm_opportunities" not in st.session_state:
    st.session_state.llm_opportunities = None
if "llm_debug" not in st.session_state:
    st.session_state.llm_debug = None
if "llm_error" not in st.session_state:
    st.session_state.llm_error = None


def run_analysis(client_config, days_back, post_limit, run_llm, detailed_search=False, max_keywords=5):
    """Run the analysis pipeline."""
    from reddit_sentiment.data.collector import collect
    from reddit_sentiment.data.preprocess import apply_cleaning
    from reddit_sentiment.sentiment.hf_engine import HfEngine
    from reddit_sentiment.analytics.competitor_analyzer import CompetitorAnalyzer
    from reddit_sentiment.analytics.opportunity_scorer import OpportunityScorer

    # Clear Reddit client cache to pick up any new credentials
    try:
        from reddit_sentiment.auth.reddit_auth import clear_client_cache
        clear_client_cache()
        st.info("Reddit client cache cleared - using fresh credentials")
    except ImportError:
        pass  # Old version without cache clear function

    progress = st.progress(0, text="Starting analysis...")

    # Step 1: Collect data
    # Reddit API limits results to ~100-250 per search, so we run multiple searches
    if detailed_search and client_config.search_keywords:
        # Search each keyword separately and combine (gets more data)
        all_posts = []
        all_comments = []
        keywords_to_search = client_config.search_keywords[:max_keywords]

        for i, kw in enumerate(keywords_to_search):
            # Scale progress dynamically: 5% to 25% across all keywords
            search_progress = int(5 + (i + 1) / len(keywords_to_search) * 20)
            progress.progress(
                search_progress,
                text=f"Searching for '{kw}' ({i+1}/{len(keywords_to_search)})..."
            )
            try:
                posts, comments = collect(f'"{kw}"', limit=post_limit // len(keywords_to_search), days_back=days_back)
                all_posts.append(posts)
                all_comments.append(comments)
            except Exception as e:
                st.warning(f"Search for '{kw}' failed: {e}")

        # Combine and deduplicate
        if all_posts:
            posts_df = pd.concat(all_posts, ignore_index=True).drop_duplicates(subset=["id"])
            comments_df = pd.concat(all_comments, ignore_index=True).drop_duplicates(subset=["comment_id"])
        else:
            posts_df, comments_df = pd.DataFrame(), pd.DataFrame()
    else:
        # Simple search: just brand name
        search_query = f'"{client_config.primary_brand}"'
        progress.progress(10, text=f"Collecting Reddit data for {search_query}...")
        posts_df, comments_df = collect(search_query, limit=post_limit, days_back=days_back)

    if len(comments_df) == 0:
        st.error("No comments found!")
        return None

    st.info(f"Found {len(posts_df)} posts and {len(comments_df)} comments")

    # Step 2: Clean text
    progress.progress(30, text="Cleaning text...")
    comments_df = apply_cleaning(comments_df)

    # Step 3: Sentiment analysis (using 3-class model: positive/neutral/negative)
    progress.progress(50, text="Running HuggingFace sentiment analysis...")
    engine = HfEngine()
    sentiment_df = engine.run(comments_df["body"].tolist())

    def convert_to_score(row):
        """Convert 3-class sentiment to numeric score.

        POSITIVE → +confidence (0 to +1)
        NEUTRAL  → 0 (no strong sentiment)
        NEGATIVE → -confidence (0 to -1)
        """
        sentiment = row["sentiment"]
        confidence = float(row["prob"]) if row["prob"] is not None else 0.5
        confidence = max(0.0, min(1.0, confidence))

        if sentiment == "POSITIVE":
            return confidence
        elif sentiment == "NEGATIVE":
            return -confidence
        else:  # NEUTRAL
            return 0.0

    comments_df["sentiment_score"] = sentiment_df.apply(convert_to_score, axis=1)
    # Use model's 3-class labels directly, with fallback to score-based
    comments_df["sentiment_label"] = sentiment_df["sentiment"].str.lower()

    # Store dataframes for other analyses
    st.session_state.posts_df = posts_df
    st.session_state.comments_df = comments_df

    # Step 4: Competitor analysis
    progress.progress(70, text="Analyzing competitor mentions...")
    analyzer = CompetitorAnalyzer(client_config=client_config)
    snapshot = analyzer.analyze_competitors(
        comments_df,
        period_days=days_back,
        posts_analyzed=len(posts_df),
    )

    # Step 5: Content Opportunity Scoring
    progress.progress(80, text="Scoring content opportunities...")
    try:
        scorer = OpportunityScorer()
        opportunities = scorer.score_dataframe(
            posts_df, comments_df, client_config.primary_brand
        )
        st.session_state.opportunities = opportunities
    except Exception as e:
        st.warning(f"Opportunity scoring failed: {e}")
        st.session_state.opportunities = []

    # Step 6: Calculate KPI Metrics
    st.session_state.kpi_metrics = calculate_kpis(posts_df, comments_df, snapshot)

    # Step 7: LLM analysis (optional)
    if run_llm:
        progress.progress(90, text="Running LLM deep analysis...")
        snapshot = run_llm_analysis(snapshot, comments_df, client_config)

    progress.progress(100, text="Analysis complete!")
    return snapshot


def run_llm_analysis(snapshot, comments_df, client_config):
    """Run LLM deep analysis."""
    import traceback

    # Clear previous debug/error state
    st.session_state.llm_debug = {}
    st.session_state.llm_error = None

    try:
        from reddit_sentiment.llm.openrouter_client import OpenRouterClient
    except ImportError as e:
        st.session_state.llm_error = f"LLM module not available: {e}"
        st.warning(st.session_state.llm_error)
        return snapshot

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.session_state.llm_error = "OPENROUTER_API_KEY not set"
        st.warning(st.session_state.llm_error)
        return snapshot

    # Import caching utilities
    try:
        from reddit_sentiment.llm.cache import get_cached_result, save_to_cache
        cache_available = True
    except ImportError:
        cache_available = False

    st.session_state.llm_debug["comments_count"] = len(comments_df)
    st.session_state.llm_debug["cache_available"] = cache_available

    result = {}

    # Check cache first (avoids repeat API calls)
    if cache_available:
        cached = get_cached_result(
            primary_brand=client_config.primary_brand,
            industry=client_config.industry,
            period_days=snapshot.period_days,
            comment_count=len(comments_df),
            max_age_hours=24,
        )
        if cached:
            result = cached
            st.session_state.llm_debug["cache_hit"] = True
            st.info("📦 Using cached LLM analysis (less than 24h old)")

    # If no cache hit, run unified analysis (single API call)
    if not result:
        st.session_state.llm_debug["cache_hit"] = False
        try:
            with OpenRouterClient(api_key=api_key) as client:
                st.text("Running unified LLM analysis (stratified sampling)...")

                result = client.run_unified_analysis(
                    primary_brand=client_config.primary_brand,
                    industry=client_config.industry,
                    period_days=snapshot.period_days,
                    comments_df=comments_df,
                )

                st.session_state.llm_debug["unified_success"] = True
                st.session_state.llm_debug["result_keys"] = list(result.keys()) if result else []

                # Save to cache for future use
                if cache_available and result and "executive_summary" in result:
                    save_to_cache(
                        result=result,
                        primary_brand=client_config.primary_brand,
                        industry=client_config.industry,
                        period_days=snapshot.period_days,
                        comment_count=len(comments_df),
                    )
                    st.session_state.llm_debug["cache_saved"] = True

        except Exception as e:
            st.session_state.llm_error = f"LLM analysis error: {e}\n{traceback.format_exc()}"
            st.session_state.llm_debug["unified_error"] = str(e)
            st.error(st.session_state.llm_error)
            return snapshot

    # Map unified result to snapshot fields
    if result:
        snapshot.llm_executive_summary = result.get("executive_summary")
        snapshot.llm_brand_perception = result.get("brand_perception", {})
        snapshot.llm_unanswered_questions = result.get("unanswered_questions", [])
        snapshot.llm_competitive_insights = result.get("competitors_mentioned", result.get("competitive_insights", []))
        snapshot.llm_recommendations = result.get("actionable_recommendations", [])
        snapshot.llm_risk_signals = result.get("risk_signals", [])

        # Themes from unified analysis
        snapshot.llm_themes = result.get("key_themes", result.get("themes", []))

        # Content opportunities from unified analysis
        opportunities = result.get("content_opportunities", result.get("opportunities", []))
        if opportunities:
            st.session_state.llm_opportunities = opportunities

        # LLM-verified pain points (more accurate than rule-based)
        verified_pain_points = result.get("verified_pain_points", [])
        if verified_pain_points:
            st.session_state.llm_verified_pain_points = verified_pain_points
            st.session_state.llm_debug["verified_pain_points"] = len(verified_pain_points)

    # Summary
    cache_status = "📦 cached" if st.session_state.llm_debug.get("cache_hit") else "🔄 fresh"
    themes_count = len(snapshot.llm_themes or [])
    opps_count = len(result.get("content_opportunities", []))
    recs_count = len(snapshot.llm_recommendations or [])

    st.success(f"✅ LLM analysis complete ({cache_status}) | "
              f"{themes_count} themes, {opps_count} opportunities, {recs_count} recommendations")

    if snapshot.llm_executive_summary:
        st.success("✅ Executive summary generated")
    else:
        st.warning("⚠️ No executive summary returned")

    return snapshot


def calculate_kpis(posts_df, comments_df, snapshot):
    """Calculate KPI metrics from the analysis."""
    kpis = {}

    # Volume metrics
    kpis["total_comments"] = len(comments_df)
    kpis["total_posts"] = len(posts_df)
    kpis["comments_per_post"] = len(comments_df) / max(len(posts_df), 1)

    # Engagement metrics
    if "score" in comments_df.columns:
        kpis["avg_comment_score"] = float(comments_df["score"].mean())
        kpis["max_comment_score"] = int(comments_df["score"].max())
    else:
        kpis["avg_comment_score"] = 0
        kpis["max_comment_score"] = 0

    if "score" in posts_df.columns:
        kpis["avg_post_score"] = float(posts_df["score"].mean())
    else:
        kpis["avg_post_score"] = 0

    # Author diversity
    if "author" in comments_df.columns:
        kpis["unique_authors"] = comments_df["author"].nunique()
        kpis["author_diversity"] = kpis["unique_authors"] / max(len(comments_df), 1)
    else:
        kpis["unique_authors"] = 0
        kpis["author_diversity"] = 0

    # Subreddit diversity
    if "subreddit" in comments_df.columns:
        kpis["unique_subreddits"] = comments_df["subreddit"].nunique()
        kpis["top_subreddits"] = comments_df["subreddit"].value_counts().head(5).to_dict()
    else:
        kpis["unique_subreddits"] = 0
        kpis["top_subreddits"] = {}

    # Sentiment metrics
    kpis["sentiment_mean"] = snapshot.primary_sentiment
    kpis["sentiment_positive_pct"] = snapshot.primary_positive_pct
    kpis["sentiment_neutral_pct"] = snapshot.primary_neutral_pct
    kpis["sentiment_negative_pct"] = snapshot.primary_negative_pct

    # Sentiment volatility (std dev)
    if "sentiment_score" in comments_df.columns:
        kpis["sentiment_volatility"] = float(comments_df["sentiment_score"].std())
    else:
        kpis["sentiment_volatility"] = 0

    # Health score (composite)
    # Clamp each component to 0-1 range before combining
    positive_component = min(max(kpis["sentiment_positive_pct"] / 100, 0), 1)
    diversity_component = min(max(kpis["author_diversity"], 0), 1)
    engagement_component = min(kpis["comments_per_post"] / 10, 1)
    # Volatility: std dev can exceed 1, so clamp (1 - volatility) to 0-1
    stability_component = min(max(1 - kpis["sentiment_volatility"], 0), 1)

    health_score = (
        positive_component * 0.4  # 40% weight on positive sentiment
        + diversity_component * 0.2  # 20% weight on author diversity
        + engagement_component * 0.2  # 20% weight on engagement
        + stability_component * 0.2  # 20% weight on stability
    )
    kpis["health_score"] = min(max(health_score, 0), 1)  # Final clamp 0-1

    return kpis


def display_results(snapshot):
    """Display analysis results."""
    st.markdown("""
    <div class="section-header">
        <h2>📊 Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💬 Comments Analyzed", f"{snapshot.comments_analyzed:,}")
    with col2:
        st.metric("📝 Posts Analyzed", f"{snapshot.posts_analyzed:,}")
    with col3:
        sentiment_delta = "positive" if snapshot.primary_sentiment > 0 else "negative"
        emoji = "😊" if snapshot.primary_sentiment > 0 else "😟"
        st.metric(
            f"{emoji} Mean Sentiment",
            f"{snapshot.primary_sentiment:.3f}",
            delta=sentiment_delta,
        )
    with col4:
        st.metric("🏢 Competitor Mentions", f"{snapshot.total_competitor_mentions:,}")

    # Key Takeaways & Recommendations Summary (if LLM ran)
    if snapshot.llm_executive_summary:
        st.divider()

        # Executive Summary Card
        st.markdown("""
        <div class="section-header">
            <h2 style="font-size: 1.1rem;">📋 Key Takeaways</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: #ffffff; border-left: 4px solid #6366f1; border-radius: 0 12px 12px 0; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            <p style="color: #334155; font-size: 1rem; line-height: 1.6; margin: 0;">{snapshot.llm_executive_summary}</p>
        </div>
        """, unsafe_allow_html=True)

        # Two column layout for recommendations and risks
        col1, col2 = st.columns(2)

        # Top Recommendations (high priority first)
        with col1:
            if snapshot.llm_recommendations:
                st.markdown("""
                <div class="section-header" style="margin-top: 0;">
                    <h2 style="font-size: 1rem;">🎯 Top Recommendations</h2>
                </div>
                """, unsafe_allow_html=True)

                sorted_recs = sorted(
                    snapshot.llm_recommendations,
                    key=lambda r: {"high": 0, "medium": 1, "low": 2}.get(r.get("priority", "medium").lower(), 1)
                )
                for rec in sorted_recs[:3]:
                    priority = rec.get("priority", "medium").upper()
                    priority_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "⚪")
                    action = rec.get("action", "")

                    bg_color = {"HIGH": "#fef2f2", "MEDIUM": "#fffbeb", "LOW": "#f0fdf4"}.get(priority, "#f8fafc")
                    border_color = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}.get(priority, "#6366f1")
                    st.markdown(f"""
                    <div style="background: {bg_color}; border-left: 3px solid {border_color}; padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.5rem;">
                        <span style="color: #1e293b; font-weight: 600;">{priority_icon} [{priority}]</span>
                        <span style="color: #334155;"> {action}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # Risk Alerts
        with col2:
            if snapshot.llm_risk_signals:
                st.markdown("""
                <div class="section-header" style="margin-top: 0;">
                    <h2 style="font-size: 1rem;">⚠️ Risk Alerts</h2>
                </div>
                """, unsafe_allow_html=True)

                for risk in snapshot.llm_risk_signals[:3]:
                    severity = risk.get("severity", "medium").upper()
                    signal = risk.get('signal', '')
                    st.markdown(f"""
                    <div style="background: #fef2f2; border-left: 3px solid #ef4444; padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.5rem;">
                        <span style="color: #dc2626; font-weight: 600;">[{severity}]</span>
                        <span style="color: #7f1d1d;"> {signal}</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()

    # Sentiment distribution with visual indicators
    st.markdown("""
    <div class="section-header">
        <h2 style="font-size: 1.1rem;">📊 Sentiment Distribution</h2>
    </div>
    """, unsafe_allow_html=True)

    # Visual sentiment breakdown
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.25rem; background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);">
            <div style="font-size: 2rem; font-weight: 700; color: #ffffff;">{snapshot.primary_positive_pct:.1f}%</div>
            <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">😊 Positive</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.25rem; background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%); border-radius: 12px; box-shadow: 0 4px 15px rgba(107, 114, 128, 0.3);">
            <div style="font-size: 2rem; font-weight: 700; color: #ffffff;">{snapshot.primary_neutral_pct:.1f}%</div>
            <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">😐 Neutral</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.25rem; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); border-radius: 12px; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);">
            <div style="font-size: 2rem; font-weight: 700; color: #ffffff;">{snapshot.primary_negative_pct:.1f}%</div>
            <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">😟 Negative</div>
        </div>
        """, unsafe_allow_html=True)

    # Bar chart
    sentiment_data = pd.DataFrame({
        "Category": ["Positive", "Neutral", "Negative"],
        "Percentage": [
            snapshot.primary_positive_pct,
            snapshot.primary_neutral_pct,
            snapshot.primary_negative_pct,
        ],
    })
    st.bar_chart(sentiment_data.set_index("Category"))

    # Competitors
    if snapshot.competitors:
        st.markdown(f"""
        <div class="section-header">
            <h2 style="font-size: 1.1rem;">🏢 Competitor Analysis ({len(snapshot.competitors)} found)</h2>
        </div>
        """, unsafe_allow_html=True)

        for comp in snapshot.competitors[:7]:
            with st.expander(f"**{comp.competitor.upper()}** - {comp.mention_count} mentions"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Sentiment", f"{comp.avg_sentiment:.3f}")
                with col2:
                    st.metric("Switch TO", comp.switch_to_count)
                with col3:
                    st.metric("Switch FROM", comp.switch_from_count)

                st.write(f"**Positive/Neutral/Negative:** {comp.positive_mentions}/{comp.neutral_mentions}/{comp.negative_mentions}")

                if comp.better_at:
                    st.success(f"✅ Praised for: {', '.join(comp.better_at)}")
                if comp.worse_at:
                    st.error(f"❌ Criticized for: {', '.join(comp.worse_at)}")

                if comp.sample_mentions:
                    st.write("**Sample mention:**")
                    st.caption(comp.sample_mentions[0][:300] + "..." if len(comp.sample_mentions[0]) > 300 else comp.sample_mentions[0])

    # Threats & Opportunities
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚠️ Threats")
        if snapshot.threats:
            for threat in snapshot.threats:
                st.warning(threat)
        else:
            st.info("No significant threats identified")

    with col2:
        st.subheader("💡 Opportunities")
        if snapshot.opportunities:
            for opp in snapshot.opportunities:
                st.success(opp)
        else:
            st.info("No immediate opportunities identified")

    # Pain Points - prefer LLM-verified over rule-based
    st.subheader("😤 Top Pain Points")
    llm_pain_points = st.session_state.get("llm_verified_pain_points", [])
    if llm_pain_points:
        # Show LLM-verified pain points (more accurate)
        for i, pain in enumerate(llm_pain_points, 1):
            if isinstance(pain, dict):
                issue = pain.get("issue", "")
                severity = pain.get("severity", "medium")
                quote = pain.get("sample_quote", "")
                severity_color = {"high": "#dc2626", "medium": "#f59e0b", "low": "#6b7280"}.get(severity, "#f59e0b")
                st.markdown(f"""
                <div style="background: #fef2f2; border-left: 4px solid {severity_color}; padding: 0.75rem 1rem; margin-bottom: 0.5rem; border-radius: 0 8px 8px 0;">
                    <strong>{i}. {issue}</strong>
                    {f'<br><em style="color: #666;">"{quote[:150]}..."</em>' if quote else ''}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"{i}. {pain}")
    elif snapshot.top_pain_points:
        # Fallback to rule-based pain points
        for i, pain in enumerate(snapshot.top_pain_points, 1):
            st.error(f"{i}. {pain}")
    else:
        st.info("No significant pain points found")

    # LLM Deep Analysis
    if snapshot.llm_executive_summary:
        st.markdown("""
        <div class="section-header">
            <h2>🤖 LLM Deep Analysis</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: #ffffff; border-left: 4px solid #6366f1; border-radius: 0 12px 12px 0; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            <h4 style="color: #6366f1; margin: 0 0 0.5rem 0; font-size: 1rem;">Executive Summary</h4>
            <p style="color: #334155; font-size: 1rem; line-height: 1.6; margin: 0;">{snapshot.llm_executive_summary}</p>
        </div>
        """, unsafe_allow_html=True)

        # Brand Perception (new)
        if hasattr(snapshot, 'llm_brand_perception') and snapshot.llm_brand_perception:
            bp = snapshot.llm_brand_perception
            st.subheader("📊 Brand Perception")
            col1, col2 = st.columns(2)

            # Get positive items (try multiple field names)
            drivers = bp.get("sentiment_drivers", {})
            positives = bp.get("brand_strengths") or drivers.get("positive") or bp.get("what_users_love") or []
            negatives = bp.get("brand_weaknesses") or drivers.get("negative") or bp.get("what_users_hate") or []

            with col1:
                st.write("**What Users Love:**")
                for item in positives:
                    st.success(f"✅ {item}")
                if not positives:
                    st.info("No specific positives identified")
            with col2:
                st.write("**What Users Hate:**")
                for item in negatives:
                    st.error(f"❌ {item}")
                if not negatives:
                    st.info("No specific negatives identified")

        # Competitors Mentioned (LLM-identified)
        if snapshot.llm_competitive_insights:
            st.subheader("🏢 Competitors Mentioned (LLM-identified)")
            for comp in snapshot.llm_competitive_insights:
                comp_name = comp.get("competitor", "Unknown")
                with st.expander(f"**{comp_name}**"):
                    # Map LLM field names to display
                    context = comp.get("mention_context") or comp.get("context") or comp.get("perception") or "N/A"
                    perception = comp.get("user_perception") or comp.get("compared_to_brand") or "N/A"
                    why = comp.get("why_mentioned", "")

                    st.write(f"**Context:** {context}")
                    st.write(f"**User Perception:** {perception}")
                    if why:
                        st.write(f"**Why Mentioned:** {why}")

        # Key Themes
        if snapshot.llm_themes:
            st.subheader("💬 Key Themes")
            for theme in snapshot.llm_themes:
                sentiment_color = {
                    "positive": "🟢",
                    "negative": "🔴",
                    "mixed": "🟡",
                }.get(theme.get("sentiment", "mixed"), "⚪")
                theme_name = theme.get("theme", theme.get("name", "Unknown"))
                st.write(f"{sentiment_color} **{theme_name}**")
                st.caption(theme.get("description", ""))
                if theme.get("sample_quote"):
                    st.markdown(f"> _{theme.get('sample_quote')}_")

        # Unanswered Questions
        if snapshot.llm_unanswered_questions:
            st.subheader("❓ Unanswered Questions")
            for q in snapshot.llm_unanswered_questions:
                freq = q.get("frequency", "medium").upper()
                st.write(f"**[{freq}]** {q.get('question', '')}")
                if q.get("opportunity"):
                    st.caption(f"💡 Opportunity: {q.get('opportunity')}")

        # Actionable Recommendations
        if snapshot.llm_recommendations:
            st.subheader("✅ Actionable Recommendations")
            for rec in snapshot.llm_recommendations:
                priority = rec.get("priority", "medium").upper()
                priority_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "⚪")
                st.write(f"{priority_color} **[{priority}]** {rec.get('action', '')}")
                evidence = rec.get("evidence", rec.get("rationale", ""))
                if evidence:
                    st.caption(f"Evidence: {evidence}")

        # Risk Signals
        if snapshot.llm_risk_signals:
            st.subheader("⚠️ Risk Signals")
            for risk in snapshot.llm_risk_signals:
                severity = risk.get("severity", "medium").upper()
                st.error(f"**[{severity}]** {risk.get('signal', '')}")
                evidence = risk.get("evidence", risk.get("suggested_response", ""))
                if evidence:
                    st.caption(f"Evidence: {evidence}")

    # Report metadata
    st.divider()
    st.caption(f"Report ID: {snapshot.snapshot_id}")
    st.caption(f"Generated: {snapshot.measured_at.strftime('%Y-%m-%d %H:%M UTC')}")


def display_opportunities():
    """Display content opportunities tab."""
    st.markdown("""
    <div class="section-header">
        <h2>💡 Content Opportunities</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("High-interest topics with untapped potential based on engagement and LLM analysis.")

    # LLM-Identified Opportunities (primary)
    llm_opps = st.session_state.llm_opportunities
    if llm_opps:
        st.subheader("🤖 LLM-Identified Opportunities")
        st.success(f"Found {len(llm_opps)} content opportunities via LLM analysis")

        for i, opp in enumerate(llm_opps, 1):
            # Handle both dict (from LLM) and object (from scorer) formats
            if isinstance(opp, dict):
                topic = opp.get('topic', f'Opportunity {i}')
                opp_type = opp.get('opportunity_type', 'content_gap')
                target = opp.get('target_audience', '')
                recommended = opp.get('recommended_content', opp.get('recommended_action', ''))
                evidence = opp.get('evidence', '')
            else:
                topic = getattr(opp, 'topic', f'Opportunity {i}')
                opp_type = getattr(opp, 'opportunity_type', 'content_gap')
                target = getattr(opp, 'target_audience', '')
                recommended = getattr(opp, 'recommended_action', '')
                evidence = getattr(opp, 'evidence', '')

            # Color based on opportunity type
            type_colors = {
                "unanswered_question": "🔴",
                "confusion": "🟡",
                "high_interest": "🟢",
                "emerging_trend": "🔵",
            }
            type_icon = type_colors.get(opp_type, "🟢")

            with st.expander(f"{type_icon} **{topic}**", expanded=(i <= 2)):
                # Opportunity type badge
                type_labels = {
                    "unanswered_question": "Unanswered Question",
                    "confusion": "User Confusion",
                    "high_interest": "High Interest Topic",
                    "emerging_trend": "Emerging Trend",
                }
                st.caption(f"Type: {type_labels.get(opp_type, opp_type.replace('_', ' ').title())}")

                if target:
                    st.write(f"**Target Audience:** {target}")

                if recommended:
                    st.info(f"**Recommended Content:** {recommended}")

                if evidence:
                    st.write(f"**Why this matters:** {evidence}")

        st.divider()

    # Engagement-based opportunities (secondary) - only show if LLM opps are empty
    opportunities = st.session_state.opportunities
    # Filter out generic "General discussion" topics
    specific_opps = [
        opp for opp in (opportunities or [])
        if not getattr(opp, 'topic', '').startswith('General discussion')
        and not getattr(opp, 'topic', '').startswith('Trending topic:')
    ]

    if specific_opps and not llm_opps:
        st.subheader("📊 Engagement-Based Opportunities")
        st.caption("Based on comment volume, scores, and engagement patterns")

        for i, opp in enumerate(specific_opps[:5], 1):
            score_pct = getattr(opp, 'opportunity_score', 0.5) * 100
            score_color = "🟢" if score_pct >= 70 else "🟡" if score_pct >= 50 else "🔴"
            topic = getattr(opp, 'topic', f'Opportunity {i}')

            with st.expander(f"{score_color} **{topic}** (Score: {score_pct:.0f}%)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    eng_score = getattr(opp, 'engagement_score', 0.5)
                    st.metric("Engagement Score", f"{eng_score:.0%}")
                with col2:
                    comp_score = getattr(opp, 'competition_score', 0.5)
                    st.metric("Competition", f"{comp_score:.0%}", delta="Low" if comp_score < 0.4 else "High", delta_color="inverse")
                with col3:
                    opp_score = getattr(opp, 'opportunity_score', 0.5)
                    st.metric("Opportunity Score", f"{opp_score:.0%}")

                rec_action = getattr(opp, 'recommended_action', None)
                if rec_action:
                    st.write(f"**Recommended Action:** {rec_action}")

                subreddits = getattr(opp, 'subreddits', [])
                if subreddits:
                    st.write(f"**Subreddits:** {', '.join(subreddits[:5])}")

    # Show engagement summary even if no specific opportunities
    elif opportunities and not llm_opps:
        st.subheader("📊 Engagement Summary")
        best_opp = opportunities[0] if opportunities else None
        if best_opp:
            evidence = getattr(best_opp, 'evidence', {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Comments", evidence.get('total_comments', 0))
            with col2:
                st.metric("Quality Posts", evidence.get('high_quality_posts', 0))
            with col3:
                st.metric("Trend Velocity", f"{evidence.get('trend_velocity', 0):.1f}/day")

            st.info("Enable LLM analysis to identify specific content opportunities from this data.")

    if not llm_opps and not opportunities:
        st.info("No content opportunities identified. Enable LLM analysis for better results.")

        # Show debug info if LLM was attempted
        if st.session_state.llm_debug:
            with st.expander("🔍 LLM Debug Info (click to expand)", expanded=True):
                debug = st.session_state.llm_debug
                st.write(f"**Sample comments sent to LLM:** {debug.get('sample_comments_count', 0)}")

                # Theme extraction
                if debug.get('themes_success'):
                    st.success(f"✅ Theme extraction: {debug.get('themes_count', 0)} themes")
                elif 'themes_error' in debug:
                    st.error(f"❌ Theme extraction failed: {debug.get('themes_error')}")

                # Competitive intelligence
                if debug.get('ci_success'):
                    st.success(f"✅ Competitive intelligence: keys = {debug.get('ci_keys', [])}")
                elif 'ci_error' in debug:
                    st.error(f"❌ Competitive intelligence failed: {debug.get('ci_error')}")

                # Opportunities
                if debug.get('opportunities_success'):
                    st.success(f"✅ Opportunities: {debug.get('opportunities_count', 0)} found")
                elif 'opportunities_error' in debug:
                    st.error(f"❌ Opportunities failed: {debug.get('opportunities_error')}")

                # Show raw CI result if available
                if debug.get('ci_result'):
                    st.write("**Raw LLM response:**")
                    st.json(debug.get('ci_result'))

        if st.session_state.llm_error:
            st.error(f"**LLM Error:** {st.session_state.llm_error}")


def display_kpi_dashboard():
    """Display KPI metrics dashboard."""
    st.markdown("""
    <div class="section-header">
        <h2>📈 KPI Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Key performance indicators for tracking brand health and engagement metrics.")

    kpis = st.session_state.kpi_metrics

    if not kpis:
        st.warning("No KPI metrics available.")
        return

    # Health Score - Visual Badge
    health = kpis.get("health_score", 0)
    health_class = "health-good" if health >= 0.7 else "health-warning" if health >= 0.4 else "health-critical"
    health_label = "Healthy" if health >= 0.7 else "Needs Attention" if health >= 0.4 else "Critical"

    col_health, col_details = st.columns([1, 2])

    with col_health:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <div class="health-badge {health_class}">
                {health:.0%}
            </div>
            <p style="color: #a0a0a0; margin-top: 1rem; font-size: 1.1rem;">{health_label}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_details:
        st.markdown("#### Health Score Components")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Positive Sentiment:** {kpis.get('sentiment_positive_pct', 0):.1f}% (40% weight)")
            st.progress(kpis.get('sentiment_positive_pct', 0) / 100)
        with cols[1]:
            st.markdown(f"**Author Diversity:** {kpis.get('author_diversity', 0):.0%} (20% weight)")
            st.progress(kpis.get('author_diversity', 0))
        with cols[0]:
            engagement = min(kpis.get('comments_per_post', 0) / 10, 1)
            st.markdown(f"**Engagement:** {engagement:.0%} (20% weight)")
            st.progress(engagement)
        with cols[1]:
            stability = 1 - kpis.get('sentiment_volatility', 0)
            st.markdown(f"**Stability:** {stability:.0%} (20% weight)")
            st.progress(max(0, stability))

    st.divider()

    # KPI Categories with icons
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="section-header" style="margin-top: 0;">
            <h2 style="font-size: 1.1rem;">📊 Volume Metrics</h2>
        </div>
        """, unsafe_allow_html=True)
        st.metric("💬 Total Comments", f"{kpis.get('total_comments', 0):,}")
        st.metric("📝 Total Posts", f"{kpis.get('total_posts', 0):,}")
        st.metric("📈 Comments per Post", f"{kpis.get('comments_per_post', 0):.1f}")

    with col2:
        st.markdown("""
        <div class="section-header" style="margin-top: 0;">
            <h2 style="font-size: 1.1rem;">⚡ Engagement Metrics</h2>
        </div>
        """, unsafe_allow_html=True)
        st.metric("⭐ Avg Comment Score", f"{kpis.get('avg_comment_score', 0):.1f}")
        st.metric("🏆 Max Comment Score", f"{kpis.get('max_comment_score', 0):,}")
        st.metric("📊 Avg Post Score", f"{kpis.get('avg_post_score', 0):.1f}")

    with col3:
        st.markdown("""
        <div class="section-header" style="margin-top: 0;">
            <h2 style="font-size: 1.1rem;">👥 Diversity Metrics</h2>
        </div>
        """, unsafe_allow_html=True)
        st.metric("👤 Unique Authors", f"{kpis.get('unique_authors', 0):,}")
        st.metric("🎯 Author Diversity", f"{kpis.get('author_diversity', 0):.0%}")
        st.metric("📍 Unique Subreddits", f"{kpis.get('unique_subreddits', 0):,}")

    # Sentiment KPIs
    st.markdown("### 😊 Sentiment KPIs")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Sentiment", f"{kpis.get('sentiment_mean', 0):.3f}")
    with col2:
        st.metric("Positive %", f"{kpis.get('sentiment_positive_pct', 0):.1f}%")
    with col3:
        st.metric("Negative %", f"{kpis.get('sentiment_negative_pct', 0):.1f}%")
    with col4:
        st.metric("Volatility", f"{kpis.get('sentiment_volatility', 0):.3f}")

    # Top Subreddits
    top_subs = kpis.get("top_subreddits", {})
    if top_subs:
        st.markdown("### 📍 Top Subreddits")
        sub_df = pd.DataFrame([
            {"Subreddit": f"r/{k}", "Comments": v}
            for k, v in top_subs.items()
        ])
        st.bar_chart(sub_df.set_index("Subreddit"))

    # KPI Thresholds / Alerts
    st.markdown("### ⚠️ Alert Thresholds")
    alerts = []
    if kpis.get("sentiment_mean", 0) < -0.2:
        alerts.append(("🔴 CRITICAL", "Mean sentiment below -0.2"))
    elif kpis.get("sentiment_mean", 0) < 0:
        alerts.append(("🟡 WARNING", "Mean sentiment is negative"))

    if kpis.get("sentiment_negative_pct", 0) > 50:
        alerts.append(("🔴 CRITICAL", "More than 50% negative comments"))
    elif kpis.get("sentiment_negative_pct", 0) > 30:
        alerts.append(("🟡 WARNING", "More than 30% negative comments"))

    if kpis.get("author_diversity", 0) < 0.3:
        alerts.append(("🟡 WARNING", "Low author diversity - few unique voices"))

    if alerts:
        for level, message in alerts:
            if "CRITICAL" in level:
                st.error(f"{level}: {message}")
            else:
                st.warning(f"{level}: {message}")
    else:
        st.success("✅ All KPIs within healthy thresholds")


def display_recommendations(snapshot):
    """Display comprehensive recommendations tab."""
    st.markdown("""
    <div class="section-header">
        <h2>🎯 Actionable Recommendations</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Prioritized actions based on LLM analysis of Reddit discussions.")

    if not snapshot.llm_recommendations and not snapshot.llm_unanswered_questions:
        st.info("Enable LLM Deep Analysis to generate recommendations.")

        # Show debug info if LLM was attempted
        if st.session_state.llm_debug:
            with st.expander("🔍 LLM Debug Info", expanded=True):
                debug = st.session_state.llm_debug

                if debug.get('ci_success'):
                    st.success(f"✅ LLM call succeeded but returned: {debug.get('ci_keys', [])}")
                    if debug.get('ci_result'):
                        recs = debug['ci_result'].get('actionable_recommendations', [])
                        questions = debug['ci_result'].get('unanswered_questions', [])
                        st.write(f"- Recommendations in response: {len(recs)}")
                        st.write(f"- Unanswered questions in response: {len(questions)}")
                elif 'ci_error' in debug:
                    st.error(f"❌ LLM call failed: {debug.get('ci_error')}")

        if st.session_state.llm_error:
            st.error(f"**LLM Error:** {st.session_state.llm_error}")

        return

    # Priority Recommendations
    if snapshot.llm_recommendations:
        st.subheader("📋 Priority Actions")

        # Group by priority
        high_priority = [r for r in snapshot.llm_recommendations if r.get("priority", "").lower() == "high"]
        medium_priority = [r for r in snapshot.llm_recommendations if r.get("priority", "").lower() == "medium"]
        low_priority = [r for r in snapshot.llm_recommendations if r.get("priority", "").lower() == "low"]

        if high_priority:
            st.markdown("#### 🔴 High Priority")
            for rec in high_priority:
                with st.container():
                    st.error(f"**{rec.get('action', '')}**")
                    if rec.get("evidence"):
                        st.caption(f"Evidence: {rec.get('evidence')}")

        if medium_priority:
            st.markdown("#### 🟡 Medium Priority")
            for rec in medium_priority:
                with st.container():
                    st.warning(f"**{rec.get('action', '')}**")
                    if rec.get("evidence"):
                        st.caption(f"Evidence: {rec.get('evidence')}")

        if low_priority:
            st.markdown("#### 🟢 Low Priority")
            for rec in low_priority:
                with st.container():
                    st.success(f"**{rec.get('action', '')}**")
                    if rec.get("evidence"):
                        st.caption(f"Evidence: {rec.get('evidence')}")

    st.divider()

    # Unanswered Questions = Content Opportunities
    if snapshot.llm_unanswered_questions:
        st.subheader("❓ Unanswered Questions (Content Opportunities)")
        st.markdown("Questions users are asking that aren't being answered - great content opportunities!")

        for q in snapshot.llm_unanswered_questions:
            freq = q.get("frequency", "medium").upper()
            freq_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(freq, "⚪")

            with st.expander(f"{freq_color} **{q.get('question', '')}**"):
                st.write(f"**Frequency:** {freq}")
                if q.get("opportunity"):
                    st.info(f"💡 **How to address:** {q.get('opportunity')}")

    st.divider()

    # Risk Signals
    if snapshot.llm_risk_signals:
        st.subheader("⚠️ Risk Signals to Monitor")

        for risk in snapshot.llm_risk_signals:
            severity = risk.get("severity", "medium").upper()
            severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")

            with st.expander(f"{severity_icon} **[{severity}]** {risk.get('signal', '')}"):
                if risk.get("evidence"):
                    st.write(f"**Evidence:** {risk.get('evidence')}")
                st.markdown("**Suggested monitoring:** Track this in subsequent analyses")

    # Key Themes for Content Strategy
    if snapshot.llm_themes:
        st.divider()
        st.subheader("💬 Key Themes for Content Strategy")

        for theme in snapshot.llm_themes:
            sentiment = theme.get("sentiment", "mixed")
            sentiment_icon = {"positive": "🟢", "negative": "🔴", "mixed": "🟡"}.get(sentiment, "⚪")
            theme_name = theme.get("theme", theme.get("name", "Unknown"))

            with st.expander(f"{sentiment_icon} **{theme_name}**"):
                st.write(theme.get("description", ""))
                if theme.get("sample_quote"):
                    st.markdown(f"> _{theme.get('sample_quote')}_")
                st.caption(f"Sentiment: {sentiment}")


def display_response_queue():
    """Display posts needing attention - high engagement + negative sentiment."""
    st.markdown("""
    <div class="section-header">
        <h2>🚨 Posts Needing Attention</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("High-engagement posts with negative sentiment that may need a response.")

    comments_df = st.session_state.comments_df
    posts_df = st.session_state.posts_df

    if comments_df is None or comments_df.empty:
        st.warning("No comment data available. Run an analysis first.")
        return

    # Calculate attention score: engagement * negative sentiment strength
    df = comments_df.copy()

    # Need sentiment_score and score columns
    if "sentiment_score" not in df.columns or "score" not in df.columns:
        st.warning("Required columns (sentiment_score, score) not available.")
        return

    # Filter to negative comments only
    negative_df = df[df["sentiment_label"] == "negative"].copy()

    if negative_df.empty:
        st.success("No negative posts requiring attention!")
        return

    # Calculate attention priority score
    # Higher engagement (score) + more negative sentiment = higher priority
    # Normalize scores for comparison
    max_score = negative_df["score"].max() if negative_df["score"].max() > 0 else 1
    negative_df["engagement_norm"] = negative_df["score"] / max_score

    # Sentiment strength (more negative = higher priority)
    negative_df["sentiment_strength"] = negative_df["sentiment_score"].abs()

    # Attention score: weighted combination
    negative_df["attention_score"] = (
        negative_df["engagement_norm"] * 0.6 +  # 60% weight on engagement
        negative_df["sentiment_strength"] * 0.4  # 40% weight on negativity
    )

    # Sort by attention score
    priority_df = negative_df.nlargest(20, "attention_score")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Negative Comments", len(negative_df))
    with col2:
        st.metric("High Priority", len(priority_df[priority_df["attention_score"] >= 0.5]))
    with col3:
        avg_score = priority_df["score"].mean()
        st.metric("Avg Engagement (Top 20)", f"{avg_score:.0f}")
    with col4:
        if "subreddit" in priority_df.columns:
            top_sub = priority_df["subreddit"].mode().iloc[0] if len(priority_df) > 0 else "N/A"
            st.metric("Most Active Subreddit", f"r/{top_sub}")

    st.divider()

    # Priority filter
    priority_filter = st.selectbox(
        "Filter by Priority",
        options=["All", "High (≥0.7)", "Medium (0.4-0.7)", "Low (<0.4)"]
    )

    if priority_filter == "High (≥0.7)":
        priority_df = priority_df[priority_df["attention_score"] >= 0.7]
    elif priority_filter == "Medium (0.4-0.7)":
        priority_df = priority_df[(priority_df["attention_score"] >= 0.4) & (priority_df["attention_score"] < 0.7)]
    elif priority_filter == "Low (<0.4)":
        priority_df = priority_df[priority_df["attention_score"] < 0.4]

    # Display each post needing attention
    st.subheader(f"📋 Response Queue ({len(priority_df)} items)")

    for idx, row in priority_df.iterrows():
        attention = row["attention_score"]
        if attention >= 0.7:
            priority_icon = "🔴"
            priority_label = "HIGH"
        elif attention >= 0.4:
            priority_icon = "🟡"
            priority_label = "MEDIUM"
        else:
            priority_icon = "🟢"
            priority_label = "LOW"

        # Get post info if available
        post_id = row.get("post_id", "")
        subreddit = row.get("subreddit", "unknown")

        with st.expander(
            f"{priority_icon} [{priority_label}] r/{subreddit} | Score: {row['score']} | Sentiment: {row['sentiment_score']:.2f}"
        ):
            # Comment content
            st.markdown("**Comment:**")
            body = row.get("body", "")
            st.error(body[:500] + "..." if len(body) > 500 else body)

            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Engagement Score:** {row['score']}")
            with col2:
                st.write(f"**Sentiment:** {row['sentiment_score']:.3f}")
            with col3:
                st.write(f"**Attention Score:** {attention:.2f}")

            # Author info
            author = row.get("author", "[deleted]")
            st.write(f"**Author:** u/{author}")

            # Timestamp
            if "created_utc" in row and row["created_utc"] is not None:
                st.write(f"**Posted:** {row['created_utc']}")

            # Link to post (if we have the post data)
            if posts_df is not None and not posts_df.empty and post_id:
                matching_post = posts_df[posts_df["id"] == post_id]
                if len(matching_post) > 0:
                    post = matching_post.iloc[0]
                    st.write(f"**Post Title:** {post.get('title', 'N/A')}")
                    if "url" in post:
                        st.markdown(f"[View on Reddit]({post['url']})")

            st.divider()

            # Suggested response actions
            st.markdown("**Suggested Actions:**")
            st.info("""
            - 👀 **Monitor** - Track if others pile on
            - 💬 **Respond** - Address the concern directly
            - 📊 **Escalate** - Flag to customer service team
            - 🔍 **Investigate** - Check if this is a systemic issue
            """)

    # Export option
    st.divider()
    if st.button("📥 Export Response Queue to CSV"):
        export_df = priority_df[["body", "score", "sentiment_score", "attention_score", "subreddit", "author"]].copy()
        export_df.columns = ["Comment", "Engagement", "Sentiment", "Priority Score", "Subreddit", "Author"]
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="response_queue.csv",
            mime="text/csv",
        )


def display_executive_report(snapshot, client_config):
    """Display detailed executive report."""
    st.markdown("""
    <div class="section-header">
        <h2>📄 Executive Report</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Comprehensive analysis summary for stakeholders.")

    # Report header
    st.markdown(f"""
    ---
    ## {client_config.primary_brand.title()} - Reddit Intelligence Report

    **Report Date:** {snapshot.measured_at.strftime('%B %d, %Y')}
    **Analysis Period:** {snapshot.period_days} days
    **Industry:** {client_config.industry.replace('_', ' ').title()}

    ---
    """)

    # Executive Summary Section
    st.markdown("### 📋 Executive Summary")

    if snapshot.llm_executive_summary:
        st.info(snapshot.llm_executive_summary)
    else:
        # Generate a summary from the data
        sentiment_status = "positive" if snapshot.primary_sentiment > 0.1 else "negative" if snapshot.primary_sentiment < -0.1 else "neutral"
        st.info(f"""
        Analysis of {snapshot.comments_analyzed:,} comments across {snapshot.posts_analyzed:,} posts reveals
        **{sentiment_status}** overall sentiment ({snapshot.primary_sentiment:.2f}).

        - **{snapshot.primary_positive_pct:.1f}%** of comments are positive
        - **{snapshot.primary_negative_pct:.1f}%** of comments are negative
        - **{snapshot.total_competitor_mentions}** competitor mentions detected
        """)

    # Key Metrics Dashboard
    st.markdown("### 📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Comments Analyzed", f"{snapshot.comments_analyzed:,}")
    with col2:
        st.metric("Posts Analyzed", f"{snapshot.posts_analyzed:,}")
    with col3:
        delta_color = "normal" if snapshot.primary_sentiment >= 0 else "inverse"
        st.metric("Mean Sentiment", f"{snapshot.primary_sentiment:.3f}")
    with col4:
        st.metric("Competitor Mentions", snapshot.total_competitor_mentions)

    kpis = st.session_state.kpi_metrics
    if kpis:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            health = kpis.get("health_score", 0)
            health_icon = "🟢" if health >= 0.7 else "🟡" if health >= 0.4 else "🔴"
            st.metric(f"{health_icon} Health Score", f"{health:.0%}")
        with col2:
            st.metric("Unique Authors", kpis.get("unique_authors", 0))
        with col3:
            st.metric("Unique Subreddits", kpis.get("unique_subreddits", 0))
        with col4:
            st.metric("Avg Comment Score", f"{kpis.get('avg_comment_score', 0):.1f}")

    # Sentiment Breakdown
    st.markdown("### 😊 Sentiment Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        | Category | Percentage |
        |----------|------------|
        | Positive | {:.1f}% |
        | Neutral  | {:.1f}% |
        | Negative | {:.1f}% |
        """.format(
            snapshot.primary_positive_pct,
            snapshot.primary_neutral_pct,
            snapshot.primary_negative_pct
        ))

    with col2:
        sentiment_data = pd.DataFrame({
            "Category": ["Positive", "Neutral", "Negative"],
            "Percentage": [
                snapshot.primary_positive_pct,
                snapshot.primary_neutral_pct,
                snapshot.primary_negative_pct,
            ],
        })
        st.bar_chart(sentiment_data.set_index("Category"))

    # Brand Perception
    if hasattr(snapshot, 'llm_brand_perception') and snapshot.llm_brand_perception:
        st.markdown("### 💭 Brand Perception")
        bp = snapshot.llm_brand_perception

        # Get positive/negative items (try multiple field names)
        drivers = bp.get("sentiment_drivers", {})
        positives = bp.get("brand_strengths") or drivers.get("positive") or bp.get("what_users_love") or []
        negatives = bp.get("brand_weaknesses") or drivers.get("negative") or bp.get("what_users_hate") or []

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**What Users Love:**")
            for item in positives:
                st.success(f"✅ {item}")
            if not positives:
                st.info("No specific positives identified")
        with col2:
            st.markdown("**What Users Dislike:**")
            for item in negatives:
                st.error(f"❌ {item}")
            if not negatives:
                st.info("No specific negatives identified")

    # Competitive Landscape
    st.markdown("### 🏢 Competitive Landscape")

    if snapshot.competitors:
        # Summary table
        comp_data = []
        for comp in snapshot.competitors[:10]:
            comp_data.append({
                "Competitor": comp.competitor.title(),
                "Mentions": comp.mention_count,
                "Sentiment": f"{comp.avg_sentiment:.2f}",
                "Positive": comp.positive_mentions,
                "Negative": comp.negative_mentions,
                "Switch To": comp.switch_to_count,
                "Switch From": comp.switch_from_count,
            })

        if comp_data:
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Detailed competitor insights
        for comp in snapshot.competitors[:5]:
            with st.expander(f"**{comp.competitor.title()}** - Detailed Analysis"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Mentions", comp.mention_count)
                with col2:
                    st.metric("Avg Sentiment", f"{comp.avg_sentiment:.3f}")
                with col3:
                    net_switch = comp.switch_from_count - comp.switch_to_count
                    st.metric("Net Switch (Favorable)", net_switch)

                if comp.better_at:
                    st.success(f"**Strengths:** {', '.join(comp.better_at)}")
                if comp.worse_at:
                    st.error(f"**Weaknesses:** {', '.join(comp.worse_at)}")
    else:
        st.info("No competitor mentions detected in this analysis.")

    # LLM-Identified Competitors
    if snapshot.llm_competitive_insights:
        st.markdown("#### AI-Identified Competitor Insights")
        for insight in snapshot.llm_competitive_insights:
            comp_name = insight.get("competitor", "Unknown")
            st.markdown(f"**{comp_name}:** {insight.get('context', insight.get('perception', 'N/A'))}")

    # Key Themes
    st.markdown("### 💬 Key Discussion Themes")

    if snapshot.llm_themes:
        for theme in snapshot.llm_themes:
            sentiment = theme.get("sentiment", "mixed")
            sentiment_icon = {"positive": "🟢", "negative": "🔴", "mixed": "🟡"}.get(sentiment, "⚪")
            theme_name = theme.get("theme", theme.get("name", "Unknown"))

            st.markdown(f"{sentiment_icon} **{theme_name}**")
            st.caption(theme.get("description", ""))
            if theme.get("sample_quote"):
                st.markdown(f"> _{theme.get('sample_quote')[:200]}..._" if len(theme.get('sample_quote', '')) > 200 else f"> _{theme.get('sample_quote')}_")
    else:
        st.info("Enable LLM analysis for theme extraction.")

    # Threats & Opportunities
    st.markdown("### ⚠️ Threats & Opportunities")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Threats:**")
        if snapshot.threats:
            for threat in snapshot.threats:
                st.error(f"⚠️ {threat}")
        else:
            st.success("No significant threats identified")

    with col2:
        st.markdown("**Opportunities:**")
        if snapshot.opportunities:
            for opp in snapshot.opportunities:
                st.success(f"💡 {opp}")
        else:
            st.info("No immediate opportunities identified")

    # Risk Signals
    if snapshot.llm_risk_signals:
        st.markdown("### 🚨 Risk Signals")
        for risk in snapshot.llm_risk_signals:
            severity = risk.get("severity", "medium").upper()
            severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")
            st.markdown(f"{severity_icon} **[{severity}]** {risk.get('signal', '')}")
            if risk.get("evidence"):
                st.caption(f"Evidence: {risk.get('evidence')}")

    # Pain Points - prefer LLM-verified over rule-based
    st.markdown("### 😤 Customer Pain Points")
    llm_pain_points = st.session_state.get("llm_verified_pain_points", [])
    if llm_pain_points:
        for i, pain in enumerate(llm_pain_points, 1):
            if isinstance(pain, dict):
                issue = pain.get("issue", "")
                severity = pain.get("severity", "medium")
                quote = pain.get("sample_quote", "")
                impact = pain.get("business_impact", "")
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "🟡")
                st.markdown(f"{severity_icon} **{issue}**")
                if quote:
                    st.caption(f'"{quote[:200]}..."')
                if impact:
                    st.caption(f"Impact: {impact}")
            else:
                st.error(f"{i}. {pain}")
    elif snapshot.top_pain_points:
        for i, pain in enumerate(snapshot.top_pain_points, 1):
            st.error(f"{i}. {pain}")
    else:
        st.success("No significant pain points identified")

    # Recommendations
    st.markdown("### 🎯 Recommendations")

    if snapshot.llm_recommendations:
        # Sort by priority
        sorted_recs = sorted(
            snapshot.llm_recommendations,
            key=lambda r: {"high": 0, "medium": 1, "low": 2}.get(r.get("priority", "medium").lower(), 1)
        )

        for rec in sorted_recs:
            priority = rec.get("priority", "medium").upper()
            priority_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "⚪")
            st.markdown(f"{priority_icon} **[{priority}]** {rec.get('action', '')}")
            if rec.get("evidence"):
                st.caption(f"_Rationale: {rec.get('evidence')}_")
    else:
        st.info("Enable LLM analysis for personalized recommendations.")

    # Unanswered Questions
    if snapshot.llm_unanswered_questions:
        st.markdown("### ❓ Unanswered Customer Questions")
        st.markdown("These are questions customers are asking that aren't being addressed:")

        for q in snapshot.llm_unanswered_questions:
            freq = q.get("frequency", "medium").upper()
            st.markdown(f"- **[{freq}]** {q.get('question', '')}")
            if q.get("opportunity"):
                st.caption(f"  💡 _Opportunity: {q.get('opportunity')}_")

    # Top Subreddits
    st.markdown("### 📍 Top Subreddits")

    if kpis and kpis.get("top_subreddits"):
        sub_data = pd.DataFrame([
            {"Subreddit": f"r/{k}", "Comments": v}
            for k, v in kpis["top_subreddits"].items()
        ])
        st.bar_chart(sub_data.set_index("Subreddit"))
    else:
        st.info("Subreddit data not available")

    # Report Footer
    st.markdown("---")
    st.markdown(f"""
    **Report Metadata:**
    - Report ID: `{snapshot.snapshot_id}`
    - Generated: {snapshot.measured_at.strftime('%Y-%m-%d %H:%M UTC')}
    - Data Period: {snapshot.period_start.strftime('%Y-%m-%d')} to {snapshot.period_end.strftime('%Y-%m-%d')}
    - Analysis Engine: HuggingFace Sentiment + OpenRouter LLM
    """)

    # Export options
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        # Export as JSON
        if st.button("📥 Export Report Data (JSON)"):
            import json
            report_data = {
                "brand": client_config.primary_brand,
                "report_date": snapshot.measured_at.isoformat(),
                "period_days": snapshot.period_days,
                "metrics": {
                    "comments_analyzed": snapshot.comments_analyzed,
                    "posts_analyzed": snapshot.posts_analyzed,
                    "sentiment_mean": snapshot.primary_sentiment,
                    "positive_pct": snapshot.primary_positive_pct,
                    "negative_pct": snapshot.primary_negative_pct,
                    "competitor_mentions": snapshot.total_competitor_mentions,
                },
                "executive_summary": snapshot.llm_executive_summary,
                "themes": snapshot.llm_themes,
                "recommendations": snapshot.llm_recommendations,
                "risk_signals": snapshot.llm_risk_signals,
                "threats": snapshot.threats,
                "opportunities": snapshot.opportunities,
                "pain_points": st.session_state.get("llm_verified_pain_points") or snapshot.top_pain_points,
            }
            json_str = json.dumps(report_data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"report_{client_config.primary_brand}_{snapshot.measured_at.strftime('%Y%m%d')}.json",
                mime="application/json",
            )

    with col2:
        st.info("💡 Tip: Use browser print (Ctrl+P) to save as PDF")


def extract_keywords_from_comments(comments_df, primary_brand: str, max_words: int = 100) -> list[dict]:
    """Extract keywords from comments for word cloud visualization.

    Returns list of dicts with 'name' and 'value' keys for ECharts.
    """
    if comments_df is None or comments_df.empty or "body" not in comments_df.columns:
        return []

    # Stopwords for filtering
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "must", "shall", "can", "this", "that", "these", "those", "i", "you", "he", "she",
        "it", "we", "they", "what", "which", "who", "whom", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "also", "now", "here", "there", "then", "if", "as", "any", "my", "your", "his",
        "her", "its", "our", "their", "about", "up", "out", "into", "over", "after",
        "before", "between", "under", "again", "further", "once", "during", "while",
        "because", "although", "through", "until", "unless", "since", "even", "though",
        "get", "got", "getting", "like", "one", "two", "first", "new", "way", "use",
        "used", "using", "make", "made", "making", "know", "think", "see", "come",
        "go", "going", "take", "want", "look", "give", "day", "good", "back", "much",
        "well", "year", "work", "still", "right", "people", "thing", "things", "really",
        "say", "said", "need", "try", "lot", "let", "something", "anything", "everything",
        "nothing", "someone", "anyone", "everyone", "edit", "deleted", "removed", "http",
        "https", "www", "com", "org", "amp", "reddit", "subreddit"
    }

    # Combine all comment bodies
    all_text = " ".join(comments_df["body"].fillna("").astype(str).tolist())

    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())

    # Filter stopwords and short words
    brand_lower = primary_brand.lower()
    filtered_words = [
        w for w in words
        if w not in stopwords
        and len(w) >= 3
        and not w.isdigit()
    ]

    # Count frequencies
    word_counts = Counter(filtered_words)

    # Convert to ECharts format
    word_data = [
        {"name": word, "value": count}
        for word, count in word_counts.most_common(max_words)
    ]

    return word_data


def display_word_cloud():
    """Display interactive word cloud using ECharts."""
    st.markdown("""
    <div class="section-header">
        <h2>☁️ Discussion Word Cloud</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Visualize the most frequently discussed topics in the analyzed comments.")

    # Check if ECharts is available
    if not ECHARTS_AVAILABLE:
        st.warning("Word cloud requires streamlit-echarts. Install with: `pip install streamlit-echarts`")
        return

    comments_df = st.session_state.get("comments_df")
    snapshot = st.session_state.get("snapshot")

    if comments_df is None or comments_df.empty:
        st.warning("No comment data available. Run an analysis first.")
        return

    # Get primary brand for context
    primary_brand = ""
    if snapshot and hasattr(snapshot, "primary_brand"):
        primary_brand = snapshot.primary_brand

    # Extract keywords
    word_data = extract_keywords_from_comments(comments_df, primary_brand, max_words=80)

    if not word_data:
        st.info("Not enough text data to generate word cloud.")
        return

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Words", len(word_data))
    with col2:
        st.metric("Top Word", word_data[0]["name"] if word_data else "N/A")
    with col3:
        st.metric("Top Word Count", word_data[0]["value"] if word_data else 0)

    # ECharts word cloud option
    option = {
        "tooltip": {
            "show": True,
            "formatter": "{b}: {c} mentions"
        },
        "series": [{
            "type": "wordCloud",
            "shape": "circle",
            "keepAspect": False,
            "left": "center",
            "top": "center",
            "width": "90%",
            "height": "90%",
            "right": None,
            "bottom": None,
            "sizeRange": [14, 60],
            "rotationRange": [-45, 45],
            "rotationStep": 15,
            "gridSize": 8,
            "drawOutOfBound": False,
            "layoutAnimation": True,
            "textStyle": {
                "fontFamily": "sans-serif",
                "fontWeight": "bold",
                "color": "function () { return 'hsl(' + Math.random() * 360 + ', 70%, 50%)'; }"
            },
            "emphasis": {
                "focus": "self",
                "textStyle": {
                    "textShadowBlur": 10,
                    "textShadowColor": "#333"
                }
            },
            "data": word_data
        }]
    }

    # Render the word cloud
    st_echarts(option, height="500px", key="wordcloud")

    # Show top words table
    with st.expander("📊 Top 20 Words"):
        top_words_df = pd.DataFrame(word_data[:20])
        top_words_df.columns = ["Word", "Mentions"]
        top_words_df.index = range(1, len(top_words_df) + 1)
        st.dataframe(top_words_df, use_container_width=True)

    # Sentiment breakdown by top words
    if "sentiment_label" in comments_df.columns:
        with st.expander("🎭 Sentiment by Top Keywords"):
            st.markdown("See how sentiment varies for comments containing top keywords:")

            top_5_words = [w["name"] for w in word_data[:5]]
            sentiment_by_word = []

            for word in top_5_words:
                mask = comments_df["body"].str.lower().str.contains(word, na=False)
                word_comments = comments_df[mask]
                if len(word_comments) > 0:
                    pos = (word_comments["sentiment_label"] == "positive").sum()
                    neu = (word_comments["sentiment_label"] == "neutral").sum()
                    neg = (word_comments["sentiment_label"] == "negative").sum()
                    total = len(word_comments)
                    sentiment_by_word.append({
                        "Word": word,
                        "Positive %": round(pos / total * 100, 1),
                        "Neutral %": round(neu / total * 100, 1),
                        "Negative %": round(neg / total * 100, 1),
                        "Total Mentions": total
                    })

            if sentiment_by_word:
                st.dataframe(pd.DataFrame(sentiment_by_word), use_container_width=True)


def display_shift_tracking():
    """Display sentiment shift tracking tab."""
    st.markdown("""
    <div class="section-header">
        <h2>📉 Sentiment Shift Tracking</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Track how sentiment changes over time to identify trends and anomalies.")

    # Since we don't have historical data yet, show current analysis context
    snapshot = st.session_state.snapshot
    kpis = st.session_state.kpi_metrics

    st.info("""
    **Shift tracking requires historical data.**

    To enable shift detection:
    1. Run analyses regularly (daily/weekly)
    2. Save results to BigQuery with the "Save to BigQuery" button
    3. Historical data enables week-over-week and month-over-month comparisons
    """)

    # Show current baseline
    st.subheader("📊 Current Baseline")
    st.markdown("These metrics will be compared against future analyses to detect shifts.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Sentiment",
            f"{snapshot.primary_sentiment:.3f}",
            help="Baseline for shift detection"
        )
    with col2:
        st.metric(
            "Positive Ratio",
            f"{snapshot.primary_positive_pct:.1f}%",
            help="Baseline positive sentiment %"
        )
    with col3:
        st.metric(
            "Volume",
            snapshot.comments_analyzed,
            help="Baseline comment volume"
        )

    # Shift Detection Explanation
    st.subheader("🔍 How Shift Detection Works")
    st.markdown("""
    When historical data is available, the system will:

    1. **Compare Time Periods** - Week-over-week, month-over-month
    2. **Statistical Significance** - Uses Welch's t-test to determine if changes are significant (p < 0.05)
    3. **Minimum Threshold** - Only flags shifts > 5% to filter noise
    4. **Generate Explanations** - LLM analyzes comments to explain *why* sentiment shifted

    **Example output:**
    > "Sentiment declined 15% week-over-week (p=0.023). Analysis suggests this is driven by
    > customer service complaints following the recent app update."
    """)

    # Mock shift visualization
    st.subheader("📈 Sentiment Trend (Simulated)")
    st.caption("This chart shows how trends will appear once historical data is collected")

    # Create mock trend data
    import numpy as np
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    baseline = snapshot.primary_sentiment
    mock_sentiment = baseline + np.random.normal(0, 0.1, 30).cumsum() * 0.01
    mock_sentiment = np.clip(mock_sentiment, -1, 1)

    trend_df = pd.DataFrame({
        "Date": dates,
        "Sentiment": mock_sentiment
    })
    st.line_chart(trend_df.set_index("Date"))
    st.caption("⚠️ Simulated data for demonstration. Real trends require historical analysis runs.")


# Main UI - Custom Header
st.markdown("""
<div class="main-header">
    <h1>🔍 Reddit Competitive Intelligence</h1>
    <p>Analyze Reddit discussions for competitive insights using HuggingFace sentiment + LLM deep analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
        <h2 style="background: linear-gradient(90deg, #6366f1, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.4rem; margin: 0;">⚙️ Configuration</h2>
    </div>
    """, unsafe_allow_html=True)

    # Try to load existing clients
    try:
        from reddit_sentiment.config.clients import get_registry, ClientConfig

        registry = get_registry()
        clients = registry.list_clients(enabled_only=False)
        client_options = {c.client_id: c for c in clients}
    except Exception:
        client_options = {}
        # Define a simple ClientConfig class for manual configuration
        from dataclasses import dataclass, field
        from typing import List

        @dataclass
        class ClientConfig:
            client_id: str
            client_name: str
            primary_brand: str
            search_keywords: List[str] = field(default_factory=list)
            competitors: List[str] = field(default_factory=list)
            industry: str = "other"
            enabled: bool = True

    if client_options:
        selected_client = st.selectbox(
            "Select Client",
            options=list(client_options.keys()),
            format_func=lambda x: f"{client_options[x].client_name} ({x})",
        )
        client_config = client_options[selected_client]
        st.info(f"Brand: {client_config.primary_brand}")
        st.info(f"Industry: {client_config.industry}")
        st.info(f"Competitors: {len(client_config.competitors)}")
    else:
        st.warning("No clients configured. Using manual config.")
        selected_client = None

        # Manual configuration
        brand = st.text_input("Primary Brand", value="Capital One")
        industry = st.selectbox("Industry", ["financial_services", "tech", "retail", "other"])
        keywords = st.text_input("Search Keywords (comma-separated)", value="capital one, capitalone")
        competitors = st.text_area(
            "Competitors (one per line)",
            value="chase\namerican express\nbank of america\ndiscover\nciti",
        )

        client_config = ClientConfig(
            client_id="manual_config",
            client_name="Manual Configuration",
            primary_brand=brand,
            search_keywords=[k.strip() for k in keywords.split(",")],
            competitors=[c.strip() for c in competitors.strip().split("\n") if c.strip()],
            industry=industry,
        )

    st.divider()

    # Analysis parameters
    days_back = st.slider("Days to analyze", min_value=1, max_value=90, value=30)
    post_limit = st.slider("Max posts to fetch", min_value=50, max_value=1000, value=500)

    # Search mode
    detailed_search = st.checkbox(
        "Use detailed keyword search",
        value=False,
        help="Off: Search for brand name only (more results). On: Use all specific keywords (fewer but more targeted results)"
    )

    # Keyword count slider (only shown when detailed search is enabled)
    if detailed_search and client_config.search_keywords:
        max_keywords = st.slider(
            "Keywords to search",
            min_value=1,
            max_value=len(client_config.search_keywords),
            value=min(5, len(client_config.search_keywords)),
            help=f"Search up to {len(client_config.search_keywords)} keywords. More keywords = more data but slower."
        )
        st.caption(f"Available: {', '.join(client_config.search_keywords[:max_keywords])}")
    else:
        max_keywords = 5

    run_llm = st.checkbox("Include LLM Deep Analysis", value=True)

    if run_llm and not os.getenv("OPENROUTER_API_KEY"):
        st.warning("⚠️ OPENROUTER_API_KEY not set")

    st.divider()

    # Run button
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.session_state.running = True

# Main content area
if st.session_state.running:
    with st.spinner("Running analysis..."):
        try:
            snapshot = run_analysis(client_config, days_back, post_limit, run_llm, detailed_search, max_keywords)
            st.session_state.snapshot = snapshot
            st.session_state.running = False
            st.rerun()
        except Exception as e:
            import traceback
            error_msg = str(e)
            st.error(f"Analysis failed: {error_msg}")

            # Show detailed error for debugging
            with st.expander("🔍 Error Details (click to expand)", expanded=True):
                st.code(traceback.format_exc())

                # Check for common Reddit authentication issues
                if "401" in error_msg or "authentication" in error_msg.lower():
                    st.warning("""
                    **Reddit Authentication Error - Common Causes:**

                    1. **Wrong app type**: Your Reddit app at reddit.com/prefs/apps must be type "script" (not "web app" or "installed app")

                    2. **2FA enabled**: If you have two-factor authentication on your Reddit account, you need to use an app password or disable 2FA

                    3. **Wrong credentials**: Double-check username/password in Streamlit secrets

                    4. **IP blocked**: Some cloud providers are blocked by Reddit. The username/password auth helps bypass this.

                    **To fix:**
                    - Go to reddit.com/prefs/apps
                    - Delete the current app and create a new one with type "script"
                    - Update REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in Streamlit secrets
                    """)

            st.session_state.running = False

elif st.session_state.snapshot:
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Analysis Results",
        "📄 Executive Report",
        "🚨 Response Queue",
        "💡 Opportunities",
        "🎯 Recommendations",
        "📈 KPI Dashboard",
        "☁️ Word Cloud",
        "📉 Sentiment Shifts"
    ])

    with tab1:
        display_results(st.session_state.snapshot)

    with tab2:
        display_executive_report(st.session_state.snapshot, client_config)

    with tab3:
        display_response_queue()

    with tab4:
        display_opportunities()

    with tab5:
        display_recommendations(st.session_state.snapshot)

    with tab6:
        display_kpi_dashboard()

    with tab7:
        display_word_cloud()

    with tab8:
        display_shift_tracking()

    # Save to BigQuery option
    with st.sidebar:
        st.divider()
        if st.button("💾 Save to BigQuery"):
            project = os.getenv("REDDIT_GCP_PROJECT")
            if not project:
                st.error("REDDIT_GCP_PROJECT not set")
            else:
                try:
                    from reddit_sentiment.analytics.bq_store import BigQueryStore

                    store = BigQueryStore(project)
                    store.save_competitor_snapshot(st.session_state.snapshot)
                    st.success(f"Saved to BigQuery!")
                except Exception as e:
                    st.error(f"Failed to save: {e}")

else:
    # Welcome screen with feature cards
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h2 style="color: #1e293b; font-weight: 500;">Welcome to Reddit Competitive Intelligence</h2>
        <p style="color: #64748b; font-size: 1.1rem;">Configure your analysis in the sidebar and click <strong style="color: #6366f1;">Run Analysis</strong> to start.</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🔍</div>
            <h3>Sentiment Analysis</h3>
            <p>HuggingFace-powered sentiment analysis</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🏢</div>
            <h3>Competitor Tracking</h3>
            <p>Monitor mentions and comparisons</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🤖</div>
            <h3>LLM Deep Analysis</h3>
            <p>AI-powered insights via OpenRouter</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Show environment status
    with st.expander("🔧 Environment Status", expanded=False):
        env_vars = {
            "REDDIT_CLIENT_ID": bool(os.getenv("REDDIT_CLIENT_ID")),
            "REDDIT_CLIENT_SECRET": bool(os.getenv("REDDIT_CLIENT_SECRET")),
            "REDDIT_USER_AGENT": bool(os.getenv("REDDIT_USER_AGENT")),
            "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY")),
            "REDDIT_GCP_PROJECT": bool(os.getenv("REDDIT_GCP_PROJECT")),
        }

        col1, col2 = st.columns(2)
        items = list(env_vars.items())
        for i, (var, is_set) in enumerate(items):
            with col1 if i % 2 == 0 else col2:
                if is_set:
                    st.success(f"✅ {var}")
                else:
                    st.warning(f"⚠️ {var} not set")

        # Debug: show masked credential values
        st.markdown("---")
        st.markdown("**Debug - Credential Preview (masked):**")
        for var in ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT",
                    "REDDIT_USERNAME", "REDDIT_PASSWORD"]:
            val = os.getenv(var, "")
            if val:
                masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "****"
                st.code(f"{var}: {masked} (len={len(val)})")
            else:
                st.code(f"{var}: NOT SET")
