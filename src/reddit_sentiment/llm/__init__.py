# src/reddit_sentiment/llm/__init__.py
"""
LLM integration module for deep content analysis.

Uses OpenRouter to access various LLM providers for:
- Theme extraction
- Content opportunity identification
- Insight generation
- Unified competitive intelligence analysis (recommended)

Usage:
    from reddit_sentiment.llm import OpenRouterClient

    with OpenRouterClient() as client:
        # Recommended: single API call with stratified sampling
        result = client.run_unified_analysis(
            primary_brand="Capital One",
            industry="financial_services",
            period_days=7,
            comments_df=comments_df,
        )
"""

from .openrouter_client import OpenRouterClient
from .prompts import PROMPTS
from .prompts_v2 import UNIFIED_ANALYSIS_PROMPT, build_stratified_samples
from .cache import get_cached_result, save_to_cache, clear_cache
from .comment_clustering import CommentClusterer, is_clustering_available

__all__ = [
    "OpenRouterClient",
    "PROMPTS",
    "UNIFIED_ANALYSIS_PROMPT",
    "build_stratified_samples",
    "get_cached_result",
    "save_to_cache",
    "clear_cache",
    "CommentClusterer",
    "is_clustering_available",
]
