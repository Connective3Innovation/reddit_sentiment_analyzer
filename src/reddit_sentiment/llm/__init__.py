# src/reddit_sentiment/llm/__init__.py
"""
LLM integration module for deep content analysis.

Uses OpenRouter to access various LLM providers for:
- Theme extraction
- Content opportunity identification
- Insight generation
"""

from .openrouter_client import OpenRouterClient
from .prompts import PROMPTS

__all__ = ["OpenRouterClient", "PROMPTS"]
