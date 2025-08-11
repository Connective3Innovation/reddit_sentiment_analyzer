"""Reddit Sentiment — initialize package‑wide logging and re‑export public API."""
from importlib.metadata import version

__all__ = [
    "get_client",
    "collect",
    "analyze",
    "run_etl",
]

__version__: str = version("reddit-sentiment")  # pyproject.toml driven

from .auth.reddit_auth import get_client  # noqa: E402
from .data.collector import collect  # noqa: E402
from .sentiment import analyze  # noqa: E402
from .pipeline.etl import run_etl  # noqa: E402