"""Module entry‑point so users can `python -m reddit_sentiment …`."""
from .cli import app  # pragma: no cover

if __name__ == "__main__":
    app()