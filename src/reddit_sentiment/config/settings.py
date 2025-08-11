"""App configuration powered by *pydantic‑settings* (built‑in from pydantic v2)."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ────────────────────── Reddit
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str = "reddit‑sentiment‑etl (by u/yourname)"

    # ────────────────────── Search params
    max_posts: int = 500
    days_back: int = 7
    cache_backend: str = "local"          # "local" or "gcs"
    cache_max_age_hours: int = 24         # 0 = always accept cache if present
    gcs_cache_prefix: str | None = None   # e.g. "gs://my-bucket/reddit-cache"
    gcp_project: str | None = None 
    # ────────────────────── Sentiment
    sentiment_engine: str = "vader"  # or "hf"

    # ────────────────────── Output
    output_dir: Path = Path("data")

    # pydantic meta
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_prefix="REDDIT_", env_file=".env", case_sensitive=False
    )


_SETTINGS_LOGGER: Final = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a singleton validated settings object."""
    config = Settings()  # reads env / .env
    _SETTINGS_LOGGER.info("Loaded settings: %s", config.model_dump(exclude={"reddit_client_secret"}))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return config