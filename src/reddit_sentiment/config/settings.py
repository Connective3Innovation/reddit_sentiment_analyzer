from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal
from functools import lru_cache
from pathlib import Path
import logging

class Settings(BaseSettings):
    # Map directly to REDDIT_CLIENT_ID etc.
    reddit_client_id: str = Field(validation_alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(validation_alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(validation_alias="REDDIT_USER_AGENT")

    max_posts: int = Field(500, validation_alias="REDDIT_MAX_POSTS")
    days_back: int = Field(90, validation_alias="REDDIT_DAYS_BACK")
    sentiment_engine: Literal["hf", "vader"] = Field(
        "hf", validation_alias="REDDIT_SENTIMENT_ENGINE"
    )
    output_dir: str = Field("data", validation_alias="REDDIT_OUTPUT_DIR")

    cache_backend: Literal["local", "gcs"] = Field(
        "gcs", validation_alias="REDDIT_CACHE_BACKEND"
    )
    gcs_cache_prefix: str | None = Field(
        default=None, validation_alias="REDDIT_GCS_CACHE_PREFIX"
    )
    gcp_project: str | None = Field(
        default=None, validation_alias="REDDIT_GCP_PROJECT"
    )
    cache_max_age_hours: int = Field(
        0, validation_alias="REDDIT_CACHE_MAX_AGE_HOURS"
    )

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

_SETTINGS_LOGGER = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cfg = Settings()
    safe = cfg.model_dump()
    safe["reddit_client_secret"] = "***"
    _SETTINGS_LOGGER.info("Loaded settings: %s", safe)
    if not str(cfg.output_dir).startswith("gs://"):
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg
