"""Central Reddit client factory.

This keeps PRAW config & rate‑limit info in one place so the rest of the app can remain stateless.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Final
from dotenv import load_dotenv 
import praw

_LOGGER: Final = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> praw.Reddit:
    """Return a singleton OAuth2‑authenticated PRAW client.

    Credentials are loaded from environment variables (recommended) or a ``.env`` file.
    """
       # Load variables from .env *once* at import-time
    load_dotenv(override=False)       # ← N
    missing = [v for v in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT") if v not in os.environ]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    # Use full authentication if username/password provided (needed for cloud environments)
    # Otherwise fall back to read-only mode
    reddit_kwargs = {
        "client_id": os.environ["REDDIT_CLIENT_ID"],
        "client_secret": os.environ["REDDIT_CLIENT_SECRET"],
        "user_agent": os.environ["REDDIT_USER_AGENT"],
        "ratelimit_seconds": 60,  # emergency sleep if throttled
    }

    # Add username/password if available (helps bypass cloud IP restrictions)
    if os.environ.get("REDDIT_USERNAME") and os.environ.get("REDDIT_PASSWORD"):
        reddit_kwargs["username"] = os.environ["REDDIT_USERNAME"]
        reddit_kwargs["password"] = os.environ["REDDIT_PASSWORD"]
        _LOGGER.info("Using authenticated mode with username/password")
    else:
        _LOGGER.info("Using read-only mode (no username/password)")

    client = praw.Reddit(**reddit_kwargs)

    if client.read_only is False:  # pragma: no cover
        _LOGGER.warning("Client has mutation scopes — consider read‑only keys for production.")

    _LOGGER.info("Authenticated to Reddit as %s", client.auth.scopes())
    return client