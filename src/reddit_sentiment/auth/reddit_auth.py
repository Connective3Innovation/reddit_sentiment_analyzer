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

# Module-level cache for the client
_cached_client = None


def clear_client_cache():
    """Clear the cached Reddit client (useful when credentials change)."""
    global _cached_client
    _cached_client = None
    get_client.cache_clear()


@lru_cache(maxsize=1)
def get_client() -> praw.Reddit:
    """Return a singleton OAuth2‑authenticated PRAW client.

    Credentials are loaded from environment variables (recommended) or a ``.env`` file.
    """
    # Load variables from .env *once* at import-time
    load_dotenv(override=False)

    missing = [v for v in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT") if v not in os.environ]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    # Log credential status for debugging
    _LOGGER.info("Reddit credentials status:")
    _LOGGER.info("  CLIENT_ID: %s (len=%d)", os.environ["REDDIT_CLIENT_ID"][:4] + "...", len(os.environ["REDDIT_CLIENT_ID"]))
    _LOGGER.info("  CLIENT_SECRET: %s (len=%d)", os.environ["REDDIT_CLIENT_SECRET"][:4] + "...", len(os.environ["REDDIT_CLIENT_SECRET"]))
    _LOGGER.info("  USER_AGENT: %s", os.environ["REDDIT_USER_AGENT"])

    username = os.environ.get("REDDIT_USERNAME", "")
    password = os.environ.get("REDDIT_PASSWORD", "")
    _LOGGER.info("  USERNAME: %s (len=%d)", username[:4] + "..." if username else "NOT SET", len(username))
    _LOGGER.info("  PASSWORD: %s (len=%d)", "****" if password else "NOT SET", len(password))

    # Use full authentication if username/password provided (needed for cloud environments)
    # Otherwise fall back to read-only mode
    reddit_kwargs = {
        "client_id": os.environ["REDDIT_CLIENT_ID"],
        "client_secret": os.environ["REDDIT_CLIENT_SECRET"],
        "user_agent": os.environ["REDDIT_USER_AGENT"],
        "ratelimit_seconds": 60,  # emergency sleep if throttled
    }

    # Add username/password if available (helps bypass cloud IP restrictions)
    if username and password:
        reddit_kwargs["username"] = username
        reddit_kwargs["password"] = password
        _LOGGER.info("Using authenticated mode with username/password")
    else:
        _LOGGER.info("Using read-only mode (no username/password)")

    try:
        client = praw.Reddit(**reddit_kwargs)

        # Test the authentication immediately by making a simple API call
        # This will catch 401 errors early instead of later during search
        _LOGGER.info("Testing Reddit authentication...")
        try:
            # This call will trigger authentication
            user = client.user.me()
            if user:
                _LOGGER.info("Authenticated as user: %s", user.name)
            else:
                _LOGGER.info("Authenticated in read-only mode (no user)")
        except Exception as auth_error:
            _LOGGER.error("Authentication test failed: %s", str(auth_error))
            # Re-raise with more context
            raise RuntimeError(f"Reddit authentication failed: {auth_error}") from auth_error

        if client.read_only is False:
            _LOGGER.warning("Client has mutation scopes — consider read‑only keys for production.")

        _LOGGER.info("Reddit client ready. Scopes: %s", client.auth.scopes())
        return client

    except Exception as e:
        _LOGGER.error("Failed to create Reddit client: %s", str(e))
        raise