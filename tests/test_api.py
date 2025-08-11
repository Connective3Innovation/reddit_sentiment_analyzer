# tests/test_api.py

"""Ensure RedditClient.search_posts builds the correct query without network calls."""

import os
import re
from types import SimpleNamespace

# ──────────────────────────────────────────────────────────────────────────────
# Provide dummy credentials so get_client() passes its env check
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("REDDIT_CLIENT_ID", "dummy")
os.environ.setdefault("REDDIT_CLIENT_SECRET", "dummy")
os.environ.setdefault("REDDIT_USER_AGENT", "pytest")

# ──────────────────────────────────────────────────────────────────────────────
# Stub the get_client inside the reddit_client module itself
# ──────────────────────────────────────────────────────────────────────────────
import reddit_sentiment.api.reddit_client as rc_mod  # noqa: E402

class DummyClient:
    def subreddit(self, *args, **kwargs):
        # This will be overridden in the test itself
        return SimpleNamespace(search=lambda *_, **__: [])

    auth = SimpleNamespace(scopes=lambda: [], limits={"remaining": 20, "reset_timestamp": 0})

rc_mod.get_client = lambda: DummyClient()

# ──────────────────────────────────────────────────────────────────────────────
# Now import the class under test
# ──────────────────────────────────────────────────────────────────────────────
from reddit_sentiment.api.reddit_client import RedditClient  # noqa: E402

def test_search_query(monkeypatch):
    rc = RedditClient()
    captured = {}

    def fake_search(*args, **kwargs):
        captured.update(kwargs)
        return []

    # **Key change**: override the client's subreddit() method so every call
    # returns an object whose .search() is our fake_search.
    monkeypatch.setattr(
        rc._client,
        "subreddit",
        lambda name: SimpleNamespace(search=fake_search)
    )

    # Run the method under test
    rc.search_posts("openai", limit=10, days_back=1)

    # Verify that our fake_search was called with the right kwargs
    assert "query" in captured, f"No query captured: {captured}"
    assert re.search(r"openai", captured["query"], re.IGNORECASE)
    assert captured["limit"] == 10
