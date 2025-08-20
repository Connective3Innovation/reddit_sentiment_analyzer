# Reddit Sentiment Analysis Pipeline 🦝📈

End‑to‑end, production‑ready toolkit that:

1. Authenticates to Reddit via OAuth2 (PRAW).
2. Searches posts & expands comment trees for arbitrary keywords.
3. Cleans and dedups the text.
4. Runs sentiment scoring using either VADER (fast) or a HuggingFace model (accurate).
5. Saves results to Parquet **and/or** streams to a data warehouse.
6. Exposes a `Typer` CLI plus a pluggable ETL class.

```bash
# Install
pip install -r requirements.txt
# Quick run (VADER)
export REDDIT_CLIENT_ID=…
export REDDIT_CLIENT_SECRET=…
export REDDIT_USER_AGENT="sentiment‑demo (by u/yourname)"

reddit-sentiment run "openai" --days 2 --engine vader
```
```
reddit
├─ .pytest_cache
│  ├─ CACHEDIR.TAG
│  ├─ README.md
│  └─ v
│     └─ cache
│        ├─ lastfailed
│        └─ nodeids
├─ analysis.ipynb
├─ config
├─ pyproject.toml
├─ README.md
├─ reddit_sentiment.egg-info
│  ├─ dependency_links.txt
│  ├─ entry_points.txt
│  ├─ PKG-INFO
│  ├─ SOURCES.txt
│  └─ top_level.txt
├─ requirements-dev.txt
├─ requirements.txt
├─ sentiment_summary.csv
├─ setup.cfg
├─ src
│  ├─ reddit_sentiment
│  │  ├─ api
│  │  │  ├─ reddit_client.py
│  │  │  └─ __init__.py
│  │  ├─ auth
│  │  │  ├─ reddit_auth.py
│  │  │  └─ __init__.py
│  │  ├─ cache.py
│  │  ├─ cli.py
│  │  ├─ config
│  │  │  ├─ settings.py
│  │  │  └─ __init__.py
│  │  ├─ main.py
│  │  ├─ pipeline
│  │  │  ├─ etl.py
│  │  │  └─ __init__.py
│  │  ├─ sentiment
│  │  │  ├─ hf_engine.py
│  │  │  ├─ vader_engine.py
│  │  │  └─ __init__.py
│  │  └─ __init__.py
│  └─ reddit_sentiment.egg-info
│     ├─ dependency_links.txt
│     ├─ entry_points.txt
│     ├─ PKG-INFO
│     ├─ SOURCES.txt
│     └─ top_level.txt
├─ template.py
└─ tests
   ├─ test_api.py
   ├─ test_pipeline.py
   ├─ test_sentiment.py
   └─ __init__.py

```