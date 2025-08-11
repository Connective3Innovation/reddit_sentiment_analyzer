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