"""
FastAPI service for live Reddit sentiment + plot-ready JSON datasets.

Run locally:
  uvicorn api:app --reload --port 8080
"""

from __future__ import annotations

import io
import os
import time
import typing as t
import datetime as dt
from dataclasses import asdict, dataclass
import inspect

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
    PlainTextResponse,
    RedirectResponse,
)
from starlette.middleware.gzip import GZipMiddleware

# Optional: faster JSON as default response class (endpoints still use JSONResponse for safety)
try:
    from fastapi.responses import ORJSONResponse  # type: ignore
    DEFAULT_RESPONSE_CLASS = ORJSONResponse
except Exception:
    DEFAULT_RESPONSE_CLASS = JSONResponse

# ─────────────────────────────────────────────────────────────────────────────
# Import project internals with resilient fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def _import_path_variants():
    """Import utilities, tolerating either `reddit_sentiment.*` or flat layout."""
    mod: dict[str, t.Any] = {}

    # ETL
    try:
        from reddit_sentiment.pipeline.etl import run_etl as _run_etl
        mod["run_etl"] = _run_etl
    except Exception:
        try:
            from etl import run_etl as _run_etl  # type: ignore
            mod["run_etl"] = _run_etl
        except Exception:
            mod["run_etl"] = None

    # Collector / preprocess
    try:
        from reddit_sentiment.data.collector import collect as _collect
    except Exception:
        try:
            from collector import collect as _collect  # type: ignore
        except Exception:
            _collect = None
    mod["collect"] = _collect

    try:
        from reddit_sentiment.data.preprocess import apply_cleaning as _clean
    except Exception:
        try:
            from preprocess import apply_cleaning as _clean  # type: ignore
        except Exception:
            _clean = None
    mod["clean"] = _clean

    # Engines
    try:
        from reddit_sentiment.sentiment.hf_engine import HfEngine as _Hf
    except Exception:
        try:
            from hf_engine import HfEngine as _Hf  # type: ignore
        except Exception:
            _Hf = None
    mod["HfEngine"] = _Hf

    try:
        from reddit_sentiment.sentiment.vader_engine import VaderEngine as _Vader
    except Exception:
        try:
            from vader_engine import VaderEngine as _Vader  # type: ignore
        except Exception:
            _Vader = None
    mod["VaderEngine"] = _Vader

    return mod

MODS = _import_path_variants()

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

ENGINE_CHOICES = ("hf", "vader")

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _to_utc_series(s: t.Any) -> pd.Series:
    """Convert a column/sequence to UTC pandas datetime (safe)."""
    return pd.to_datetime(s, utc=True, errors="coerce")

def _to_utc_iso(val: t.Any) -> t.Optional[str]:
    """Convert a single value to ISO UTC string."""
    try:
        ts = pd.to_datetime(val, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        # ts may be pandas.Timestamp
        return ts.isoformat()
    except Exception:
        return None

def _reddit_comment_url(post_id: str, comment_id: str) -> str:
    return f"https://www.reddit.com/comments/{post_id}/_/{comment_id}"

# ─────────────────────────────────────────────────────────────────────────────
# Report builders (plot-ready datasets)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Meta:
    keyword: str
    engine: str
    took_ms: int
    rows: int
    window_days: int
    generated_at: str

def _sentiment_label(row: pd.Series) -> str:
    # HF returns 'positive'/'neutral'/'negative'; VADER returns compound score
    if "sentiment" in row and isinstance(row.get("sentiment"), str) and row["sentiment"]:
        return row["sentiment"].lower()
    c = row.get("compound")
    if pd.notna(c):
        if c >= 0.05:
            return "positive"
        elif c <= -0.05:
            return "negative"
        else:
            return "neutral"
    return "neutral"

def build_report(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "timeseries_daily": [],
            "sentiment_share": [],
            "histogram": [],
            "subreddits": [],
            "authors": [],
            "top_positive": [],
            "top_negative": [],
        }

    df = df.copy()

    # Ensure created column
    if "created" in df.columns:
        df["created"] = _to_utc_series(df["created"])
    elif "created_utc" in df.columns:
        df["created"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    else:
        df["created"] = _now_utc()

    # Ensure text & prob
    if "body" in df.columns:
        df["body"] = df["body"].astype(str)
    if "prob" not in df.columns and "compound" in df.columns:
        df["prob"] = df["compound"].abs()

    df["label"] = df.apply(_sentiment_label, axis=1)

    # Timeseries (daily)
    by_day = (
        df.set_index("created").sort_index()
        .groupby(pd.Grouper(freq="D"))
        .agg(mean_sentiment=("prob", "mean"), count=("body", "count"))
        .reset_index()
    )
    ts_daily = [
        {
            "date": r["created"].date().isoformat(),
            "mean_sentiment": (float(r["mean_sentiment"]) if pd.notna(r["mean_sentiment"]) else None),
            "count": int(r["count"]),
        }
        for _, r in by_day.iterrows()
    ]

    # Sentiment share
    share = (
        df["label"].value_counts(dropna=False)
        .reindex(["positive", "neutral", "negative"], fill_value=0)
    )
    total = int(share.sum()) or 1
    sentiment_share = [
        {"label": k, "count": int(v), "pct": round(int(v) * 100 / total, 2)}
        for k, v in share.to_dict().items()
    ]

    # Histogram of probabilities (0..1)
    probs = df["prob"].fillna(0).clip(0, 1)
    bins = np.linspace(0, 1, 21)
    hist, edges = np.histogram(probs, bins=bins)
    histogram = [
        {"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(hist[i])}
        for i in range(len(hist))
    ]

    # Subreddit breakdown
    if "subreddit" in df.columns:
        sub = (
            df.groupby("subreddit")
            .agg(count=("body", "count"), mean_prob=("prob", "mean"))
            .sort_values(["count", "mean_prob"], ascending=[False, False])
            .head(25)
            .reset_index()
        )
        subreddits = [
            {
                "subreddit": r["subreddit"],
                "count": int(r["count"]),
                "mean_prob": float(r["mean_prob"]) if pd.notna(r["mean_prob"]) else None,
            }
            for _, r in sub.iterrows()
        ]
    else:
        subreddits = []

    # Author breakdown
    if "author" in df.columns:
        auth = (
            df.groupby("author")
            .agg(count=("body", "count"), mean_prob=("prob", "mean"))
            .sort_values(["count", "mean_prob"], ascending=[False, False])
            .head(25)
            .reset_index()
        )
        authors = [
            {
                "author": str(r["author"]),
                "count": int(r["count"]),
                "mean_prob": float(r["mean_prob"]) if pd.notna(r["mean_prob"]) else None,
            }
            for _, r in auth.iterrows()
        ]
    else:
        authors = []

    # Top comments
    def _row_to_comment(r: pd.Series) -> dict:
        post_id = str(r.get("post_id", "") or "")
        comment_id = str(r.get("comment_id", "") or "")
        return {
            "post_id": post_id,
            "comment_id": comment_id,
            "subreddit": r.get("subreddit"),
            "author": str(r.get("author")),
            "body": r.get("body"),
            "prob": float(r.get("prob")) if pd.notna(r.get("prob")) else None,
            "score": int(r.get("score")) if pd.notna(r.get("score")) else None,
            "created": _to_utc_iso(r.get("created")),
            "url": _reddit_comment_url(post_id, comment_id) if post_id and comment_id else None,
        }

    top_positive = [
        _row_to_comment(r)
        for _, r in df.sort_values("prob", ascending=False).head(10).iterrows()
    ]
    top_negative = [
        _row_to_comment(r)
        for _, r in df.sort_values("prob", ascending=True).head(10).iterrows()
    ]

    return {
        "timeseries_daily": ts_daily,
        "sentiment_share": sentiment_share,
        "histogram": histogram,
        "subreddits": subreddits,
        "authors": authors,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Core execute: run ETL then load DF
# ─────────────────────────────────────────────────────────────────────────────

def _run_live(keyword: str, days_back: int, limit: int, engine: str) -> tuple[pd.DataFrame, Meta]:
    start = time.time()

    # We do NOT call run_etl() here because it reads cached settings and ignores request params.
    if not (MODS["collect"] and MODS["clean"] and (MODS["HfEngine"] or MODS["VaderEngine"])):
        raise RuntimeError("Project internals not importable; please install your package.")

    collect_fn = MODS["collect"]
    # Always honor request-scoped overrides
    posts_df, comments_df = collect_fn(keyword, limit=limit, days_back=days_back)  # type: ignore[arg-type]
    comments_df = MODS["clean"](comments_df, column="body")  # type: ignore[call-arg]

    if engine == "hf":
        eng = MODS["HfEngine"]()  # type: ignore[operator]
        sent_df = eng.run(comments_df["body"].tolist())
    else:
        eng = MODS["VaderEngine"]()  # type: ignore[operator]
        sent_df = eng.run(comments_df["body"].tolist())
        if "prob" not in sent_df.columns and "compound" in sent_df.columns:
            sent_df["prob"] = sent_df["compound"].abs()

    df = pd.concat(
        [comments_df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1
    )

    # Normalize expected columns
    for col in ("post_id", "comment_id", "body"):
        if col not in df.columns:
            df[col] = None
    if "created" not in df.columns and "created_utc" in df.columns:
        df["created"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)

    took_ms = int((time.time() - start) * 1000)
    meta = Meta(
        keyword=keyword,
        engine=engine,
        took_ms=took_ms,
        rows=int(len(df)),
        window_days=int(days_back),
        generated_at=_now_utc().isoformat(),
    )
    return df, meta

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Reddit Sentiment API",
    version="1.0.0",
    default_response_class=DEFAULT_RESPONSE_CLASS,
)

# CORS first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # keep False with wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Then GZip once
app.add_middleware(GZipMiddleware, minimum_size=1024)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return JSONResponse(status_code=204, content=None)

@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"

@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Search keyword"),
    days: int = Query(90, ge=1, le=365 * 3),
    months: int | None = Query(None, ge=1, le=36, description="Alternative to days"),
    years: int | None = Query(None, ge=1, le=3, description="Alternative to days"),
    limit: int = Query(500, ge=1, le=2000),
    engine: str = Query("vader", pattern="^(hf|vader)$"),
    include_raw: bool = Query(False, description="Include raw rows (truncated)"),
):
    # compute window
    days_back = days
    if months:
        days_back = months * 30
    if years:
        days_back = years * 365

    df, meta = _run_live(q, days_back=days_back, limit=limit, engine=engine)

    report = build_report(df)

    payload: dict[str, t.Any] = {
        "meta": asdict(meta),
        "report": report,
    }

    if include_raw:
        # keep it bounded (e.g., first 1000 rows)
        sample = df.head(1000).copy()
        if "created" in sample.columns:
            sample["created"] = _to_utc_series(sample["created"]).astype(str)
        payload["raw_sample"] = sample.to_dict(orient="records")

    # Safe serialization (avoids orjson failures on pandas/numpy types)
    return JSONResponse(content=jsonable_encoder(payload))

@app.get("/download")
def download(
    q: str = Query(..., min_length=1),
    days: int = Query(90, ge=1),
    limit: int = Query(500, ge=5, le=2000),
    engine: str = Query("vader", pattern="^(hf|vader)$"),
    format: str = Query("csv", pattern="^(csv|parquet)$"),
):
    df, meta = _run_live(q, days_back=days, limit=limit, engine=engine)

    if format == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        headers = {"Content-Disposition": f"attachment; filename={q.replace(' ', '_')}.csv"}
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)
    else:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        headers = {"Content-Disposition": f"attachment; filename={q.replace(' ', '_')}.parquet"}
        return StreamingResponse(iter([buf.getvalue()]), media_type="application/octet-stream", headers=headers)

@app.get("/plots")
def plots(
    q: str = Query(..., min_length=1),
    days: int = Query(90, ge=1),
    limit: int = Query(500, ge=5, le=2000),
    engine: str = Query("vader", pattern="^(hf|vader)$"),
):
    df, meta = _run_live(q, days_back=days, limit=limit, engine=engine)
    return JSONResponse(content=jsonable_encoder(build_report(df)))

@app.get("/debug/env", include_in_schema=False)
def debug_env():
    keys = [
        "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT",
        "REDDIT_OUTPUT_DIR", "REDDIT_CACHE_BACKEND", "REDDIT_GCS_CACHE_PREFIX",
        "REDDIT_MORE_LIMIT", "REDDIT_COMMENTS_PER_POST", "REDDIT_SKIP_LARGE_POSTS",
    ]
    return {k: bool(os.getenv(k)) for k in keys}
