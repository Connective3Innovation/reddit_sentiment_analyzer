# src/reddit_sentiment/cli.py

"""Typer-powered CLI wrapper for Reddit Sentiment ETL, with days|months|years options."""
from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from .pipeline.etl import run_etl

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# Engine enum
# ──────────────────────────────────────────────────────────────────────────────
class Engine(str, Enum):
    vader = "vader"
    hf = "hf"

# ──────────────────────────────────────────────────────────────────────────────
# CLI app
# ──────────────────────────────────────────────────────────────────────────────
app = typer.Typer(add_completion=False, help="Keyword-based Reddit sentiment ETL.")

@app.command()
def run(
    keyword: str = typer.Argument(
        ..., help="Search keyword, e.g. 'nvidia earnings'"
    ),
    days: int = typer.Option(
        7, "--days", "-d", help="Fallback look-back window in days"
    ),
    months: int = typer.Option(
        0, "--months", "-m", help="Look-back window in months (≈30 days each)"
    ),
    years: int = typer.Option(
        0, "--years", "-y", help="Look-back window in years (≈365 days each)"
    ),
    limit: int = typer.Option(
        500, "--limit", "-l", help="Maximum number of posts to fetch"
    ),
    engine: Engine = typer.Option(
        Engine.vader,
        "--engine",
        "-e",
        help="Sentiment engine: rule-based (vader) or transformer (hf)",
        case_sensitive=False,
        show_choices=True,
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Override the default output file path"
    ),
):
    """
    Run ETL for KEYWORD, looking back DAYS, MONTHS, or YEARS, scoring with ENGINE.
    """
    # Compute total look-back in days
    if years or months:
        total_days = years * 365 + months * 30
        logging.info("Using look-back of %d days (%d years + %d months)", total_days, years, months)
    else:
        total_days = days
        logging.info("Using look-back of %d days", total_days)

    # Override environment-based config
    os.environ.setdefault("REDDIT_DAYS_BACK", str(total_days))
    os.environ.setdefault("REDDIT_MAX_POSTS", str(limit))
    os.environ.setdefault("REDDIT_SENTIMENT_ENGINE", engine.value)

    # Execute ETL
    out_path = run_etl(keyword)

    # Handle custom output path
    if output:
        out_path.rename(output)
        typer.echo(f"Saved to {output}")
    else:
        typer.echo(f"Saved to {out_path}")

if __name__ == "__main__":
    app()
