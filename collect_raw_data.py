#!/usr/bin/env python
"""
Collect raw Reddit data for any configured client.
Exports posts and comments to CSV files.

IMPROVEMENTS (v2):
- Targets specific finance subreddits instead of all Reddit
- Filters posts that don't actually mention the brand
- Much higher relevance rate (expected 50%+ vs previous 0.8%)

Usage:
    python collect_raw_data.py --client capital_one_consumer
    python collect_raw_data.py --client capital_one_b2b
    python collect_raw_data.py --client capital_one_tech
    python collect_raw_data.py --list  # List available clients
"""

import argparse
import os
import re
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from tqdm import tqdm

from reddit_sentiment.api.reddit_client import RedditClient
from reddit_sentiment.config.clients import get_client, get_registry

# Configuration
DAYS_BACK = 30  # Reduced from 90 to avoid rate limiting
POSTS_PER_KEYWORD = 5000  # Match UI settings for consistent volumes
OUTPUT_DIR = Path("raw_data_export")


def get_brand_terms(client_config) -> list[str]:
    """Extract brand terms from client config for filtering."""
    brand = client_config.primary_brand.lower()
    # Start with primary brand
    terms = [brand]
    # Add common variations
    terms.append(brand.replace(" ", ""))  # "capital one" -> "capitalone"
    # Add short form if brand has multiple words
    words = brand.split()
    if len(words) > 1:
        terms.append(f"{words[0][:3]} {words[-1]}")  # "capital one" -> "cap one"
    return list(set(terms))


def contains_brand(text: str, brand_terms: list[str]) -> bool:
    """Check if text contains any brand term."""
    if not text:
        return False
    text_lower = text.lower()
    for term in brand_terms:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            return True
    return False


def list_clients():
    """List all available clients."""
    registry = get_registry()
    clients = registry.list_clients(enabled_only=False)
    print("Available clients:")
    print("-" * 60)
    for c in clients:
        status = "enabled" if c.enabled else "disabled"
        print(f"  {c.client_id:<25} {c.client_name} ({status})")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Collect raw Reddit data for a client")
    parser.add_argument("--client", "-c", required=False, help="Client ID to collect data for")
    parser.add_argument("--list", "-l", action="store_true", help="List available clients")
    args = parser.parse_args()

    if args.list:
        list_clients()
        return

    if not args.client:
        print("ERROR: --client is required. Use --list to see available clients.")
        list_clients()
        return

    # Get client config
    client_config = get_client(args.client)
    if not client_config:
        print(f"ERROR: Client '{args.client}' not found")
        list_clients()
        return

    keywords = client_config.search_keywords
    subreddits = client_config.target_subreddits
    brand_terms = get_brand_terms(client_config)

    print("=" * 60)
    print(f"REDDIT DATA COLLECTION: {client_config.client_name}")
    print("=" * 60)
    print(f"Client ID: {client_config.client_id}")
    print(f"Keywords: {len(keywords)}")
    print(f"Days back: {DAYS_BACK}")
    print(f"Posts per keyword: {POSTS_PER_KEYWORD}")
    print(f"Target subreddits: {', '.join(subreddits) if subreddits else 'ALL (not recommended)'}")
    print(f"Brand filter: {brand_terms}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Initialize Reddit client
    reddit = RedditClient()

    all_posts = []
    all_comments = []
    stats = {"total_fetched": 0, "after_brand_filter": 0}

    # Collect data for each keyword
    for i, keyword in enumerate(keywords, 1):
        print(f"\n[{i}/{len(keywords)}] Searching: {keyword}")

        try:
            # Search posts in targeted subreddits
            posts = reddit.search_posts(
                keyword,
                limit=POSTS_PER_KEYWORD,
                days_back=DAYS_BACK,
                subreddits=subreddits,  # NEW: target specific subreddits
            )
            stats["total_fetched"] += len(posts)
            print(f"  Fetched {len(posts)} posts from Reddit")

            if not posts:
                continue

            # Filter posts that don't actually contain the brand
            relevant_posts = [
                p for p in posts
                if contains_brand(f"{p.title} {getattr(p, 'selftext', '')}", brand_terms)
            ]
            filtered_out = len(posts) - len(relevant_posts)
            stats["after_brand_filter"] += len(relevant_posts)

            if filtered_out > 0:
                print(f"  Filtered {filtered_out} posts (no brand mention)")
            print(f"  Relevant posts: {len(relevant_posts)}")

            if not relevant_posts:
                continue

            # Fetch comments for relevant posts only
            comments_map = {}
            for post in tqdm(relevant_posts, desc="  Fetching comments", leave=False):
                comments_map[post.id] = reddit.fetch_comments(post, more_limit=5, per_post_limit=500)

            # Convert to DataFrames
            posts_df, comments_df = reddit.to_dataframe(relevant_posts, comments_map)

            # Add keyword column for tracking
            posts_df["search_keyword"] = keyword
            comments_df["search_keyword"] = keyword

            all_posts.append(posts_df)
            all_comments.append(comments_df)

            print(f"  Collected {len(posts_df)} posts, {len(comments_df)} comments")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Combine all data
    print("\n" + "=" * 60)
    print("COMBINING DATA...")

    if all_posts:
        posts_combined = pd.concat(all_posts, ignore_index=True)
        # Remove duplicates (same post might appear for multiple keywords)
        posts_combined = posts_combined.drop_duplicates(subset=["id"])
        print(f"Total unique posts: {len(posts_combined)}")
    else:
        posts_combined = pd.DataFrame()
        print("No posts collected")

    if all_comments:
        comments_combined = pd.concat(all_comments, ignore_index=True)
        # Remove duplicates
        comments_combined = comments_combined.drop_duplicates(subset=["comment_id"])
        print(f"Total unique comments: {len(comments_combined)}")
    else:
        comments_combined = pd.DataFrame()
        print("No comments collected")

    # Export to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    client_id = client_config.client_id

    posts_file = OUTPUT_DIR / f"{client_id}_posts_{timestamp}.csv"
    comments_file = OUTPUT_DIR / f"{client_id}_comments_{timestamp}.csv"

    if not posts_combined.empty:
        posts_combined.to_csv(posts_file, index=False, encoding="utf-8-sig")
        print(f"\nPosts saved to: {posts_file}")

    if not comments_combined.empty:
        comments_combined.to_csv(comments_file, index=False, encoding="utf-8-sig")
        print(f"Comments saved to: {comments_file}")

    # Print summary with relevance stats
    print("\n" + "=" * 60)
    print(f"SUMMARY: {client_config.client_name}")
    print("=" * 60)
    print(f"Client ID: {client_id}")
    print(f"Keywords searched: {len(keywords)}")
    print(f"Date range: Last {DAYS_BACK} days")
    print(f"Subreddits targeted: {len(subreddits)}")
    print()
    print("DATA QUALITY:")
    print(f"  Posts fetched from Reddit: {stats['total_fetched']}")
    print(f"  Posts after brand filter: {stats['after_brand_filter']}")
    if stats['total_fetched'] > 0:
        relevance_pct = (stats['after_brand_filter'] / stats['total_fetched']) * 100
        print(f"  Relevance rate: {relevance_pct:.1f}%")
    print()
    print(f"FINAL OUTPUT:")
    print(f"  Unique posts: {len(posts_combined)}")
    print(f"  Unique comments: {len(comments_combined)}")

    if not posts_combined.empty:
        print(f"\nTop subreddits by post count:")
        print(posts_combined["subreddit"].value_counts().head(10).to_string())

    print(f"\nFiles saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
