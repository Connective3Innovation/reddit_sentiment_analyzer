#!/usr/bin/env python
"""
Monthly Competitive Intelligence Analysis for Capital One.

Run this monthly to track sentiment and competitor activity.
Data is stored in BigQuery for historical comparison.

Usage:
    python test_capital_one.py                    # Run analysis, print report
    python test_capital_one.py --save             # Run and save to BigQuery
    python test_capital_one.py --days 30          # Custom lookback period
    python test_capital_one.py --posts 500        # Custom post limit
"""

import argparse
import os
from datetime import datetime, timezone

from reddit_sentiment.data.collector import collect
from reddit_sentiment.data.preprocess import apply_cleaning
from reddit_sentiment.sentiment.vader_engine import VaderEngine
from reddit_sentiment.analytics.competitor_analyzer import CompetitorAnalyzer


def run_analysis(days_back: int = 30, post_limit: int = 500, save_to_bq: bool = False):
    """Run competitive intelligence analysis."""

    print('=' * 70)
    print('CAPITAL ONE COMPETITIVE INTELLIGENCE REPORT')
    print(f'Analysis Period: Last {days_back} days')
    print(f'Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}')
    print('=' * 70)

    # Collect data
    print(f'\n[1/4] Collecting Reddit data for "capital one"...')
    print(f'      Fetching up to {post_limit} posts from last {days_back} days...')
    posts_df, comments_df = collect('capital one', limit=post_limit, days_back=days_back)
    print(f'      Found {len(posts_df)} posts and {len(comments_df)} comments')

    if len(comments_df) == 0:
        print('No comments found!')
        return None

    # Clean and score
    print('\n[2/4] Applying text cleaning...')
    comments_df = apply_cleaning(comments_df)

    print('\n[3/4] Running sentiment analysis...')
    engine = VaderEngine()
    sentiment_df = engine.run(comments_df['body'].tolist())
    comments_df['sentiment_score'] = sentiment_df['compound']
    comments_df['sentiment_label'] = comments_df['sentiment_score'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
    )

    # Competitive analysis
    print('\n[4/4] Analyzing competitor mentions...')
    analyzer = CompetitorAnalyzer(primary_brand='capital one')
    snapshot = analyzer.analyze_competitors(
        comments_df,
        period_days=days_back,
        posts_analyzed=len(posts_df),
    )

    # Display results
    _print_report(snapshot)

    # Save to BigQuery if requested
    if save_to_bq:
        _save_to_bigquery(snapshot)

    return snapshot


def _print_report(snapshot):
    """Print formatted report."""

    print('\n' + '=' * 70)
    print('OVERALL SENTIMENT')
    print('=' * 70)
    print(f'''
  Total Comments Analyzed: {snapshot.comments_analyzed}
  Posts Analyzed: {snapshot.posts_analyzed}
  Mean Sentiment Score: {snapshot.primary_sentiment:.3f}

  Distribution:
    Positive: {snapshot.primary_positive_pct:.1f}%
    Neutral:  {snapshot.primary_neutral_pct:.1f}%
    Negative: {snapshot.primary_negative_pct:.1f}%
''')

    print('=' * 70)
    print(f'COMPETITOR ANALYSIS ({snapshot.total_competitor_mentions} total mentions)')
    print('=' * 70)

    if snapshot.competitors:
        for comp in snapshot.competitors[:7]:  # Top 7 competitors
            print(f'''
  {comp.competitor.upper()}
  ────────────────────────────────────────
    Mentions: {comp.mention_count}
    Avg Sentiment: {comp.avg_sentiment:.3f}
    Positive/Neutral/Negative: {comp.positive_mentions}/{comp.neutral_mentions}/{comp.negative_mentions}

    Switching Behavior:
      → Users switching TO {comp.competitor}: {comp.switch_to_count}
      ← Users switching FROM {comp.competitor}: {comp.switch_from_count}''')

            if comp.better_at:
                print(f'    ✓ Praised for: {", ".join(comp.better_at)}')
            if comp.worse_at:
                print(f'    ✗ Criticized for: {", ".join(comp.worse_at)}')

            if comp.sample_mentions:
                print(f'\n    Sample mention:')
                sample = comp.sample_mentions[0]
                context = sample[:200] + '...' if len(sample) > 200 else sample
                print(f'      "{context}"')
    else:
        print('  No competitor mentions found in the data.')

    print('\n' + '=' * 70)
    print('COMPETITIVE THREATS')
    print('=' * 70)
    if snapshot.threats:
        for i, threat in enumerate(snapshot.threats, 1):
            print(f'  {i}. ⚠️  {threat}')
    else:
        print('  No significant threats identified.')

    print('\n' + '=' * 70)
    print('OPPORTUNITIES')
    print('=' * 70)
    if snapshot.opportunities:
        for i, opp in enumerate(snapshot.opportunities, 1):
            print(f'  {i}. 💡 {opp}')
    else:
        print('  No immediate opportunities identified.')

    print('\n' + '=' * 70)
    print('TOP PAIN POINTS (Most Negative Comments)')
    print('=' * 70)
    if snapshot.top_pain_points:
        for i, pain in enumerate(snapshot.top_pain_points, 1):
            print(f'\n  {i}. {pain}')
    else:
        print('  No significant pain points found.')

    print('\n' + '=' * 70)
    print(f'Report ID: {snapshot.snapshot_id}')
    print(f'Generated: {snapshot.measured_at.strftime("%Y-%m-%d %H:%M UTC")}')
    print('=' * 70)


def _save_to_bigquery(snapshot):
    """Save snapshot to BigQuery."""
    project = os.getenv('REDDIT_GCP_PROJECT')
    if not project:
        print('\n⚠️  Cannot save: REDDIT_GCP_PROJECT not set')
        return

    try:
        from reddit_sentiment.analytics.bq_store import BigQueryStore
        store = BigQueryStore(project)
        store.save_competitor_snapshot(snapshot)
        print(f'\n✅ Saved to BigQuery: {project}.reddit_sentiment.competitor_snapshots')
    except Exception as e:
        print(f'\n❌ Failed to save to BigQuery: {e}')


def main():
    parser = argparse.ArgumentParser(
        description='Run monthly competitive intelligence analysis for Capital One'
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=30,
        help='Number of days to analyze (default: 30)'
    )
    parser.add_argument(
        '--posts', '-p',
        type=int,
        default=500,
        help='Maximum posts to fetch (default: 500)'
    )
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save results to BigQuery'
    )

    args = parser.parse_args()
    run_analysis(
        days_back=args.days,
        post_limit=args.posts,
        save_to_bq=args.save,
    )


if __name__ == '__main__':
    main()
