#!/usr/bin/env python
"""
Multi-Client Competitive Intelligence Analysis.

Run monthly competitive intelligence analysis for any configured client.
Data is stored in BigQuery for historical comparison and multi-tenant dashboards.

Usage:
    python run_analysis.py --client capital_one           # Run for specific client
    python run_analysis.py --client capital_one --save    # Run and save to BigQuery
    python run_analysis.py --client capital_one --llm     # Include LLM deep analysis
    python run_analysis.py --list                         # List available clients
    python run_analysis.py --client capital_one --days 30 --posts 500

Environment Variables:
    REDDIT_GCP_PROJECT     - GCP project for BigQuery storage
    REDDIT_CLIENTS_CONFIG  - Path to JSON config file for additional clients
    OPENROUTER_API_KEY     - Required for --llm deep analysis
"""

import argparse
import os
import sys
from datetime import datetime, timezone

from reddit_sentiment.data.collector import collect
from reddit_sentiment.data.preprocess import apply_cleaning
from reddit_sentiment.sentiment.hf_engine import HfEngine
from reddit_sentiment.analytics.competitor_analyzer import CompetitorAnalyzer
from reddit_sentiment.config.clients import get_registry, get_client, ClientConfig


def list_clients():
    """List all configured clients."""
    registry = get_registry()
    clients = registry.list_clients(enabled_only=False)

    print('=' * 70)
    print('CONFIGURED CLIENTS')
    print('=' * 70)

    if not clients:
        print('  No clients configured.')
        print('  Add clients via REDDIT_CLIENTS_CONFIG JSON file or programmatically.')
        return

    for client in clients:
        status = 'ENABLED' if client.enabled else 'DISABLED'
        print(f'''
  {client.client_id}
  ────────────────────────────────────────
    Name: {client.client_name}
    Brand: {client.primary_brand}
    Industry: {client.industry}
    Keywords: {", ".join(client.search_keywords)}
    Competitors: {len(client.competitors)} configured
    Status: {status}
''')


def run_analysis(
    client_config: ClientConfig,
    days_back: int = None,
    post_limit: int = None,
    save_to_bq: bool = False,
    run_llm: bool = False,
):
    """Run competitive intelligence analysis for a client."""

    # Use client defaults if not overridden
    days = days_back or client_config.days_back
    posts = post_limit or client_config.post_limit

    # Determine step count based on LLM flag
    total_steps = 5 if run_llm else 4

    print('=' * 70)
    print(f'{client_config.client_name.upper()} COMPETITIVE INTELLIGENCE REPORT')
    print(f'Client ID: {client_config.client_id}')
    print(f'Industry: {client_config.industry}')
    print(f'Analysis Period: Last {days} days')
    if run_llm:
        print('LLM Deep Analysis: ENABLED')
    print(f'Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}')
    print('=' * 70)

    # Collect data using client's search keywords
    search_query = ' OR '.join(client_config.search_keywords)
    print(f'\n[1/{total_steps}] Collecting Reddit data for "{search_query}"...')
    print(f'      Fetching up to {posts} posts from last {days} days...')
    posts_df, comments_df = collect(search_query, limit=posts, days_back=days)
    print(f'      Found {len(posts_df)} posts and {len(comments_df)} comments')

    if len(comments_df) == 0:
        print('No comments found!')
        return None

    # Clean and score
    print(f'\n[2/{total_steps}] Applying text cleaning...')
    comments_df = apply_cleaning(comments_df)

    print(f'\n[3/{total_steps}] Running sentiment analysis (HuggingFace 3-class)...')
    engine = HfEngine()
    sentiment_df = engine.run(comments_df['body'].tolist())

    # Convert 3-class sentiment (POSITIVE/NEUTRAL/NEGATIVE) to numeric score
    def convert_to_score(row):
        sentiment = row['sentiment']
        confidence = float(row['prob']) if row['prob'] is not None else 0.5
        confidence = max(0.0, min(1.0, confidence))

        if sentiment == 'POSITIVE':
            return confidence
        elif sentiment == 'NEGATIVE':
            return -confidence
        else:  # NEUTRAL
            return 0.0

    comments_df['sentiment_score'] = sentiment_df.apply(convert_to_score, axis=1)
    # Use model's 3-class labels directly
    comments_df['sentiment_label'] = sentiment_df['sentiment'].str.lower()

    # Competitive analysis using client config
    print(f'\n[4/{total_steps}] Analyzing competitor mentions...')
    analyzer = CompetitorAnalyzer(client_config=client_config)
    snapshot = analyzer.analyze_competitors(
        comments_df,
        period_days=days,
        posts_analyzed=len(posts_df),
    )

    # LLM deep analysis (optional)
    if run_llm:
        print(f'\n[5/{total_steps}] Running LLM deep analysis...')
        snapshot = _run_llm_analysis(snapshot, comments_df, client_config)

    # Display results
    _print_report(snapshot, client_config, show_llm=run_llm)

    # Save to BigQuery if requested
    if save_to_bq:
        _save_to_bigquery(snapshot)

    return snapshot


def _run_llm_analysis(snapshot, comments_df, client_config: ClientConfig):
    """Run LLM deep analysis using unified analysis (single API call)."""
    try:
        from reddit_sentiment.llm.openrouter_client import OpenRouterClient
    except ImportError as e:
        print(f'      [!] LLM module not available: {e}')
        return snapshot

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print('      [!] OPENROUTER_API_KEY not set, skipping LLM analysis')
        return snapshot

    # Try to use cache
    try:
        from reddit_sentiment.llm.cache import get_cached_result, save_to_cache
        cache_available = True
    except ImportError:
        cache_available = False

    result = None

    # Check cache first
    if cache_available:
        cached = get_cached_result(
            primary_brand=client_config.primary_brand,
            industry=client_config.industry,
            period_days=snapshot.period_days,
            comment_count=len(comments_df),
            max_age_hours=24,
        )
        if cached:
            result = cached
            print('      Using cached LLM analysis (< 24h old)')

    # Run unified analysis if no cache hit
    if not result:
        try:
            print('      Calling OpenRouter API (unified analysis)...')
            with OpenRouterClient(api_key=api_key) as client:
                result = client.run_unified_analysis(
                    primary_brand=client_config.primary_brand,
                    industry=client_config.industry,
                    period_days=snapshot.period_days,
                    comments_df=comments_df,
                )

            # Save to cache
            if cache_available and result and result.get('executive_summary'):
                save_to_cache(
                    result=result,
                    primary_brand=client_config.primary_brand,
                    industry=client_config.industry,
                    period_days=snapshot.period_days,
                    comment_count=len(comments_df),
                )
                print('      Cached LLM result for future use')

        except Exception as e:
            print(f'      [!] LLM analysis failed: {e}')
            return snapshot

    # Update snapshot with LLM insights
    if result:
        snapshot.llm_executive_summary = result.get('executive_summary')
        snapshot.llm_themes = result.get('key_themes', result.get('themes', []))
        snapshot.llm_unanswered_questions = result.get('unanswered_questions', [])
        snapshot.llm_competitive_insights = result.get('competitors_mentioned', result.get('competitive_insights', []))
        snapshot.llm_recommendations = result.get('actionable_recommendations', [])
        snapshot.llm_risk_signals = result.get('risk_signals', [])

        # Additional fields from unified analysis
        if result.get('brand_perception'):
            snapshot.llm_brand_perception = result.get('brand_perception')
        if result.get('competitive_position'):
            snapshot.llm_competitive_position = result.get('competitive_position')

        print('      LLM analysis complete!')

    return snapshot


def _print_report(snapshot, client_config: ClientConfig, show_llm: bool = False):
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
    print(f'Tracking {len(client_config.competitors)} competitors in {client_config.industry}')
    print('=' * 70)

    if snapshot.competitors:
        for comp in snapshot.competitors[:7]:  # Top 7 competitors
            print(f'''
  {comp.competitor.upper()}
  ----------------------------------------
    Mentions: {comp.mention_count}
    Avg Sentiment: {comp.avg_sentiment:.3f}
    Positive/Neutral/Negative: {comp.positive_mentions}/{comp.neutral_mentions}/{comp.negative_mentions}

    Switching Behavior:
      -> Users switching TO {comp.competitor}: {comp.switch_to_count}
      <- Users switching FROM {comp.competitor}: {comp.switch_from_count}''')

            if comp.better_at:
                print(f'    + Praised for: {", ".join(comp.better_at)}')
            if comp.worse_at:
                print(f'    - Criticized for: {", ".join(comp.worse_at)}')

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
            print(f'  {i}. [!] {threat}')
    else:
        print('  No significant threats identified.')

    print('\n' + '=' * 70)
    print('OPPORTUNITIES')
    print('=' * 70)
    if snapshot.opportunities:
        for i, opp in enumerate(snapshot.opportunities, 1):
            print(f'  {i}. [*] {opp}')
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

    # LLM Deep Analysis Section
    if show_llm and snapshot.llm_executive_summary:
        print('\n' + '=' * 70)
        print('LLM DEEP ANALYSIS')
        print('=' * 70)

        print('\n  EXECUTIVE SUMMARY:')
        print(f'  {snapshot.llm_executive_summary}')

        if snapshot.llm_themes:
            print('\n  DISCUSSION THEMES:')
            for i, theme in enumerate(snapshot.llm_themes, 1):
                name = theme.get('name', 'Unknown')
                desc = theme.get('description', '')
                sentiment = theme.get('sentiment', 'mixed')
                print(f'    {i}. {name} ({sentiment})')
                if desc:
                    print(f'       {desc}')

        if snapshot.llm_unanswered_questions:
            print('\n  UNANSWERED QUESTIONS:')
            for i, q in enumerate(snapshot.llm_unanswered_questions, 1):
                question = q.get('question', '')
                frequency = q.get('frequency', 'medium')
                print(f'    {i}. [{frequency.upper()}] {question}')
                if q.get('opportunity'):
                    print(f'       -> Opportunity: {q["opportunity"]}')

        if snapshot.llm_competitive_insights:
            print('\n  COMPETITIVE INSIGHTS:')
            for insight in snapshot.llm_competitive_insights:
                comp = insight.get('competitor', 'Unknown')
                perception = insight.get('perception', '')
                print(f'    [{comp.upper()}]')
                print(f'      Perception: {perception}')
                if insight.get('strengths_mentioned'):
                    print(f'      Strengths: {", ".join(insight["strengths_mentioned"])}')
                if insight.get('weaknesses_mentioned'):
                    print(f'      Weaknesses: {", ".join(insight["weaknesses_mentioned"])}')

        if snapshot.llm_recommendations:
            print('\n  ACTIONABLE RECOMMENDATIONS:')
            for i, rec in enumerate(snapshot.llm_recommendations, 1):
                priority = rec.get('priority', 'medium').upper()
                action = rec.get('action', '')
                rationale = rec.get('rationale', '')
                print(f'    {i}. [{priority}] {action}')
                if rationale:
                    print(f'       Why: {rationale}')

        if snapshot.llm_risk_signals:
            print('\n  RISK SIGNALS:')
            for i, risk in enumerate(snapshot.llm_risk_signals, 1):
                signal = risk.get('signal', '')
                severity = risk.get('severity', 'medium').upper()
                response = risk.get('suggested_response', '')
                print(f'    {i}. [{severity}] {signal}')
                if response:
                    print(f'       Response: {response}')

    print('\n' + '=' * 70)
    print(f'Report ID: {snapshot.snapshot_id}')
    print(f'Client ID: {snapshot.client_id or "N/A"}')
    print(f'Generated: {snapshot.measured_at.strftime("%Y-%m-%d %H:%M UTC")}')
    if show_llm and snapshot.llm_executive_summary:
        print('LLM Analysis: Included')
    print('=' * 70)


def _save_to_bigquery(snapshot):
    """Save snapshot to BigQuery."""
    project = os.getenv('REDDIT_GCP_PROJECT')
    if not project:
        print('\n[!] Cannot save: REDDIT_GCP_PROJECT not set')
        return

    try:
        from reddit_sentiment.analytics.bq_store import BigQueryStore
        store = BigQueryStore(project)
        store.save_competitor_snapshot(snapshot)
        print(f'\n[+] Saved to BigQuery: {project}.reddit_sentiment.competitor_snapshots')
        print(f'    Client ID: {snapshot.client_id}')
        print(f'    Snapshot ID: {snapshot.snapshot_id}')
        if snapshot.llm_executive_summary:
            print('    LLM Analysis: Included')
    except Exception as e:
        print(f'\n[-] Failed to save to BigQuery: {e}')


def main():
    parser = argparse.ArgumentParser(
        description='Run competitive intelligence analysis for configured clients',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python run_analysis.py --list                          # List clients
    python run_analysis.py --client capital_one            # Run analysis
    python run_analysis.py --client capital_one --save     # Save to BigQuery
    python run_analysis.py --client capital_one --llm      # Include LLM deep analysis
    python run_analysis.py --client capital_one -d 7 -p 100  # Quick test

Environment:
    REDDIT_GCP_PROJECT      GCP project for BigQuery
    REDDIT_CLIENTS_CONFIG   Path to clients JSON config
    OPENROUTER_API_KEY      Required for --llm analysis
'''
    )

    parser.add_argument(
        '--client', '-c',
        type=str,
        help='Client ID to run analysis for'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all configured clients'
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        help='Number of days to analyze (overrides client default)'
    )
    parser.add_argument(
        '--posts', '-p',
        type=int,
        help='Maximum posts to fetch (overrides client default)'
    )
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save results to BigQuery'
    )
    parser.add_argument(
        '--llm',
        action='store_true',
        help='Run LLM deep analysis (requires OPENROUTER_API_KEY)'
    )

    args = parser.parse_args()

    # List clients if requested
    if args.list:
        list_clients()
        return

    # Require client ID for analysis
    if not args.client:
        print('Error: --client is required for analysis')
        print('Use --list to see available clients')
        parser.print_help()
        sys.exit(1)

    # Get client config
    client_config = get_client(args.client)
    if not client_config:
        print(f'Error: Client "{args.client}" not found')
        print('Use --list to see available clients')
        sys.exit(1)

    if not client_config.enabled:
        print(f'Warning: Client "{args.client}" is disabled')
        confirm = input('Continue anyway? [y/N]: ')
        if confirm.lower() != 'y':
            sys.exit(0)

    # Run analysis
    run_analysis(
        client_config=client_config,
        days_back=args.days,
        post_limit=args.posts,
        save_to_bq=args.save,
        run_llm=args.llm,
    )


if __name__ == '__main__':
    main()
