# src/reddit_sentiment/scheduler/runner.py
"""
Entry point for scheduled job execution.

Can be called via:
- Command line: python -m reddit_sentiment.scheduler.runner
- Cloud Scheduler HTTP trigger to /scheduler/run endpoint
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from .scheduler import get_scheduler, init_from_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_all_due() -> dict:
    """Run all due scheduled jobs."""
    scheduler = init_from_env()
    results = scheduler.run_due_jobs()

    summary = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "jobs_run": len(results),
        "successful": sum(1 for r in results if r.get("status") == "success"),
        "failed": sum(1 for r in results if r.get("status") == "error"),
        "results": results,
    }

    return summary


def run_keyword(keyword: str) -> dict:
    """Run aggregation for a specific keyword."""
    scheduler = get_scheduler()
    return scheduler.run_job(keyword)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run scheduled sentiment analysis jobs"
    )
    parser.add_argument(
        "--keyword", "-k",
        help="Run for a specific keyword instead of all due jobs",
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show scheduler status and exit",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List tracked keywords and exit",
    )
    parser.add_argument(
        "--add", "-a",
        help="Add a keyword to tracking",
    )
    parser.add_argument(
        "--remove", "-r",
        help="Remove a keyword from tracking",
    )
    parser.add_argument(
        "--schedule",
        default="0 6 * * *",
        help="Cron schedule when adding keyword (default: '0 6 * * *')",
    )

    args = parser.parse_args()
    scheduler = init_from_env()

    # Handle status command
    if args.status:
        status = scheduler.get_status()
        print(f"\nScheduler Status:")
        print(f"  Total tracked: {status['total_tracked']}")
        print(f"  Enabled: {status['enabled']}")
        print(f"  Due now: {status['due_now']}")
        return 0

    # Handle list command
    if args.list:
        keywords = scheduler.get_tracked_keywords(enabled_only=False)
        print(f"\nTracked Keywords ({len(keywords)}):")
        for kw in keywords:
            status = "enabled" if kw.enabled else "disabled"
            last = kw.last_run.strftime("%Y-%m-%d %H:%M") if kw.last_run else "never"
            next_run = kw.next_run.strftime("%Y-%m-%d %H:%M") if kw.next_run else "n/a"
            print(f"  {kw.keyword} [{status}]")
            print(f"    Schedule: {kw.schedule}")
            print(f"    Last run: {last}")
            print(f"    Next run: {next_run}")
        return 0

    # Handle add command
    if args.add:
        tracked = scheduler.add_keyword(args.add, schedule=args.schedule)
        print(f"Added '{args.add}' with schedule '{args.schedule}'")
        if tracked.next_run:
            print(f"Next run: {tracked.next_run.strftime('%Y-%m-%d %H:%M UTC')}")
        return 0

    # Handle remove command
    if args.remove:
        scheduler.remove_keyword(args.remove)
        print(f"Removed '{args.remove}' from tracking")
        return 0

    # Run specific keyword
    if args.keyword:
        logger.info(f"Running aggregation for '{args.keyword}'")
        try:
            result = run_keyword(args.keyword)
            if result.get("status") == "success":
                print(f"Success: {result.get('volume', 0)} comments processed")
                print(f"  Alerts: {len(result.get('alerts', []))}")
                print(f"  Opportunities: {len(result.get('opportunities', []))}")
            else:
                print(f"Warning: {result.get('status')}")
            return 0
        except Exception as e:
            logger.error(f"Failed: {e}")
            return 1

    # Run all due jobs
    logger.info("Running all due jobs")
    summary = run_all_due()

    print(f"\nExecution Summary:")
    print(f"  Jobs run: {summary['jobs_run']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")

    if summary['failed'] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
