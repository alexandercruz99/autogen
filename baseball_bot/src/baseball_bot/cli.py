from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

from baseball_bot.mlb_client import MLBClient
from baseball_bot.predict import analyze_slate
from baseball_bot.report import print_report, write_json


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baseball-bot",
        description="Analyze today's MLB games: who wins, who hits HRs, and best picks.",
    )
    parser.add_argument(
        "--date",
        type=_parse_date,
        default=None,
        help="Slate date YYYY-MM-DD (default: today in US/Eastern).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write full JSON report.",
    )
    parser.add_argument(
        "--top-hrs",
        type=int,
        default=15,
        help="How many HR candidates to show (default 15).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    client = MLBClient()
    game_date = args.date or client.today_et()
    games = client.fetch_schedule(game_date)
    report = analyze_slate(games, game_date.isoformat())
    print_report(report, top_hrs=args.top_hrs)

    if args.json_out:
        write_json(report, args.json_out)
        print(f"\nWrote JSON report to {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
