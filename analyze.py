#!/usr/bin/env python3
"""
analyze.py — Entry point for the training routine analyzer.

Available modes (--mode):
  global       (default) Full analysis of the entire history. Detects trends
               and evaluates whether the long-term goal is being met.
  new-routine  Post pdf2xls: analyzes the new routine against the history.
               Is it suitable for the goal? What would change?
  monthly      Balance of the most recent month. How did it go? Was the goal met?
  weekly       Compares the current week with the previous one (Sunday cron).

Flow:
  1. Connects to Google Sheets with the service account.
  2. Loads the necessary periods according to the mode.
  3. Builds the appropriate prompt and calls Groq.
  4. Saves the analysis to a .md file and optionally sends it by email.

Minimal usage (config in .env):
    python3 analyze.py
    python3 analyze.py --mode new-routine
    python3 analyze.py --mode weekly --mock
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from helpers.reader import get_service, load_all_periods, get_latest_week_indices, extract_week_data
from helpers.ai import analyze, translate_to_spanish
from helpers.mailer import send_analysis

load_dotenv()

HASH_FILE_TEMPLATE = ".last_data_hash_{mode}"

EMAIL_SUBJECTS = {
    "global":      lambda d: f"Global analysis — {d}",
    "new-routine": lambda d: f"New routine — Is it suitable for the goal?",
    "monthly":     lambda d: f"Monthly balance — {d}",
    "weekly":      lambda d: f"Training week — {d}",
}


def compute_hash(data):
    """Generates an MD5 hash of the serialized content to detect changes."""
    content = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()


def has_changed(data, mode):
    """
    Returns True if the data has changed since the last run of the given mode.
    Updates the hash file if there were changes.
    """
    hash_file = Path(HASH_FILE_TEMPLATE.format(mode=mode.replace("-", "_")))
    current_hash = compute_hash(data)
    if hash_file.exists() and hash_file.read_text().strip() == current_hash:
        return False
    hash_file.write_text(current_hash)
    return True


ANALYSES_DIR = Path("analyses")


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered gym training analysis."
    )
    parser.add_argument("--mode",
        choices=["global", "new-routine", "monthly", "weekly"],
        default="global",
        help="Analysis mode (default: global)")
    parser.add_argument("--sheets-id",      default=os.getenv("SHEETS_ID"))
    parser.add_argument("--credentials",    default=os.getenv("CREDENTIALS"))
    parser.add_argument("--api-key",        default=os.getenv("GROQ_API_KEY"))
    parser.add_argument("--goal",           default=os.getenv("GOAL", "hypertrophy"),
        help="Training objective injected into all prompts (default: hypertrophy)")
    parser.add_argument("--mock",           action="store_true")
    parser.add_argument("--max-periods",    type=int, default=None)
    parser.add_argument("--email-to",       default=os.getenv("EMAIL_TO"))
    parser.add_argument("--email-from",     default=os.getenv("EMAIL_FROM"))
    parser.add_argument("--email-password", default=os.getenv("EMAIL_PASSWORD"))
    args = parser.parse_args()

    if not args.sheets_id or not args.credentials or not args.api_key:
        print("Error: --sheets-id, --credentials and --api-key are required (or set them in .env).")
        sys.exit(1)

    today = datetime.now().strftime("%d/%m/%Y")

    print(f"Mode: {args.mode} | Goal: {args.goal}")
    print("Connecting to Google Sheets...")
    service = get_service(args.credentials)

    print("Loading training periods...")
    periods = load_all_periods(service, args.sheets_id)

    if not periods:
        print("No data found in the spreadsheet.")
        sys.exit(1)

    if args.max_periods:
        periods = periods[:args.max_periods]

    print(f"Found {len(periods)} period(s): {', '.join(p['period'] for p in periods)}")

    # --- Prepare data according to the mode and detect changes ---

    current_week_data = None
    prev_week_data    = None
    current_week_num  = None

    if args.mode == "weekly":
        current_period = periods[0]
        current_idx, prev_idx = get_latest_week_indices(current_period)

        if current_idx is None:
            print("No data found in the current period for weekly analysis.")
            sys.exit(1)

        current_week_num  = current_idx + 1  # convert to 1-based for display
        current_week_data = extract_week_data(current_period, current_idx)
        prev_week_data    = extract_week_data(current_period, prev_idx) if prev_idx is not None else None

        change_data = current_week_data
    elif args.mode == "new-routine":
        change_data = periods[0]["days"]  # only the new routine
    elif args.mode == "monthly":
        change_data = periods[0]
    else:
        change_data = periods

    if not args.mock and not has_changed(change_data, args.mode):
        print("No changes since last run. Nothing to do.")
        sys.exit(0)

    print("Analyzing with Groq...")
    analysis = analyze(
        periods,
        args.api_key,
        mock=args.mock,
        mode=args.mode,
        goal=args.goal,
        current_week_data=current_week_data,
        prev_week_data=prev_week_data,
        current_week_num=current_week_num,
    )

    if not args.mock:
        print("Translating to Spanish...")
        analysis = translate_to_spanish(analysis, args.api_key)

    ANALYSES_DIR.mkdir(exist_ok=True)

    # Delete the previous analysis of this mode before saving the new one
    for old_file in ANALYSES_DIR.glob(f"analysis_{args.mode}_*.md"):
        old_file.unlink()

    output_path = ANALYSES_DIR / f"analysis_{args.mode}_{datetime.now().strftime('%Y%m%d')}.md"
    output_path.write_text(analysis, encoding="utf-8")
    print(f"Analysis saved to: {output_path}")

    if not args.email_from or not args.email_password:
        print("Error: EMAIL_FROM and EMAIL_PASSWORD are required (set them in .env).")
        sys.exit(1)
    subject = EMAIL_SUBJECTS.get(args.mode, EMAIL_SUBJECTS["global"])(today)
    print(f"Sending email to {args.email_to}...")
    send_analysis(args.email_from, args.email_password, args.email_to, subject, analysis)
    print("Done.")


if __name__ == "__main__":
    main()
