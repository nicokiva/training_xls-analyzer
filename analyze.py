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
  4. Translates the analysis to Spanish.
  5. Saves the analysis to a .md file and sends it by email.

Minimal usage (config in .env):
    python3 analyze.py
    python3 analyze.py --mode new-routine
    python3 analyze.py --mode weekly --mock
"""

import argparse   # standard library: parses command-line arguments like --mode, --mock
import hashlib    # standard library: generates MD5 hashes to detect data changes
import json       # standard library: serializes Python dicts to strings (needed for hashing)
import os         # standard library: reads environment variables with os.getenv()
import sys        # standard library: sys.exit() stops the script with an exit code
from datetime import datetime  # we only need the datetime class, not the whole module
from pathlib import Path       # Path is the modern way to handle file paths in Python (vs raw strings)

from dotenv import load_dotenv  # third-party: reads .env file into environment variables

from helpers.reader import get_service, load_all_periods, get_latest_week_indices, extract_week_data
from helpers.ai import analyze, translate_to_spanish
from helpers.mailer import send_analysis

# load_dotenv() must be called before os.getenv() so the .env values are available.
# CLI arguments parsed below will override these if provided.
load_dotenv()

# Template for per-mode hash files. The {mode} placeholder is filled in has_changed().
# We use one file per mode so each mode tracks its own "last run" state independently.
HASH_FILE_TEMPLATE = ".last_data_hash_{mode}"

# Dict mapping mode names to functions that generate the email subject.
# Lambda is a one-line anonymous function: lambda d: f"..." is equivalent to
# def make_subject(d): return f"..."
EMAIL_SUBJECTS = {
    "global":      lambda d: f"Global analysis — {d}",
    "new-routine": lambda d: f"New routine — Is it suitable for the goal?",
    "monthly":     lambda d: f"Monthly balance — {d}",
    "weekly":      lambda d: f"Training week — {d}",
}


def compute_hash(data):
    """
    Generates an MD5 hash of the serialized content to detect changes.

    We serialize the data to JSON first (json.dumps) so we can hash any
    Python dict/list as a stable string. sort_keys=True ensures the same
    dict always produces the same string regardless of key insertion order.
    """
    content = json.dumps(data, sort_keys=True, ensure_ascii=False)
    # encode() converts the string to bytes — hashlib works with bytes, not strings
    return hashlib.md5(content.encode()).hexdigest()


def has_changed(data, mode):
    """
    Returns True if the data has changed since the last run of the given mode.
    Updates the hash file if there were changes.

    This prevents sending duplicate emails when the spreadsheet hasn't changed.
    Each mode has its own hash file so they don't interfere with each other.
    """
    hash_file = Path(HASH_FILE_TEMPLATE.format(mode=mode.replace("-", "_")))
    current_hash = compute_hash(data)
    # Path.exists() checks if the file is on disk. Path.read_text() reads it as a string.
    if hash_file.exists() and hash_file.read_text().strip() == current_hash:
        return False
    # Write the new hash so next run can compare against it
    hash_file.write_text(current_hash)
    return True


# Path() creates a path object. "analyses" is relative to wherever the script is run from.
# Using Path instead of a plain string lets us use / to join paths: ANALYSES_DIR / "file.md"
ANALYSES_DIR = Path("analyses")


def main():
    # argparse lets users pass options like: python3 analyze.py --mode weekly --mock
    # Each add_argument() call defines one option. default= sets the value when not provided.
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
    # os.getenv("GOAL", "hypertrophy") reads GOAL from .env, falling back to "hypertrophy"
    parser.add_argument("--goal",           default=os.getenv("GOAL", "hypertrophy"),
        help="Training objective injected into all prompts (default: hypertrophy)")
    # store_true means the flag is a boolean: present = True, absent = False
    parser.add_argument("--mock",           action="store_true")
    parser.add_argument("--max-periods",    type=int, default=None)
    parser.add_argument("--email-to",       default=os.getenv("EMAIL_TO"))
    parser.add_argument("--email-from",     default=os.getenv("EMAIL_FROM"))
    parser.add_argument("--email-password", default=os.getenv("EMAIL_PASSWORD"))
    # parse_args() reads sys.argv (the command line) and returns a Namespace object.
    # Access values as: args.mode, args.mock, args.sheets_id (note: hyphens become underscores)
    args = parser.parse_args()

    if not args.sheets_id or not args.credentials or not args.api_key:
        print("Error: --sheets-id, --credentials and --api-key are required (or set them in .env).")
        # sys.exit(1) exits with error code 1 — convention: 0 = success, anything else = error
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

    # List slicing: periods[:N] returns the first N elements.
    # If max_periods is None, this block is skipped and we use all periods.
    if args.max_periods:
        periods = periods[:args.max_periods]

    # Generator expression inside join(): iterates periods and extracts "period" key
    # without building an intermediate list — more memory-efficient than a list comprehension
    print(f"Found {len(periods)} period(s): {', '.join(p['period'] for p in periods)}")

    # --- Prepare data according to the mode and detect changes ---

    # These are initialized to None. Only weekly mode will fill them in.
    current_week_data = None
    prev_week_data    = None
    current_week_num  = None

    if args.mode == "weekly":
        current_period = periods[0]
        current_idx, prev_idx = get_latest_week_indices(current_period)

        if current_idx is None:
            print("No data found in the current period for weekly analysis.")
            sys.exit(1)

        # +1 to convert from 0-based index to human-readable week number (Week 1, 2, 3, 4)
        current_week_num  = current_idx + 1
        current_week_data = extract_week_data(current_period, current_idx)
        # Ternary expression: value_if_true if condition else value_if_false
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

    # Translate only real analyses — mock output is already readable for testing
    if not args.mock:
        print("Translating to Spanish...")
        analysis = translate_to_spanish(analysis, args.api_key)

    # mkdir(exist_ok=True) creates the directory if it doesn't exist.
    # Without exist_ok=True it would raise an error if the dir already exists.
    ANALYSES_DIR.mkdir(exist_ok=True)

    # glob() finds files matching a pattern. The * is a wildcard matching any characters.
    # We delete the previous file for this mode so we never accumulate stale analyses.
    for old_file in ANALYSES_DIR.glob(f"analysis_{args.mode}_*.md"):
        old_file.unlink()  # unlink() deletes a file (same as os.remove())

    output_path = ANALYSES_DIR / f"analysis_{args.mode}_{datetime.now().strftime('%Y%m%d')}.md"
    # write_text() writes a string to a file, creating it if it doesn't exist.
    # encoding="utf-8" is important for Spanish characters (ó, é, ñ, etc.)
    output_path.write_text(analysis, encoding="utf-8")
    print(f"Analysis saved to: {output_path}")

    if not args.email_from or not args.email_password:
        print("Error: EMAIL_FROM and EMAIL_PASSWORD are required (set them in .env).")
        sys.exit(1)
    # .get() on a dict returns None if the key doesn't exist, avoiding a KeyError
    subject = EMAIL_SUBJECTS.get(args.mode, EMAIL_SUBJECTS["global"])(today)
    print(f"Sending email to {args.email_to}...")
    send_analysis(args.email_from, args.email_password, args.email_to, subject, analysis)
    print("Done.")


# This block only runs when the script is executed directly with python3.
# If another file imports this module, this block does NOT run.
# It's a Python convention to protect the entry point.
if __name__ == "__main__":
    main()
