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

from helpers.reader import get_service, load_all_periods, get_latest_week_indices, extract_week_data, get_active_period, get_last_completed_period
from helpers.ai import analyze, translate_to_spanish
from helpers.mailer import send_analysis
from helpers.events import consume_pending_events, mark_event_processed
from training_shared.events import EventType

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


# Map EventType values to their --mode string equivalents.
# str(EventType.RUN_GLOBAL) == "run:global", which matches the --mode choices.
EVENT_TO_MODE = {
    EventType.RUN_GLOBAL:      "global",
    EventType.RUN_MONTHLY:     "monthly",
    EventType.RUN_NEW_ROUTINE: "new-routine",
    EventType.RUN_WEEKLY:      "weekly",
}


def run_analysis(mode, args, service, periods, periods_override=None):
    """
    Runs one analysis mode end-to-end: build prompt → call AI → translate → save → email.

    Args:
        mode:             One of "global", "monthly", "new-routine", "weekly".
        args:             The parsed argparse Namespace.
        service:          Authenticated Google Sheets service object.
        periods:          Full list of periods (used by global and weekly).
        periods_override: If provided, replaces periods[0] for monthly/new-routine.
                          Used by the routine:uploaded handler to pass the correct
                          period (last completed for monthly, active for new-routine).

    Returns:
        True if the analysis ran and was emailed, False if skipped (no changes).
    """
    today = datetime.now().strftime("%d/%m/%Y")

    # For monthly and new-routine, allow the caller to specify which period to use.
    # For global and weekly, periods[0] is always the right starting point.
    target_period = periods_override if periods_override is not None else periods[0]

    current_week_data = None
    prev_week_data    = None
    current_week_num  = None

    if mode == "weekly":
        current_idx, prev_idx = get_latest_week_indices(target_period)

        if current_idx is None:
            print("No data found in the current period for weekly analysis.")
            return False

        current_week_num  = current_idx + 1
        current_week_data = extract_week_data(target_period, current_idx)
        prev_week_data    = extract_week_data(target_period, prev_idx) if prev_idx is not None else None
        change_data       = current_week_data
    elif mode == "new-routine":
        change_data = target_period["days"]
    elif mode == "monthly":
        change_data = target_period
    else:
        change_data = periods

    # Skip if data hasn't changed since last run (only for real runs, not mock).
    if not args.mock and not has_changed(change_data, mode):
        print(f"[{mode}] No changes since last run. Skipping.")
        return False

    # Build a periods list where index 0 is the target period, for prompt builders.
    periods_for_prompt = [target_period] + [p for p in periods if p is not target_period]

    print(f"[{mode}] Analyzing with Groq...")
    analysis = analyze(
        periods_for_prompt,
        args.api_key,
        mock=args.mock,
        mode=mode,
        goal=args.goal,
        current_week_data=current_week_data,
        prev_week_data=prev_week_data,
        current_week_num=current_week_num,
    )

    if not args.mock:
        print(f"[{mode}] Translating to Spanish...")
        analysis = translate_to_spanish(analysis, args.api_key)

    ANALYSES_DIR.mkdir(exist_ok=True)
    for old_file in ANALYSES_DIR.glob(f"analysis_{mode}_*.md"):
        old_file.unlink()

    output_path = ANALYSES_DIR / f"analysis_{mode}_{datetime.now().strftime('%Y%m%d')}.md"
    output_path.write_text(analysis, encoding="utf-8")
    print(f"[{mode}] Analysis saved to: {output_path}")

    if not args.email_from or not args.email_password:
        print("Error: EMAIL_FROM and EMAIL_PASSWORD are required (set them in .env).")
        sys.exit(1)

    subject = EMAIL_SUBJECTS.get(mode, EMAIL_SUBJECTS["global"])(today)
    print(f"[{mode}] Sending email to {args.email_to}...")
    send_analysis(args.email_from, args.email_password, args.email_to, subject, analysis)
    print(f"[{mode}] Done.")
    return True


def main():
    # argparse lets users pass options like: python3 analyze.py --mode weekly --mock
    # Each add_argument() call defines one option. default= sets the value when not provided.
    parser = argparse.ArgumentParser(
        description="AI-powered gym training analysis."
    )
    parser.add_argument("--mode",
        choices=["global", "new-routine", "monthly", "weekly"],
        default=None,
        help="Analysis mode. If omitted, pending events from events.db are consumed instead.")
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
        sys.exit(1)

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

    # --- Event consumer mode (no --mode flag passed) ---
    if args.mode is None:
        events = consume_pending_events()
        if not events:
            print("No pending events. Nothing to do.")
            return
        print(f"Found {len(events)} pending event(s).")
        for event in events:
            event_type = event["event_type"]
            print(f"\nProcessing event: {event_type}")

            if event_type == EventType.ROUTINE_UPLOADED:
                # Semantic event from pdf2xls-generator: a full upload cycle just completed.
                # We run 3 analyses, each using the correct period:
                #   - monthly + global → last COMPLETED period (Fecha-Fecha, not the new one)
                #   - new-routine      → current ACTIVE period (Fecha-...)
                last_completed = get_last_completed_period(periods)
                active         = get_active_period(periods)

                if last_completed:
                    print(f"  Last completed period: {last_completed['period']}")
                    run_analysis("monthly",     args, service, periods, periods_override=last_completed)
                    run_analysis("global",      args, service, periods)
                else:
                    print("  No completed period found — skipping monthly and global.")

                if active:
                    print(f"  Active period: {active['period']}")
                    run_analysis("new-routine", args, service, periods, periods_override=active)
                else:
                    print("  No active period found — skipping new-routine.")

            else:
                # Manual run:* events (triggered via CLI or for testing)
                mode = EVENT_TO_MODE.get(event_type)
                if mode is None:
                    print(f"  Unknown event type: {event_type!r} — skipping.")
                else:
                    run_analysis(mode, args, service, periods)

            mark_event_processed(event["id"])
        return

    # --- Single-mode CLI path (--mode was passed explicitly) ---
    print(f"Mode: {args.mode} | Goal: {args.goal}")
    run_analysis(args.mode, args, service, periods)


# This block only runs when the script is executed directly with python3.
# If another file imports this module, this block does NOT run.
# It's a Python convention to protect the entry point.
if __name__ == "__main__":
    main()
