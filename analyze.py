#!/usr/bin/env python3
"""
analyze.py — Entry point for the training routine analyzer.

Available modes (--mode):
  global       Full analysis of the entire history. Detects trends
               and evaluates whether the long-term goal is being met.
  new-routine  Post pdf2xls: analyzes the new routine against the history.
  monthly      Balance of the most recent month. How did it go? Was the goal met?
  weekly       Compares the current week with the previous one.

Scheduled runs (NOT YET ACTIVE — enable via crontab when ready):
  weekly  → Saturdays at 12:00
              cron: 0 12 * * 6 cd /path/to/routine-analyzer && python3 analyze.py --mode weekly

  global  → Saturdays at 12:00, BUT only if the last training day of the active
              period is complete (all exercises have at least one peso filled in,
              excluding the abdomen group).
              Requires a pre-check before calling run_analysis("global", ...).
              cron: 0 12 * * 6 cd /path/to/routine-analyzer && python3 analyze.py --mode global

  (no-args) → Daily at 08:00, to detect and close completed periods
              (runs monthly + global + renames the tab when 2 open tabs are found).
              cron: 0 8 * * * cd /path/to/routine-analyzer && python3 analyze.py

Flow:
  1. Connects to Google Sheets with the service account.
  2. Loads the necessary periods according to the mode.
  3. Builds the appropriate prompt and calls Groq.
  4. Saves the analysis to a .md file and sends it by email.

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

from helpers.reader import get_service, get_write_service, load_all_periods, get_latest_week_indices, extract_week_data, get_active_period, get_last_completed_period, get_all_open_periods, rename_tab, is_active_period
from helpers.ai import analyze, get_settled_weights_dict, get_weight_suggestions
from helpers.mailer import send_analysis
from helpers.events import consume_pending_events, mark_event_processed
from helpers.writer import strip_suggestions_block, write_suggestions_to_sheet, format_suggestions_for_email, validate_suggestions
from helpers.catalog import ensure_classified, calculate_volume, format_volume_block, get_axial_load_exercises
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
    "global":      lambda d: f"Análisis global — {d}",
    "new-routine": lambda d: f"Nueva rutina — ¿Es adecuada para el objetivo?",
    "monthly":     lambda d: f"Balance mensual — {d}",
    "weekly":      lambda d: f"Semana de entrenamiento — {d}",
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


def try_close_completed_periods(args, service, periods):
    """
    Detects periods that have finished but haven't been closed yet, runs
    monthly + global analysis on them, and then closes (renames) their tab.

    Logic:
      - A period is "open" when its tab name ends with '-...'
      - Normally there is exactly one open period (the current active one)
      - When a new routine is uploaded before the old one is closed, there are
        temporarily TWO open periods: the new active one and the just-finished one
      - The just-finished one is the SECOND-TO-LAST (anteúltimo) open period
        (periods are ordered most-recent-first, so it's index 1)
      - End date for closing: one day before the newer period's start date

    This runs daily so that as soon as the new routine is uploaded the old period
    gets its reports and gets closed — without pdf2xls needing to know about it.
    """
    from datetime import datetime, timedelta

    open_periods = get_all_open_periods(periods)
    if len(open_periods) < 2:
        return  # Nothing to close

    # The anteúltimo is the second in the list (index 1 = second-most-recent)
    to_close  = open_periods[1]
    active    = open_periods[0]   # the new, truly active period

    tab_name  = to_close["period"]   # e.g. "18/05/26-..."
    start_str = tab_name.replace("-...", "")  # e.g. "18/05/26"

    # Compute end date = one day before the new period's start date
    new_start_str = active["period"].replace("-...", "")  # e.g. "25/05/26"
    try:
        new_start = datetime.strptime(new_start_str, "%d/%m/%y")
        end_date  = (new_start - timedelta(days=1)).strftime("%d/%m/%y")
    except ValueError:
        end_date = datetime.now().strftime("%d/%m/%y")

    closed_name = f"{start_str}-{end_date}"
    print(f"\n[close] Detected completed period: '{tab_name}'")
    print(f"[close] Will run monthly + global, then rename to '{closed_name}'")

    # Run monthly on the just-finished period
    run_analysis("monthly", args, service, periods, periods_override=to_close)

    # Run global (uses full history — no periods_override needed)
    run_analysis("global", args, service, periods)

    # Close the period: rename the tab
    write_svc = get_write_service(args.credentials)
    rename_tab(write_svc, args.sheets_id, tab_name, closed_name)
    print(f"[close] Period closed: '{closed_name}'")


def run_analysis(mode, args, service, periods, periods_override=None, return_only=False):
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
        return_only:      If True, skip sending email and return (subject, analysis)
                          so the caller can combine multiple analyses into one email.

    Returns:
        - When return_only=False: True if emailed, False if skipped.
        - When return_only=True:  (subject, analysis) tuple, or None if skipped.
    """
    today = datetime.now().strftime("%d/%m/%Y")

    # ── Period selection ───────────────────────────────────────────────────────
    # monthly: analyze the last COMPLETED period — the active tab only has
    #          AI suggestions, not real training data.
    if mode == "monthly" and periods_override is None:
        periods_override = get_last_completed_period(periods)
        if periods_override is None:
            print(f"[monthly] No completed period found. Skipping.")
            return False

    # For weekly/new-routine: use override if given, else periods[0] (active).
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

    # Skip if data hasn't changed since last run (only for real runs, not mock or forced).
    if not args.mock and not getattr(args, "force", False) and not has_changed(change_data, mode):
        print(f"[{mode}] No changes since last run. Skipping.")
        return False

    # Load the previous report for this mode (if any) so the AI can follow up on it.
    ANALYSES_DIR.mkdir(exist_ok=True)
    prev_report = None
    prev_files = sorted(ANALYSES_DIR.glob(f"analysis_{mode}_*.md"))
    if prev_files:
        prev_report = prev_files[-1].read_text(encoding="utf-8")

    # ── Build periods list for the prompt ──────────────────────────────────────
    # global: only completed periods — the active tab has no real data yet
    #         (only AI suggestions written in italic).
    if mode == "global":
        periods_for_prompt = [p for p in periods if not is_active_period(p)]
    else:
        periods_for_prompt = [target_period] + [p for p in periods if p is not target_period]

    print(f"[{mode}] Analyzing with Groq...")

    # For new-routine: pre-calculate volume so the AI gets real numbers, not estimates.
    volume_block = None
    axial_load_exercises = None
    if mode == "new-routine":
        all_exercise_names = [
            ex["name"]
            for day in target_period["days"]
            for ex in day["exercises"]
        ]
        catalog = ensure_classified(all_exercise_names, args.api_key)
        volume  = calculate_volume(target_period, catalog)
        volume_block = format_volume_block(volume)
        axial_load_exercises = get_axial_load_exercises(all_exercise_names, catalog)

    analysis = analyze(
        periods_for_prompt,
        args.api_key,
        mock=args.mock,
        mode=mode,
        goal=args.goal,
        current_week_data=current_week_data,
        prev_week_data=prev_week_data,
        current_week_num=current_week_num,
        prev_report=prev_report,
        volume_block=volume_block,
        axial_load_exercises=axial_load_exercises,
    )

    # The system prompt now instructs the AI to respond directly in Spanish,
    # so no separate translation call is needed (and it would waste ~4k tokens).

    # For new-routine: get structured weight suggestions via separate Gemini call.
    if mode == "new-routine" and not args.mock:
        prior_for_suggestions = periods[1:] if len(periods) > 1 else []
        suggestions = get_weight_suggestions(
            target_period,
            prior_for_suggestions,
            args.api_key,
            goal=args.goal,
            axial_load_exercises=axial_load_exercises,
        )
        if suggestions:
            # Validate against pre-calculated baselines (safety net for outliers)
            settled_dict = get_settled_weights_dict(target_period, prior_for_suggestions)
            suggestions = validate_suggestions(suggestions, settled_dict)
            import re as _re
            active_tab = next(
                (p["period"] for p in periods
                 if _re.match(r"^\d{2}/\d{2}/\d{2}-\.\.\.$", p["period"])),
                None
            )
            if active_tab:
                print(f"[new-routine] Writing {len(suggestions)} weight suggestion(s) to '{active_tab}'...")
                write_service = get_write_service(args.credentials)
                write_suggestions_to_sheet(write_service, args.sheets_id, active_tab, suggestions, overwrite_italic=True)
            else:
                print("[new-routine] No active tab found — skipping sheet write.")
            analysis += format_suggestions_for_email(suggestions)
        else:
            print("[new-routine] Structured call returned no suggestions.")

    for old_file in ANALYSES_DIR.glob(f"analysis_{mode}_*.md"):
        old_file.unlink()

    output_path = ANALYSES_DIR / f"analysis_{mode}_{datetime.now().strftime('%Y%m%d')}.md"
    output_path.write_text(analysis, encoding="utf-8")
    print(f"[{mode}] Analysis saved to: {output_path}")

    subject = EMAIL_SUBJECTS.get(mode, EMAIL_SUBJECTS["global"])(today)

    if return_only:
        # Caller will aggregate multiple analyses into one email.
        print(f"[{mode}] Done (held for combined email).")
        return subject, analysis

    if not args.email_from or not args.email_password:
        print("Error: EMAIL_FROM and EMAIL_PASSWORD are required (set them in .env).")
        sys.exit(1)

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
    parser.add_argument("--api-key",        default=os.getenv("GEMINI_API_KEY"))
    # os.getenv("GOAL", "hypertrophy") reads GOAL from .env, falling back to "hypertrophy"
    parser.add_argument("--goal",           default=os.getenv("GOAL", "hypertrophy"),
        help="Training objective injected into all prompts (default: hypertrophy)")
    # store_true means the flag is a boolean: present = True, absent = False
    parser.add_argument("--mock",           action="store_true")
    parser.add_argument("--force",          action="store_true",
                        help="Skip the 'no changes' check and always run the analysis")
    parser.add_argument("--max-periods",    type=int, default=None)
    parser.add_argument("--target-period",  default=None,
                        help="Tab name to use as target period for monthly/new-routine, e.g. '20/04/26-15/05/26'")
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
        if events:
            print(f"Found {len(events)} pending event(s).")
            for event in events:
                event_type = event["event_type"]
                print(f"\nProcessing event: {event_type}")

                if event_type == EventType.ROUTINE_UPLOADED:
                    # Semantic event from pdf2xls-generator: only run new-routine.
                    # monthly and global are handled by try_close_completed_periods() below.
                    active = get_active_period(periods)

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
        else:
            print("No pending events.")

        # Daily check: close any period that finished but hasn't been closed yet.
        # Runs regardless of whether there were events — safe to call every day.
        try_close_completed_periods(args, service, periods)
        return

    # --- Single-mode CLI path (--mode was passed explicitly) ---
    print(f"Mode: {args.mode} | Goal: {args.goal}")
    target_override = None
    if args.target_period:
        target_override = next((p for p in periods if p["period"] == args.target_period), None)
        if target_override is None:
            print(f"Error: period '{args.target_period}' not found. Available: {[p['period'] for p in periods]}")
            sys.exit(1)
        print(f"Using target period: {target_override['period']}")
    run_analysis(args.mode, args, service, periods, periods_override=target_override)


# This block only runs when the script is executed directly with python3.
# If another file imports this module, this block does NOT run.
# It's a Python convention to protect the entry point.
if __name__ == "__main__":
    main()
