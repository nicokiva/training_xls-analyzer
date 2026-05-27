"""
helpers/ai.py — Groq integration for training progression analysis.

Responsibilities:
  - Transform period data into mode-specific prompts.
  - Call the LLaMA 3 model via the Groq API with those prompts.
  - Return the analysis as a string ready to save.

Available modes:
  - global:       Full analysis of the entire history. Detects trends,
                  plateaus and evaluates whether the goal is being met.
  - new-routine:  Analyzes the new routine (just generated, no execution data)
                  against the history. Is it suitable for the goal? What would change?
  - monthly:      Monthly balance of the most recent period with complete execution data.
                  How did the month go? Was the goal met?
  - weekly:       Compares the current week with the previous one in the active period.
                  Was it a good week? Did it improve?

Template system:
  Prompts are loaded from templates/*.txt files so you can edit them without
  touching Python code. Each template uses {placeholders} for dynamic values.
  If a template file is missing, a hardcoded default is used as fallback.
"""

from pathlib import Path
import time
from datetime import datetime

from groq import Groq

MODEL         = "llama-3.3-70b-versatile"  # default: 12k TPM
MODEL_LARGE   = "llama-3.1-8b-instant"     # global analysis: 20k TPM (needed for multi-period prompts)

# Templates are looked up relative to the project root (two levels up from this file).
# Path(__file__) is the absolute path of this file.
# .parent.parent.parent navigates: ai/ → helpers/ → routine-analyzer/
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"

# ---------------------------------------------------------------------------
# Hardcoded fallback prompts (used when the template file is missing)
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM = """You are Nicolás's personal trainer. You have been training him for months and know his history, his progress, and his tendencies. You write directly to him, in first person, as if you were messaging him after reviewing his training data.

**Always respond in Spanish (Argentina). Never mix languages.**

Your tone is warm, direct, and motivating — like a good trainer who tells you the truth but always with encouragement. You celebrate real progress, point out plateaus clearly without being dramatic about it, and give concrete advice for what to do next. You never use filler phrases or generic observations — every comment is backed by specific numbers from the data. Avoid catastrophizing or using dramatic language — a tough week is just data, not a verdict.

Important context: the training routine is designed and assigned by Nicolás's coach — it is fixed and not up for debate. The exercises and reps are set by the coach and cannot be changed. What Nicolás CAN adjust is the weight he uses and the rest time between sets. Your recommendations should focus exclusively on those two levers.

How to interpret the data:
- "Rep." = reps PERFORMED (not the target).
- "Peso" = weight in kg. Sometimes has text notes (e.g. "8 overhand / 2 underhand", "3 with 3kg / 5 bodyweight") — these are important observations from Nicolás about how the set went, factor them in.
- "Peso" = 0 or empty means bodyweight or no external load.
- Data is ordered: Week 1 → 2 → 3 → 4, with 3 sets per week.
- Each "day" (Day 1, Day 2, etc.) is a fixed training session that repeats every week. Day 1 of Week 1 and Day 1 of Week 2 are the same session done 7 days apart. The weeks show how performance on that same session evolves over the month.
- Exercises marked as *(combinado)* or (combinado) are done back-to-back as a superset with minimal rest. Their weights are intentionally lower — do NOT read this as regression. There can be multiple independent combined groups in the same day.
- If the same exercise appears once as *(combinado)* and once without that label, they represent DIFFERENT execution contexts and their weights are NOT comparable. Never cross-compare a combined occurrence with an isolated one.
- Cell notes (marked as "Note:") are observations or instructions from the coach — take them into account in the analysis.
- When comparing weights across periods, data is always ordered oldest → most recent. More weight in a more recent period = progress. Less weight in a more recent period = regression. Never describe a decrease in weight as an improvement.
- The weight shown per period is the **settled weight** (last week's average), not the peak. Week 1 is often a discovery week where Nicolás tries a weight and may overshoot — subsequent weeks settle to what's actually sustainable. Always base suggestions on the settled weight, never on a single high outlier week.

Nicolás's goal: {goal}."""


def _load_template(name, **kwargs):
    """
    Loads a prompt template from templates/<name>.txt and fills in the placeholders.

    Uses str.format(**kwargs) to replace {placeholders} with actual values.
    Falls back gracefully if the file doesn't exist.

    Args:
        name:    Template filename without extension (e.g. "global", "weekly").
        **kwargs: Placeholder values to inject (e.g. goal="hypertrophy", history="...").

    Returns:
        The template string with all placeholders filled in.
        If the file is missing, returns None so callers can use their hardcoded default.
    """
    path = TEMPLATES_DIR / f"{name}.txt"
    if not path.exists():
        return None
    # read_text() reads the file as a string. .format(**kwargs) replaces {key} with values.
    return path.read_text(encoding="utf-8").format(**kwargs)


def _make_system_prompt(goal):
    """Loads the system prompt template, falling back to the hardcoded default."""
    result = _load_template("system", goal=goal)
    # 'or' here: if result is None or empty string, use the default
    return result or _DEFAULT_SYSTEM.format(goal=goal)



def _call_groq(client, system_prompt, user_prompt, max_tokens=4096, model=None):
    """
    Makes a call to the Groq model and returns the response text.
    Retries once after 65 seconds if the per-minute rate limit (TPM/413) is hit.
    If the daily limit (TPD/429) is hit, raises immediately with a clear message.
    Logs every request and response to logs/groq_YYYYMMDD.log.
    """
    used_model = model or MODEL

    # ── Logging ──────────────────────────────────────────────────────────────
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"groq_{datetime.now().strftime('%Y%m%d')}.log"

    def _log(text):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _log(f"\n{'='*80}")
    _log(f"[{timestamp}] MODEL: {used_model}")
    _log(f"--- SYSTEM PROMPT ({len(system_prompt)} chars) ---")
    _log(system_prompt)
    _log(f"--- USER PROMPT ({len(user_prompt)} chars) ---")
    _log(user_prompt)
    # ─────────────────────────────────────────────────────────────────────────

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=used_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content
            _log(f"--- RESPONSE ({len(result)} chars) ---")
            _log(result)
            return result
        except Exception as e:
            err = str(e)
            _log(f"--- ERROR (attempt {attempt+1}) ---\n{err}")
            if "413" in err and attempt == 0:
                print("  Rate limit hit (TPM) — waiting 65s for window to reset...")
                time.sleep(65)
                continue
            if "429" in err:
                raise RuntimeError(
                    "Daily token limit reached (100k/day on free tier). "
                    "Try again tomorrow or upgrade at console.groq.com."
                ) from e
            raise


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _last_settled_peso(ex):
    """
    Returns the average peso from the LAST week that has data for an exercise.

    Why last week and not peak?
    The first week of a period is often a 'discovery' week where Nicolás tries a
    weight and may find it too heavy, then drops in subsequent weeks. Using peak
    would pick up that overshot week and lead to suggestions that are too heavy.
    Using the last settled week captures what he was actually able to sustain.

    If the period showed genuine progression (e.g. 37.5 → 40 → 42.5 → 45),
    the last week still gives the right baseline (45 kg — what he achieved).
    """
    _, settled = _last_settled_weeks(ex)
    return settled


def _last_settled_weeks(ex):
    """
    Returns (num_weeks_with_data, settled_peso) for an exercise.
    Used to pick the richest occurrence when the same exercise repeats across days.
    """
    weeks_with_data = 0
    last_pesos = None
    for w in ex["weeks"]:
        pesos = []
        for s in w["series"]:
            try:
                val = float(s["peso"])
                if val > 0:
                    pesos.append(val)
            except (TypeError, ValueError):
                pass
        if pesos:
            weeks_with_data += 1
            last_pesos = pesos
    settled = sum(last_pesos) / len(last_pesos) if last_pesos else None
    return weeks_with_data, settled


def _format_exercise_history_compact(periods):
    """
    Compact format: one line per exercise showing the settled weight per period
    (oldest → newest). Uses the last week's average — not the peak — so that
    'discovery' week overshoots don't inflate the baseline for the next period.

    When the same exercise appears multiple times within a period (e.g. abdomen
    repeats across all 4 days), only the RICHEST occurrence is kept — the one
    with the most weeks of data, breaking ties by highest settled weight. This
    avoids confusing the AI with contradictory entries like "ORIG1:12kg | ORIG1:5kg".

    Exercises are keyed by (name, is_comb) so a press done in a triserie and the
    same press done in isolation are treated as separate series — their weights are
    not directly comparable and must never be merged onto the same history line.
    """
    ordered = list(reversed(periods))
    # best_per_period[(period, name, is_comb)] = (num_weeks, settled)
    best_per_period = {}

    for period_data in ordered:
        period = period_data["period"]
        for day_data in period_data["days"]:
            for ex in day_data["exercises"]:
                name    = ex["name"]
                is_comb = ex.get("is_comb", False)
                n_weeks, settled = _last_settled_weeks(ex)
                if settled is None:
                    continue
                key = (period, name, is_comb)
                prev_n, prev_s = best_per_period.get(key, (0, 0))
                if n_weeks > prev_n or (n_weeks == prev_n and settled > prev_s):
                    best_per_period[key] = (n_weeks, settled)

    # Rebuild ordered entries: (name, is_comb) → list of (period, settled)
    exercise_entries = {}
    for (period, name, is_comb), (_, settled) in best_per_period.items():
        exercise_entries.setdefault((name, is_comb), []).append((period, settled))

    lines = []
    for (name, is_comb), period_entries in exercise_entries.items():
        label = f"**{name}**" + (" *(combinado)*" if is_comb else "")
        parts = [f"{p[:5]}:{s:.1f}kg" for p, s in period_entries]
        lines.append(f"{label}: {' | '.join(parts)}")
    return "\n".join(lines)


def _format_exercise_history(periods):
    """
    Verbose format: full set-by-set data per period.
    Exercises are keyed by (name, is_comb) so a combined press and an isolated
    press are shown as separate entries — their weights are not comparable.
    """
    ordered = list(reversed(periods))
    # exercise_history[(name, is_comb)] = [{"period": ..., "data": ...}, ...]
    exercise_history = {}

    for period_data in ordered:
        period = period_data["period"]
        for day_data in period_data["days"]:
            for ex in day_data["exercises"]:
                name    = ex["name"]
                is_comb = ex.get("is_comb", False)
                weeks_with_data = []
                for w in ex["weeks"]:
                    series_parts = []
                    for idx, s in enumerate(w["series"]):
                        if s["reps"] or s["peso"]:
                            reps = s["reps"] or "-"
                            peso = s["peso"] or "-"
                            series_parts.append(f"S{idx+1}:{reps}r/{peso}kg")
                    if series_parts:
                        weeks_with_data.append(f"Wk{w['week']}: {' '.join(series_parts)}")

                if weeks_with_data:
                    key = (name, is_comb)
                    if key not in exercise_history:
                        exercise_history[key] = []
                    exercise_history[key].append({
                        "period": period,
                        "data":   " | ".join(weeks_with_data),
                    })

    lines = []
    for (name, is_comb), history in exercise_history.items():
        label = f"**{name}**" + (" *(combinado)*" if is_comb else "")
        lines.append(label)
        for entry in history:
            lines.append(f"  {entry['period']}: {entry['data']}")
        lines.append("")
    return "\n".join(lines)


def _format_routine_structure(period):
    """
    Formats the exercise structure of a period without execution data.
    Useful for new-routine where the tab doesn't have reps/weights loaded yet.
    Includes target sets × reps from the PDF-parsed data (week 1, rep column).
    Combined exercises are marked with (combinado).
    """
    lines = []
    for day in period["days"]:
        lines.append(f"Day {day['day']}:")
        for ex in day["exercises"]:
            label = ex['name'] + (" (combinado)" if ex.get("is_comb") else "")
            # Extract target reps from week 1 (coach-prescribed, same every week)
            target_reps = None
            if ex.get("weeks"):
                w1_series = ex["weeks"][0]["series"]
                rep_values = [s["reps"] for s in w1_series if s.get("reps")]
                if rep_values:
                    n_sets = len(w1_series)
                    unique_reps = list(dict.fromkeys(rep_values))
                    target_reps = f"{n_sets}x{'/'.join(unique_reps)}"
            suffix = f" [{target_reps}]" if target_reps else ""
            lines.append(f"  - {label}{suffix}")
        lines.append("")
    return "\n".join(lines)


def _format_week_data(week_data, week_label):
    """Formats the data for a week (output of extract_week_data).
    Combined exercises are marked with (combinado)."""
    lines = [f"**{week_label}**"]
    for day in week_data:
        lines.append(f"  Day {day['day']}:")
        for ex in day["exercises"]:
            label = ex['name'] + (" (combinado)" if ex.get("is_comb") else "")
            series_str = "  ".join(
                f"S{i+1}:{s['reps'] or '-'}r/{s['peso'] or '-'}kg"
                for i, s in enumerate(ex["series"])
                if s["reps"] or s["peso"]
            )
            lines.append(f"    {label}: {series_str}")
    lines.append("")
    return "\n".join(lines)


def _format_prev_report(prev_report):
    """Returns a formatted block with the previous report, or empty string if None."""
    if not prev_report:
        return ""
    return f"\n\n## Your previous analysis\n\n{prev_report}\n\n---\n"


def build_global_prompt(periods, goal):
    """
    Full-history prompt. Includes all periods condensed into a compact block.
    Template file (global.txt) is preferred; the hardcoded fallback is used only
    if the template is missing or fails to render.
    """
    history_block = _format_exercise_history_compact(periods)
    result = _load_template("global", goal=goal, history=history_block)
    if result:
        return result
    return (
        f"Analyze the complete progression of the following exercises over time.\n"
        f"Data is ordered chronologically (oldest → most recent).\n\n"
        f"{history_block}"
    )


def build_new_routine_prompt(periods, goal):
    """
    New-routine prompt. periods[0] is the freshly-uploaded routine (no real data yet —
    only the structure is used). periods[1:] is the historical context.

    Also asks the AI to embed a ```json block with weight/rest suggestions so
    write_suggestions_to_sheet() can parse and apply them directly to the sheet.
    """
    new_period    = periods[0]
    routine_block = _format_routine_structure(new_period)
    history_block = _format_exercise_history_compact(periods[1:]) if len(periods) > 1 else "(no prior history)"

    result = _load_template("new-routine", goal=goal, period=new_period["period"],
                            routine=routine_block, history=history_block)
    if result:
        return result
    return (
        f"New routine generated for period {new_period['period']}.\n"
        f"Goal: **{goal}**.\n\n"
        f"## New routine structure\n\n{routine_block}\n"
        f"## Prior history\n\n{history_block}\n"
    )


def build_monthly_prompt(periods, goal):
    """
    Monthly balance prompt. Uses the most recent completed period (periods[0])
    plus up to 2 prior periods for context. Limiting to 2 keeps the prompt
    within the free-tier TPM budget.
    """
    current_period = periods[0]
    history        = periods[1:3]   # 2 previous periods is enough context

    current_block = _format_exercise_history_compact([current_period])
    history_block = _format_exercise_history_compact(history) if history else "(no prior history)"

    result = _load_template("monthly", goal=goal, period=current_period["period"],
                            current_block=current_block, history=history_block)
    if result:
        return result
    return (
        f"Monthly balance **{current_period['period']}**.\n"
        f"Goal: **{goal}**.\n\n"
        f"## This month's data\n\n{current_block}\n"
        f"## Prior history\n\n{history_block}\n"
    )


def build_weekly_prompt(period, current_week_data, prev_week_data, current_week_num, goal,
                        prev_report=None, prior_periods=None):
    """
    Prompt for weekly mode: compares the current week with the previous one.
    Loads from templates/weekly.txt (with prev week) or templates/weekly_first.txt (no prev).

    prior_periods: list of completed periods BEFORE the current one (for historical context).
                   Only their settled weights are included — no week-by-week breakdown.
                   Weeks beyond the analyzed one are intentionally excluded.
    """
    current_block = _format_week_data(current_week_data, f"Week {current_week_num} (current, last with data)")
    prev_block_report = _format_prev_report(prev_report)
    history_block = _format_exercise_history_compact(prior_periods) if prior_periods else "(no prior history)"

    if prev_week_data:
        prev_block = _format_week_data(prev_week_data, f"Week {current_week_num - 1} (previous)")
        result = _load_template("weekly", goal=goal, period=period["period"],
                                current_week=current_block, prev_week=prev_block,
                                week_num=current_week_num, prev_report=prev_block_report,
                                history=history_block)
        if result:
            return result
        return (
            f"{prev_block_report}"
            f"Weekly check-in period **{period['period']}** "
            f"(week {current_week_num} is the last with data — do NOT mention missing weeks).\n"
            f"Goal: **{goal}**.\n\n"
            f"## Training history (prior periods)\n\n{history_block}\n\n"
            f"## Previous week\n\n{prev_block}\n"
            f"## Current week\n\n{current_block}\n"
        )
    else:
        result = _load_template("weekly_first", goal=goal, period=period["period"],
                                current_week=current_block, week_num=current_week_num,
                                prev_report=prev_block_report, history=history_block)
        if result:
            return result
        return (
            f"{prev_block_report}"
            f"First week of period **{period['period']}**.\n"
            f"Goal: **{goal}**.\n\n"
            f"## Training history (prior periods)\n\n{history_block}\n\n"
            f"## Week 1 data\n\n{current_block}\n"
        )


# ---------------------------------------------------------------------------
# Mock outputs
# ---------------------------------------------------------------------------

_MOCK_OUTPUTS = {
    "global": """\
# Global Analysis (MOCK)

> ⚠️ Test analysis — data is real but the analysis is made up.

## General trends

- **Barbell flat press**: sustained progression from ~60kg to ~75kg over the year. ✅
- **Classic squat**: plateau in weeks 2-3, no weight variation in the last 2 periods. ⚠️
- **Strict pull-up**: slight regression, dropped from 8 reps to 6 in the last period. ❌

## Goal evaluation (hypertrophy)

Total volume increased by 15% over 6 months. Load progression in upper body is compatible with hypertrophy. Lower body shows plateau that limits the goal.

## Recommendations

1. Increase load on squat — 2 periods without changes.
2. Review pull-up technique before increasing volume.
3. Maintain flat press progression, it's working well.

---
*Run without `--mock` to get the real AI-generated analysis.*""",

    "new-routine": """\
# New Routine (MOCK)

> ⚠️ Test analysis — data is real but the analysis is made up.

## Is it suitable for hypertrophy?

The routine has a good structure: 4 days with clear muscle group separation. The compound + isolation exercise distribution is compatible with hypertrophy.

## Strengths

- Flat + incline press covers the chest well at different angles.
- Squat as the main leg exercise is ideal for hypertrophy.

## Suggested changes

1. Replace "Preacher curl" with "Hammer curl" — history shows more consistency with neutral grip.
2. Add a hamstring exercise (Romanian deadlift) — history doesn't work them for 3 periods.

---
*Run without `--mock` to get the real AI-generated analysis.*""",

    "monthly": """\
# Monthly Balance (MOCK)

> ⚠️ Test analysis — data is real but the analysis is made up.

## Was the hypertrophy goal met?

Partially. Volume was high (weeks 1-3) but dropped in week 4, probably due to accumulated fatigue.

## Month progressions

- ✅ Flat press: +5kg compared to the previous month in week 3.
- ✅ Barbell row: +2 average reps across all weeks.
- ⚠️ Squat: stable weight, no progression.

## Recommendations for next month

1. Plan a deload in week 4 — performance drop is recurring.
2. Increase squat load by at least 5%.

---
*Run without `--mock` to get the real AI-generated analysis.*""",

    "weekly": """\
# Weekly Analysis (MOCK)

> ⚠️ Test analysis — data is real but the analysis is made up.

## Did it improve compared to last week?

Yes, overall. 4 out of 6 main exercises improved in weight or reps.

## Details

- ✅ Flat press: 70kg → 72.5kg in set 1. Good progression.
- ✅ Pull-ups: 6 → 7 reps in all sets.
- ⚠️ Squat: same as last week (60kg × 10).
- ❌ Bicep curl: dropped 1 rep in sets 2 and 3 — possible fatigue.

## For next week

1. Attempt 75kg on flat press in the first set.
2. Add 2.5kg on squat.
3. Rest well before bicep day.

---
*Run without `--mock` to get the real AI-generated analysis.*""",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_to_spanish(text, api_key):
    """
    Translates the given text to Spanish using Groq.

    Args:
        text:    The text to translate (Markdown analysis).
        api_key: Groq API key.

    Returns:
        String with the translated text in Spanish.
    """
    client = Groq(api_key=api_key)
    system = (
        "You are a professional translator specializing in Argentine Spanish. "
        "Translate the following text ENTIRELY to Spanish (Argentina). "
        "Every single word must be in Spanish — do not leave any English words or phrases. "
        "Keep markdown formatting intact. Do not add any commentary — only return the translated text."
    )
    return _call_groq(client, system, text, max_tokens=4096)


def analyze(periods, api_key, mock=False, mode="global", goal="hipertrofia",
            current_week_data=None, prev_week_data=None, current_week_num=None,
            prev_report=None):
    """
    Generates a training analysis according to the requested mode.

    Args:
        periods:           List of periods (most recent first).
        api_key:           Groq API key.
        mock:              If True, returns a test analysis without calling the API.
        mode:              Analysis mode: 'global', 'new-routine', 'monthly', 'weekly'.
        goal:              User goal (e.g. 'hypertrophy').
        current_week_data: Current week data (only for 'weekly' mode).
        prev_week_data:    Previous week data (only for 'weekly' mode, can be None).
        current_week_num:  Current week number 1-based (only for 'weekly' mode).
        prev_report:       Text of the previous analysis for this mode (Markdown).
                           Passed to the prompt so the AI can follow up on prior recommendations.

    Returns:
        String with the analysis in Markdown.
    """
    if mock:
        return _MOCK_OUTPUTS.get(mode, _MOCK_OUTPUTS["global"])

    if mode == "new-routine":
        prompt = build_new_routine_prompt(periods, goal)
    elif mode == "monthly":
        prompt = build_monthly_prompt(periods, goal)
    elif mode == "weekly":
        prompt = build_weekly_prompt(
            periods[0], current_week_data, prev_week_data, current_week_num, goal,
            prev_report=prev_report,
            prior_periods=periods[1:] if len(periods) > 1 else None,
        )
    else:
        prompt = build_global_prompt(periods, goal)

    client = Groq(api_key=api_key)
    system = _make_system_prompt(goal)
    print(f"  Sending [{mode}] prompt to Groq...", flush=True)
    return _call_groq(client, system, prompt)
