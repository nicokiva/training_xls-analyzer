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

_DEFAULT_SYSTEM = """You are a professional fitness coach and training data analyst.

How to interpret the data:
- "Rep." = repetitions PERFORMED by the user in that set (not the expected ones).
- "Peso" = weight used in kg. Sometimes contains text notes instead of or in addition to the number
  (e.g. "8 overhand / 2 underhand", "3 with 3kg / 5 bodyweight"). These notes are important
  user observations about how the set went — take them into account in the analysis.
- If "Peso" is "0" or empty, the exercise was done with bodyweight or no external load.
- Data is ordered: Week 1 → 2 → 3 → 4, with 3 sets per week.

Be concrete and reference real exercise names and numbers from the data.
Do not use generic phrases — every observation must be backed by specific data.

User goal: {goal}."""


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
    """
    used_model = model or MODEL
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
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
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

def _format_exercise_history_compact(periods):
    """
    Compact format for global analysis: one line per exercise showing peak weight
    per period (oldest → newest). Reduces tokens by ~80% vs the verbose format,
    which is enough for detecting long-term trends on the free Groq tier.

    Output example:
        **Sentadilla**: 09/25:80kg | 10/25:82kg | 11/25:82kg | 12/25:85kg → trend: +6%
    """
    ordered = list(reversed(periods))
    # exercise_name → list of (period_label, peak_weight_str)
    exercise_peaks = {}

    for period_data in ordered:
        period = period_data["period"]
        for day_data in period_data["days"]:
            for ex in day_data["exercises"]:
                name = ex["name"]
                # Collect all numeric weights across all weeks/sets
                weights = []
                for w in ex["weeks"]:
                    for s in w["series"]:
                        try:
                            weights.append(float(s["peso"]))
                        except (TypeError, ValueError):
                            pass
                if weights:
                    peak = max(weights)
                    if name not in exercise_peaks:
                        exercise_peaks[name] = []
                    exercise_peaks[name].append(f"{period[:5]}:{peak:.0f}kg")

    lines = []
    for name, entries in exercise_peaks.items():
        lines.append(f"**{name}**: {' | '.join(entries)}")
    return "\n".join(lines)


def _format_exercise_history(periods):
    """
    Verbose format: full set-by-set data per period.
    Used for monthly and new-routine modes where detail matters.
    """
    ordered = list(reversed(periods))
    exercise_history = {}

    for period_data in ordered:
        period = period_data["period"]
        for day_data in period_data["days"]:
            for ex in day_data["exercises"]:
                name = ex["name"]
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
                    if name not in exercise_history:
                        exercise_history[name] = []
                    exercise_history[name].append({
                        "period": period,
                        "data":   " | ".join(weeks_with_data),
                    })

    lines = []
    for name, history in exercise_history.items():
        lines.append(f"**{name}**")
        for entry in history:
            lines.append(f"  {entry['period']}: {entry['data']}")
        lines.append("")
    return "\n".join(lines)


def _format_routine_structure(period):
    """
    Formats the exercise structure of a period without execution data.
    Useful for new-routine where the tab doesn't have reps/weights loaded yet.
    """
    lines = []
    for day in period["days"]:
        lines.append(f"Day {day['day']}:")
        for ex in day["exercises"]:
            lines.append(f"  - {ex['name']}")
        lines.append("")
    return "\n".join(lines)


def _format_week_data(week_data, week_label):
    """Formats the data for a week (output of extract_week_data)."""
    lines = [f"**{week_label}**"]
    for day in week_data:
        lines.append(f"  Day {day['day']}:")
        for ex in day["exercises"]:
            series_str = "  ".join(
                f"S{i+1}:{s['reps'] or '-'}r/{s['peso'] or '-'}kg"
                for i, s in enumerate(ex["series"])
                if s["reps"] or s["peso"]
            )
            lines.append(f"    {ex['name']}: {series_str}")
    lines.append("")
    return "\n".join(lines)


def build_global_prompt(periods, goal):
    """
    Prompt for global mode: full history using compact format (peak weight per period).
    Uses all periods — compact format keeps it under 12k TPM even with 11 periods.
    Loads from templates/global.txt if available, otherwise uses hardcoded default.
    """
    history_block = _format_exercise_history_compact(periods)
    result = _load_template("global", goal=goal, history=history_block)
    if result:
        return result
    return (
        f"Analyze the complete progression of the following exercises over time.\n"
        f"Data is ordered chronologically (oldest → most recent).\n\n"
        f"Generate a global analysis oriented toward the **{goal}** goal that includes:\n"
        f"- General trend for each exercise (improvement, plateau, regression)\n"
        f"- Exercises with the most and least progress\n"
        f"- Evaluation of whether the history is on track to meet the {goal} goal\n"
        f"- Plateau signals with concrete evidence\n"
        f"- Concrete recommendations for the next cycles\n\n"
        f"{history_block}"
    )


def build_new_routine_prompt(periods, goal):
    """
    Prompt for new-routine mode: evaluates the newly generated routine.
    Limits history context to 3 previous periods to stay within Groq TPM limits.
    Loads from templates/new-routine.txt if available.
    """
    new_period    = periods[0]
    # new-routine only needs the routine structure — no history to stay under TPM limits.
    # The AI evaluates exercises against the goal based on the routine itself.
    routine_block = _format_routine_structure(new_period)
    history_block = "(history omitted to stay within token limits)"

    result = _load_template("new-routine", goal=goal, period=new_period["period"],
                            routine=routine_block, history=history_block)
    if result:
        return result
    return (
        f"A new training routine has just been generated.\n"
        f"The user's goal is: **{goal}**.\n\n"
        f"## New routine ({new_period['period']})\n\n"
        f"{routine_block}\n"
        f"## Previous training history\n\n"
        f"{history_block}\n"
        f"Analyze:\n"
        f"1. Is this routine suitable for the {goal} goal? Why?\n"
        f"2. Are there exercises that don't contribute to the goal or could be improved?\n"
        f"3. What concrete changes would you make to this routine given the user's history?\n"
        f"4. Are there patterns from the history that this routine leverages well or ignores?\n"
    )


def build_monthly_prompt(periods, goal):
    """
    Prompt for monthly mode: balance of the most recent month.
    Limits history context to 3 previous periods to stay within Groq TPM limits.
    Loads from templates/monthly.txt if available.
    """
    current_period = periods[0]
    # Monthly mode only needs the current month — no history context to stay under TPM limits
    history        = []

    current_block = _format_exercise_history([current_period])
    history_block = _format_exercise_history(history) if history else "(no previous history)"

    result = _load_template("monthly", goal=goal, period=current_period["period"],
                            current_block=current_block, history=history_block)
    if result:
        return result
    return (
        f"Provide a balance of the training month **{current_period['period']}**.\n"
        f"The user's goal is: **{goal}**.\n\n"
        f"## Month data\n\n"
        f"{current_block}\n"
        f"## Previous history (context)\n\n"
        f"{history_block}\n"
        f"Analyze:\n"
        f"1. Was the {goal} goal met this month? What evidence is there in the numbers?\n"
        f"2. Which exercises progressed well? Which plateaued or regressed?\n"
        f"3. How was the consistency and volume compared to previous months?\n"
        f"4. What adjustments do you recommend for next month?\n"
    )


def build_weekly_prompt(period, current_week_data, prev_week_data, current_week_num, goal):
    """
    Prompt for weekly mode: compares the current week with the previous one.
    Loads from templates/weekly.txt (with prev week) or templates/weekly_first.txt (no prev).
    """
    current_block = _format_week_data(current_week_data, f"Week {current_week_num} (current)")

    if prev_week_data:
        prev_block = _format_week_data(prev_week_data, f"Week {current_week_num - 1} (previous)")
        result = _load_template("weekly", goal=goal, period=period["period"],
                                current_week=current_block, prev_week=prev_block,
                                week_num=current_week_num)
        if result:
            return result
        return (
            f"Weekly analysis of period **{period['period']}**.\n"
            f"User goal: **{goal}**.\n\n"
            f"## Previous week\n\n{prev_block}\n"
            f"## Current week\n\n{current_block}\n"
            f"Comparing with the previous week, analyze:\n"
            f"1. Did overall performance improve this week?\n"
            f"2. Which exercises improved (more weight or more reps)? Which declined?\n"
            f"3. Was it a good week for the {goal} goal?\n"
            f"4. What adjustments do you recommend for next week?\n"
        )
    else:
        result = _load_template("weekly_first", goal=goal, period=period["period"],
                                current_week=current_block, week_num=current_week_num)
        if result:
            return result
        return (
            f"Weekly analysis of period **{period['period']}**.\n"
            f"User goal: **{goal}**.\n\n"
            f"## Current week (first of the period)\n\n{current_block}\n"
            f"This is the first week of the period, there is no previous week to compare.\n"
            f"Analyze:\n"
            f"1. How did the period start in relation to the {goal} goal?\n"
            f"2. Is there anything notable in the numbers of this first week?\n"
            f"3. What do you recommend for next week?\n"
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
        "You are a professional translator. Translate the following text to Spanish. "
        "Keep markdown formatting intact. Do not add any commentary — only return the translated text."
    )
    return _call_groq(client, system, text, max_tokens=4096)


def analyze(periods, api_key, mock=False, mode="global", goal="hipertrofia",
            current_week_data=None, prev_week_data=None, current_week_num=None):
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
            periods[0], current_week_data, prev_week_data, current_week_num, goal
        )
    else:
        prompt = build_global_prompt(periods, goal)

    client = Groq(api_key=api_key)
    system = _make_system_prompt(goal)
    print(f"  Sending [{mode}] prompt to Groq...", flush=True)
    # Global uses the compact format so it fits in 12k TPM — no special model needed
    used_model = None
    return _call_groq(client, system, prompt, model=used_model)
