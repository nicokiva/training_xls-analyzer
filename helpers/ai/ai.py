"""
helpers/ai.py — Gemini integration for training progression analysis.

Responsibilities:
  - Transform period data into mode-specific prompts.
  - Call Gemini via the google-genai SDK.
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
from typing import List

from google import genai
from google.genai import types

from .models import WeightSuggestion, WeightSuggestionList

MODEL            = "gemini-2.5-flash"       # prose analysis
MODEL_STRUCTURED = "gemini-2.5-flash"       # structured weight suggestions

# Templates are looked up relative to the project root (two levels up from this file).
# Path(__file__) is the absolute path of this file.
# .parent.parent.parent navigates: ai/ → helpers/ → routine-analyzer/
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"

# ---------------------------------------------------------------------------
# Hardcoded fallback prompts (used when the template file is missing)
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM = """You are Nicolás' personal trainer. You have been training him for months and know his history, his progress, and his tendencies. You write directly to him, in first person, as if you were messaging him after reviewing his training data.

**Always respond in Spanish (Argentina). Never mix languages. Use local gym slang naturally (ej: "ir al fallo", "series efectivas", "cargar la barra", "parate", "tirón") without being forced.**

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

Structure your response as a direct message:
1. A brief, punchy greeting evaluating the overall week/day.
2. Specific bullet points highlighting key exercises: where there was solid progressive overload (more weight) and where there was a plateau or drop that needs attention (adjusting weight or rest). Always mention the exact numbers.
3. A concrete takeaway/instruction for the next time he faces this specific day.

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



def _call_gemini(client, system_prompt, user_prompt, max_tokens=4096, model=None, thinking_budget=None):
    """
    Makes a call to Gemini and returns the response text.
    Retries once after 65 seconds if the per-minute rate limit is hit.
    Logs every request and response to logs/gemini_YYYYMMDD.log.

    thinking_budget: if set, controls how many tokens Gemini can use for
    internal reasoning (0 = disable thinking). gemini-2.5-flash uses thinking
    by default and those tokens count against max_output_tokens, so prose-only
    calls should pass thinking_budget=0 to avoid the budget being consumed by
    reasoning instead of actual output.
    """
    used_model = model or MODEL

    # ── Logging ──────────────────────────────────────────────────────────────
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"gemini_{datetime.now().strftime('%Y%m%d')}.log"

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

    config_kwargs = dict(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
    )
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=used_model,
                config=types.GenerateContentConfig(**config_kwargs),
                contents=user_prompt,
            )
            finish_reason = None
            try:
                finish_reason = response.candidates[0].finish_reason
            except Exception:
                pass
            result = response.text
            _log(f"--- RESPONSE ({len(result)} chars, finish_reason={finish_reason}) ---")
            _log(result)
            return result
        except Exception as e:
            err = str(e)
            _log(f"--- ERROR (attempt {attempt+1}) ---\n{err}")
            if ("429" in err or "RESOURCE_EXHAUSTED" in err or "503" in err or "UNAVAILABLE" in err) and attempt == 0:
                print("  Temporary error — waiting 65s and retrying...")
                time.sleep(65)
                continue
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


def _normalize_ex_name(name: str) -> str:
    """Lowercase + strip accents for case-insensitive exercise matching."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", name.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Exercise name alias map — canonical name → all known historical variants
# Canonical = name used in the current/future routines (normalised).
# All names are stored normalised (lowercase, accent-stripped) at build time.
# ---------------------------------------------------------------------------
_EXERCISE_ALIASES_RAW = {
    # ── Core ─────────────────────────────────────────────────────────────────
    "Abdominal recto largo (manos atras de la nuca)": [
        "Abdominal recto largo",
        "Abdominal recto largo en banco",
        "Abdominal recto largo en banco isométrico",
    ],
    # ── Chest push ───────────────────────────────────────────────────────────
    # Both "(banco plano)" and "en banco plano" naming conventions have been
    # used across different routine generations — canonicalise to parenthesised form.
    "Press de pecho con barra (banco plano)": [
        "Press de pecho con barra en banco plano",
    ],
    "Press de pecho con barra (banco inclinado)": [
        "Press de pecho con barra en banco inclinado",
    ],
    # Dumbbell chest press — "Press plano con mancuernas" was the historical name
    # before the current "(banco plano)" convention was adopted.
    "Press de pecho con mancuernas (banco plano)": [
        "Press plano con mancuernas",
        "Press plano con mancuerna (alternado + simétrico)",
        "Press de pecho con mancuernas en banco plano",
    ],
    "Press de pecho con mancuernas (banco inclinado)": [
        "Press inclinado con mancuernas",
        "Press de pecho con mancuernas en banco inclinado",
    ],
    "Empuje de pecho con barra en banco plano": [
        "Press plano con barra",
        "Press plano con barra (toma cerrada)",
    ],
    "Empuje de pecho con barra en banco inclinado": [
        "Press inclinado con barra",
        "Press inclinado con barra",
    ],
    "Peck Deck (pecho)": [
        "Peck Deck",
    ],
    "Empuje de pecho en Hammer": [
        "Empuje de pecho en hammer",
        "Pecho en hammer",
        "Press en hammer",
    ],
    # ── Triceps ──────────────────────────────────────────────────────────────
    "Triceps con polea (agarre prono)": [
        "Tríceps con polea (prono)",
        "Tríceps polea (prono)",
        "Triceps prono con polea",
        "Tríceps con polea (agarre prono)",
    ],
    # ── Shoulders push ───────────────────────────────────────────────────────
    "Empuje de hombros con barra (sentado)": [
        "Press militar parado con barra",
        "Press de hombros con barra sentado en smith",
        "Empuje de hombros con barra sentado",
        "Empuje de hombros con barra parado",
        "Empuje de hombros con barra sentado",
    ],
    "Empuje de hombros con mancuernas": [
        "Press de hombros con mancuernas",
        "Empuje de hombros con mancuernas sentado",
        "Empuje de hombros en mancuerna",
        "Empuje de hombros con mancuerna",
    ],
    "Vuelos laterales con mancuernas": [
        "Vuelos laterales con mancuerna",
        "Vuelos laterales con mancuernas (sentado)",
        "Vuelo lateral con mancuerna",
        "Vuelos laterales con mancuerna sentado",
        # NOTE: "Vuelos laterales en polea" intentionally excluded —
        # cable laterals use different absolute loads than dumbbell laterals.
    ],
    # ── Legs ─────────────────────────────────────────────────────────────────
    "Sentadilla clásica": [
        "Sentadilla",
        "Sentadilla clásica profunda",
        "Sentadilla clásica media",
    ],
    "Peso muerto": [
        "Peso Muerto",
    ],
    "Flexión de rodillas en máquina": [
        "Flexión de rodillas acostado",
        "Flexión acostado",
        "Flexión rodillas acostado",
        "Flexión de rodillas",
    ],
    "Estocada en banco (cada pierna)": [
        "Estocada en banco",
        "Estocada al frente",
        "Estocada en multipower (por pierna)",
    ],
    "Prensa Hammer 45°": [
        "Prensa hammer 45°",
        "Prensa 45°",
        "Prensa 45",
        "Empuje de piernas (prensa sentado)",
    ],
    "Extensión de rodillas en máquina": [
        "Extensión de rodillas",
        "Extensión rodillas en máquina",
    ],
    # ── Pull / Back ──────────────────────────────────────────────────────────
    "Dominada estricta": [
        "Dominada",
    ],
    "Tirón dorsal en polea con agarre supino": [
        "Tirón dorsal con agarre supino",
        "Tirón dorsal con agarre prono",
        "Tirón dorsal en polea agarre neutro",
        "Tirón dorsal en polea agarre prono",
        "Tirón dorsal en poleta con agarre prono",
        "Tirón dorsal con agarre prono",
        "Tirón dorsal en polea con agarre prono",
    ],
    "Biceps con polea": [
        "Bíceps polea",
        "Biceps martillo",
        "Bíceps martillo",
    ],
    "Remo al mentón": [
        "Remo al mentón con polea",
    ],
}

def _build_alias_reverse() -> dict:
    """Returns {normalized_variant: normalized_canonical} reverse lookup."""
    reverse = {}
    for canonical, variants in _EXERCISE_ALIASES_RAW.items():
        norm_canonical = _normalize_ex_name(canonical)
        for v in variants:
            reverse[_normalize_ex_name(v)] = norm_canonical
    return reverse

# Built once at import time
_ALIAS_REVERSE: dict = {}  # populated lazily on first use

def _canonical_key(name: str) -> str:
    """
    Returns the normalised canonical key for an exercise name.
    If the normalised name is a known alias variant, returns the canonical form.
    Otherwise returns the normalised name unchanged.
    """
    global _ALIAS_REVERSE
    if not _ALIAS_REVERSE:
        _ALIAS_REVERSE = _build_alias_reverse()
    norm = _normalize_ex_name(name)
    return _ALIAS_REVERSE.get(norm, norm)


# ---------------------------------------------------------------------------
# Movement pattern / biomechanical archetype system
# ---------------------------------------------------------------------------
# Each archetype is a (name, [regex patterns]) tuple.
# Order matters: more specific patterns first.
_ARCHETYPES = [
    # Horizontal push — chest
    ("horizontal_push_flat",    [r"empuje.*pecho.*plano", r"press.*plano", r"press.*bench"]),
    ("horizontal_push_incline", [r"empuje.*pecho.*inclinado", r"press.*inclinado"]),
    ("chest_fly",               [r"peck.?deck", r"apertura.*pecho", r"cruce.*polea.*pecho"]),
    ("horizontal_push_machine", [r"empuje.*pecho.*hammer", r"hammer.*pecho", r"press.*pecho.*maquina"]),
    # Vertical push — shoulders
    ("vertical_push",           [r"empuje.*hombros", r"press.*hombros", r"press.*militar", r"overhead.*press"]),
    # Lateral deltoid
    ("lateral_delt",            [r"vuelo.*lateral", r"apertura.*lateral", r"lateral.*raise", r"remo.*menton"]),
    ("rear_delt",               [r"vuelo.*posterior", r"apertura.*posterior", r"face.?pull", r"pajarito"]),
    # Vertical pull — back
    ("vertical_pull",           [r"dominada", r"tiron.*dorsal", r"lat.*pulldown", r"polea.*alta"]),
    # Horizontal pull — back
    ("horizontal_pull",         [r"remo.*polea", r"remo.*bajo", r"remo.*cable", r"depresor"]),
    ("row_barbell",             [r"remo.*barra", r"remo.*inclinado"]),
    # Knee dominant — quads
    ("squat",                   [r"sentadilla"]),
    ("leg_press",               [r"prensa.*hammer", r"prensa.*45", r"leg.*press", r"prensa.*pierna"]),
    ("quad_extension",          [r"extension.*rodilla"]),
    ("lunge",                   [r"estocada", r"zancada", r"split.*squat", r"bulgarian"]),
    # Hip dominant — posterior
    ("deadlift",                [r"peso.*muerto"]),
    ("leg_curl",                [r"flexion.*rodilla", r"curl.*femoral", r"isquio"]),
    ("hip_extension",           [r"extension.*cadera", r"hip.*thrust", r"patada.*glute"]),
    # Arms
    ("triceps",                 [r"tricep"]),
    ("biceps",                  [r"bicep", r"curl"]),
    # Core
    ("core_flexion",            [r"abdominal", r"crunch"]),
    ("core_rotation",           [r"rotacion.*disco", r"giro"]),
]

# Equipment categories for transfer factor calculation
_EQUIPMENT_PATTERNS = [
    ("barbell",    [r"\bbarra\b", r"con barra", r"barra libre"]),
    ("dumbbell",   [r"mancuerna"]),
    ("machine",    [r"hammer", r"maquina", r"polea", r"en banco", r"peck.?deck",
                    r"extension.*rodilla", r"flexion.*rodilla", r"prensa"]),
    ("cable",      [r"polea", r"cable"]),
    ("bodyweight", [r"dominada", r"estocada", r"fondo", r"plancha"]),
]

# Transfer factors: (src_equipment, dst_equipment) → factor to apply to src weight
_TRANSFER_FACTORS = {
    ("barbell",    "barbell"):    1.00,
    ("barbell",    "dumbbell"):   0.90,
    ("barbell",    "machine"):    0.90,
    ("dumbbell",   "barbell"):    1.10,
    ("dumbbell",   "dumbbell"):   1.00,
    ("dumbbell",   "machine"):    0.95,
    ("machine",    "machine"):    1.00,
    ("machine",    "barbell"):    1.10,
    ("machine",    "dumbbell"):   1.05,
    ("cable",      "cable"):      1.00,
    ("cable",      "machine"):    1.00,
    ("machine",    "cable"):      1.00,
}


def _get_archetype(name: str) -> "str | None":
    """Returns the archetype name for an exercise, or None if unclassified."""
    import re
    n = _normalize_ex_name(name)
    for archetype, patterns in _ARCHETYPES:
        for pat in patterns:
            if re.search(pat, n):
                return archetype
    return None


def _get_equipment(name: str) -> str:
    """Returns the equipment type for an exercise."""
    import re
    n = _normalize_ex_name(name)
    for equip, patterns in _EQUIPMENT_PATTERNS:
        for pat in patterns:
            if re.search(pat, n):
                return equip
    return "machine"  # default fallback


def _transfer_factor(src_name: str, dst_name: str) -> float:
    """Transfer factor to apply to src_weight when dst exercise is different equipment."""
    src_eq = _get_equipment(src_name)
    dst_eq = _get_equipment(dst_name)
    return _TRANSFER_FACTORS.get((src_eq, dst_eq), 0.90)


def _build_history_lookup(prior_periods: list) -> tuple:
    """
    Builds the name-keyed history lookup structures used by all settled-weight
    computation functions.

    Returns:
        lookup:    {(norm_name, is_comb): [(period_label, settled, n_weeks, is_comb), ...]}
                   Ordered most-recent first.
        flat_best: {(original_name, is_comb): {name, settled_peso, n_weeks, period, is_comb}}
                   Keeps only the entry with the most weeks of data per exercise.
    """
    import re

    def strip_parens(s):
        return re.sub(r"\s*\(.*?\)", "", s).strip()

    lookup: dict = {}
    for period_data in prior_periods:
        period_label = period_data["period"][:8]
        for day_data in period_data["days"]:
            for ex in day_data["exercises"]:
                n_weeks, settled = _last_settled_weeks(ex)
                if settled is None:
                    continue
                raw_key = _normalize_ex_name(strip_parens(ex["name"]))
                canon_key = _canonical_key(ex["name"])
                is_c = ex.get("is_comb", False)
                entry = (period_label, settled, n_weeks, is_c)
                lookup.setdefault((raw_key, is_c), []).append(entry)
                if canon_key != raw_key:
                    lookup.setdefault((canon_key, is_c), []).append(entry)

    flat_best: dict = {}
    for period_data in prior_periods:
        for day_data in period_data["days"]:
            for ex in day_data["exercises"]:
                n_weeks, settled = _last_settled_weeks(ex)
                if settled is None:
                    continue
                key = (ex["name"], ex.get("is_comb", False))
                prev = flat_best.get(key)
                if prev is None or n_weeks > prev["n_weeks"]:
                    flat_best[key] = {
                        "name": ex["name"],
                        "settled_peso": settled,
                        "n_weeks": n_weeks,
                        "period": period_data["period"][:8],
                        "is_comb": ex.get("is_comb", False),
                    }

    return lookup, flat_best


def _resolve_exercise_weight(
    ex: dict,
    new_is_c: bool,
    lookup: dict,
    flat_best: dict,
    shared_catalog: list,
) -> str:
    """
    Resolves the best available weight reference for a single exercise dict.

    Fallback chain (same as _compute_settled_weights):
      1. Exact canonical/alias match
      2. Partial name match (prefix)
      3. Catalog-based biomechanical match
      4. Regex archetype match
      5. Unknown (no history)

    Returns a compact weight-resolution string, e.g.:
      '20/04/26 → 55.0 kg (combinado) [4 sem]'
      'sin hist. exacto → biomec. equiv. "Press barra plano" (score 6/6) → 52.5 kg [4 sem]'
    """
    import re

    def strip_parens(s):
        return re.sub(r"\s*\(.*?\)", "", s).strip()

    raw_key = _normalize_ex_name(strip_parens(ex["name"]))
    canon_key = _canonical_key(ex["name"])

    entries = lookup.get((canon_key, new_is_c)) or lookup.get((raw_key, new_is_c))
    isolated_fallback = False
    if not entries and new_is_c:
        entries = lookup.get((canon_key, False)) or lookup.get((raw_key, False))
        if entries:
            isolated_fallback = True

    def _format_entry(entries, isolated_fallback=False):
        p_label, settled, n_weeks, is_c = entries[0]
        comb_note = " (combinado)" if is_c else ""
        weeks_note = f" [{n_weeks} sem]"
        if isolated_fallback:
            line = (
                f"sin hist. combinado → ref. aislada {p_label} → {settled:.1f} kg{weeks_note} "
                f"(~90% como punto de partida)"
            )
        else:
            line = f"{p_label} → {settled:.1f} kg{comb_note}{weeks_note}"
        if n_weeks < 3 and len(entries) > 1:
            best = max(entries[1:], key=lambda e: e[2])
            if best[2] > n_weeks:
                b_label, b_settled, b_weeks, b_comb = best
                b_note = " (combinado)" if b_comb else ""
                line += f" | más completo: {b_label} → {b_settled:.1f} kg [{b_weeks} sem]{b_note}"
        return line

    if entries:
        return _format_entry(entries, isolated_fallback)

    key = canon_key
    for (hist_key, hist_is_c), hist_entries in lookup.items():
        if hist_is_c == new_is_c and (key.startswith(hist_key) or hist_key.startswith(key)):
            return _format_entry(hist_entries)

    flat_history = [v for v in flat_best.values() if v["is_comb"] == new_is_c]
    from helpers.catalog.matcher import find_closest_exercise
    cat_match, cat_score = find_closest_exercise(ex["name"], flat_history, shared_catalog)
    if cat_match is not None and cat_score >= 0:
        factor = _transfer_factor(cat_match["name"], ex["name"])
        adjusted = round(cat_match["settled_peso"] * factor / 2.5) * 2.5
        return (
            f"sin hist. exacto → biomec. equiv. '{cat_match['name']}' "
            f"(score {cat_score}/6, {cat_match['period']}) → {adjusted:.1f} kg "
            f"[{cat_match['n_weeks']} sem]"
        )

    archetype = _get_archetype(ex["name"])
    arch_match = None
    if archetype:
        for (hist_key, hist_is_c), hist_entries in lookup.items():
            if hist_is_c != new_is_c:
                continue
            if _get_archetype(hist_key) == archetype:
                p_label, settled, n_weeks, _ = hist_entries[0]
                factor = _transfer_factor(hist_key, ex["name"])
                adjusted = round(settled * factor / 2.5) * 2.5
                if arch_match is None or n_weeks > arch_match[3]:
                    arch_match = (p_label, hist_key, adjusted, n_weeks, factor)
    if arch_match:
        p_label, src_name, adjusted, n_weeks, _ = arch_match
        return (
            f"sin hist. exacto → patrón equiv. '{src_name}' ({p_label}) "
            f"→ {adjusted:.1f} kg [{n_weeks} sem]"
        )

    return "sin historial previo (ejercicio nuevo)"


def _compute_settled_weights(new_period: dict, prior_periods: list) -> str:
    """
    For each exercise in new_period, find its most recent settled weight from
    prior_periods using normalized name matching, with biomechanical fallback
    when no exact name match exists.

    Fallback chain (in order):
      1. Exact canonical/alias name match
      2. Partial name match (prefix — e.g. "Dominada" matches "Dominada estricta")
      3. Catalog-based biomechanical match (shared catalog, patron_principal gate +
         mecanica/vector/estabilizacion scoring — see helpers/catalog/matcher.py)
      4. Regex archetype match (local _ARCHETYPES patterns — catches cases not in
         the shared catalog)

    Shows the most recent entry AND (if it has fewer than 3 weeks of data) also
    the most complete entry so the AI has reliable baselines when a recent period
    is sparse.

    Returns a formatted block to inject into the prompt so the AI doesn't need
    to parse exercise names from raw history text.
    """
    from helpers.catalog.matcher import load_shared_catalog

    lookup, flat_best = _build_history_lookup(prior_periods)
    shared_catalog = load_shared_catalog()

    lines = ["Pesos de cierre por ejercicio (calculados por el script — usar como baseline):"]
    for day_data in new_period["days"]:
        day_num = day_data["day"]
        n_in_day = len(day_data["exercises"])
        for pos, ex in enumerate(day_data["exercises"], start=1):
            new_is_c = ex.get("is_comb", False)
            pos_note = f" [#{pos}/{n_in_day} del día]"
            weight_line = _resolve_exercise_weight(ex, new_is_c, lookup, flat_best, shared_catalog)
            lines.append(f"  D{day_num} {ex['name']}{pos_note}: {weight_line}")
    return "\n".join(lines)


def _build_global_index(new_period: dict) -> dict:
    """
    Builds a {pos: name} index for all UNIQUE exercises across all days (1-indexed).

    Exercises are numbered in order of first appearance (Day 1 first, then Day 2, etc.).
    Repeated exercises — e.g. the abdomen triset that appears in every day — receive a
    single entry and the same #N key is reused wherever that exercise appears.

    This is the single source of truth for #N references in the weight-suggestions prompt.
    """
    index: dict = {}
    name_to_pos: dict = {}
    pos = 1
    for day_data in new_period["days"]:
        for ex in day_data["exercises"]:
            name = ex["name"]
            if name not in name_to_pos:
                index[pos] = name
                name_to_pos[name] = pos
                pos += 1
    return index


def _render_global_index_block(index: dict) -> str:
    """Renders the global exercise index as a numbered list."""
    lines = ["Exercise index:"]
    for pos in sorted(index):
        lines.append(f"  #{pos:<3} {index[pos]}")
    return "\n".join(lines)


def _render_period_structure_indexed(new_period: dict, index: dict) -> str:
    """
    Renders all days' exercise structure using global #N keys.
    Exercise names never appear in this block — the index provides the mapping.
    Repeated exercises (e.g. abdomen in every day) use the same key each time.
    """
    name_to_pos = {name: pos for pos, name in index.items()}
    lines = []
    for day_data in new_period["days"]:
        lines.append(f"Day {day_data['day']}:")
        groups = _group_combined_exercises(day_data["exercises"])
        for group in groups:
            if group["comb"]:
                keys = [f"#{name_to_pos[ex['name']]}" for ex in group["exercises"]]
                lines.append(f"  --- Superset: {' + '.join(keys)} ---")
                for i, ex in enumerate(group["exercises"]):
                    suffix = _reps_suffix_for_ex(ex)
                    rest   = "  ← REST HERE" if i == len(group["exercises"]) - 1 else ""
                    lines.append(f"    #{name_to_pos[ex['name']]}{suffix}{rest}")
            else:
                ex = group["exercises"][0]
                lines.append(f"  #{name_to_pos[ex['name']]}{_reps_suffix_for_ex(ex)}")
        lines.append("")
    return "\n".join(lines)


def _render_global_settled_weights_indexed(
    new_period: dict,
    prior_periods: list,
    index: dict,
) -> str:
    """
    Renders settled weights for all unique exercises using global #N keys.

    Each unique exercise appears exactly ONCE regardless of how many days it
    appears in — repeats (e.g. abdomen) share a single weight entry.
    """
    from helpers.catalog.matcher import load_shared_catalog

    lookup, flat_best = _build_history_lookup(prior_periods)
    shared_catalog    = load_shared_catalog()

    # Collect first occurrence of each unique exercise (preserves is_comb context)
    first_occurrence: dict = {}
    for day_data in new_period["days"]:
        for ex in day_data["exercises"]:
            if ex["name"] not in first_occurrence:
                first_occurrence[ex["name"]] = ex

    lines = ["Baseline weights per exercise:"]
    for pos in sorted(index):
        name    = index[pos]
        ex      = first_occurrence[name]
        new_is_c = ex.get("is_comb", False)
        weight_line = _resolve_exercise_weight(ex, new_is_c, lookup, flat_best, shared_catalog)
        lines.append(f"  #{pos}: {weight_line}")
    return "\n".join(lines)

def get_settled_weights_dict(new_period: dict, prior_periods: list) -> dict:
    """
    Returns a dict of normalised canonical exercise name → settled peso (float)
    for use in post-processing validation of AI suggestions.

    Iterates from most recent to oldest period. Keeps the most recent entry with
    ≥2 weeks of data; falls back to any entry if none has ≥2 weeks. Canonical
    alias keys are used so all name variants collapse to the same baseline.
    """
    import re

    def strip_parens(s):
        return re.sub(r"\s*\(.*?\)", "", s).strip()

    # lookup: (canonical_key, is_comb) → (n_weeks, settled, period_index)
    # Keying by is_comb ensures isolated and combined weights are never cross-compared.
    # period_index 0 = most recent prior period
    lookup: dict = {}
    for period_idx, period_data in enumerate(prior_periods):
        for day_data in period_data["days"]:
            for ex in day_data["exercises"]:
                n_weeks, settled = _last_settled_weeks(ex)
                if settled is None:
                    continue
                canon_key = _canonical_key(ex["name"])
                is_c = ex.get("is_comb", False)
                lkey = (canon_key, is_c)
                existing = lookup.get(lkey)
                if existing is None:
                    lookup[lkey] = (n_weeks, settled, period_idx)
                else:
                    prev_weeks, prev_settled, prev_idx = existing
                    if period_idx < prev_idx and n_weeks >= 2:
                        lookup[lkey] = (n_weeks, settled, period_idx)
                    elif n_weeks > prev_weeks and prev_idx == period_idx:
                        lookup[lkey] = (n_weeks, settled, period_idx)

    # Resolve each new exercise using its own is_comb flag
    result = {}
    for ex_data in (ex for d in new_period["days"] for ex in d["exercises"]):
        canon_key = _canonical_key(ex_data["name"])
        is_c = ex_data.get("is_comb", False)
        entry = lookup.get((canon_key, is_c))
        if entry:
            result[canon_key] = entry[1]
    return result


def _format_exercise_history_compact(periods):
    """
    Compact format: one line per exercise showing the settled weight per period
    (oldest → newest). Uses the last week's average — not the peak — so that
    'discovery' week overshoots don't inflate the baseline for the next period.

    When the same exercise appears multiple times within a period (e.g. abdomen
    repeats across all 4 days), only the RICHEST occurrence is kept — the one
    with the most weeks of data, breaking ties by highest settled weight. This
    avoids confusing the AI with contradictory entries like "D1:12kg | D3:5kg".

    Exercises are keyed by (name, is_comb) so a press done in a triserie and the
    same press done in isolation are treated as separate series — their weights are
    not directly comparable and must never be merged onto the same history line.
    """
    ordered = list(reversed(periods))
    # best_per_period[(period, canonical_name, is_comb)] = (num_weeks, settled, ex_note, set_notes)
    best_per_period = {}

    for period_data in ordered:
        period = period_data["period"]
        for day_data in period_data["days"]:
            for ex in day_data["exercises"]:
                canon   = _canonical_key(ex["name"])
                is_comb = ex.get("is_comb", False)
                n_weeks, settled = _last_settled_weeks(ex)
                if settled is None:
                    continue
                ex_note   = ex.get("note", "")
                set_notes = [
                    s["note"]
                    for w in ex["weeks"]
                    for s in w["series"]
                    if s.get("note")
                ]
                drop_sets = [
                    s["drop_set"]
                    for w in ex["weeks"]
                    for s in w["series"]
                    if s.get("drop_set")
                ]
                key = (period, canon, is_comb)
                prev = best_per_period.get(key)
                prev_n = prev[0] if prev else 0
                prev_s = prev[1] if prev else 0
                if n_weeks > prev_n or (n_weeks == prev_n and settled > prev_s):
                    best_per_period[key] = (n_weeks, settled, ex_note, set_notes, drop_sets)

    # Rebuild ordered entries: (canonical_name, is_comb) → list of (period, settled, ex_note, set_notes, drop_sets)
    exercise_entries = {}
    for (period, canon, is_comb), (_, settled, ex_note, set_notes, drop_sets) in best_per_period.items():
        exercise_entries.setdefault((canon, is_comb), []).append(
            (period, settled, ex_note, set_notes, drop_sets)
        )

    lines = []
    for (canon, is_comb), period_entries in exercise_entries.items():
        label = f"**{canon}**" + (" *(combinado)*" if is_comb else "")
        parts = [f"{p[:5]}:{s:.1f}kg" for p, s, _, __, ___ in period_entries]
        line = f"{label}: {' | '.join(parts)}"
        # Append any notes collected for this exercise across periods
        note_parts = []
        for p, _, ex_note, set_notes, drop_sets in period_entries:
            tag = p[:5]
            if ex_note:
                note_parts.append(f"[Note {tag}: \"{ex_note}\"]")
            for sn in set_notes:
                note_parts.append(f"[Note {tag}: \"{sn}\"]")
            for ds in drop_sets:
                note_parts.append(f"[Drop set {tag}: {ds}]")
        if note_parts:
            line += "  " + "  ".join(note_parts)
        lines.append(line)
    return "\n".join(lines)


def _format_exercise_history(periods):
    """
    Verbose format: full set-by-set data per period.
    Exercises are keyed by (name, is_comb) so a combined press and an isolated
    press are shown as separate entries — their weights are not comparable.
    """
    ordered = list(reversed(periods))
    # exercise_history[(name, is_comb)] = [{"period": ..., "data": ..., "note": ...}, ...]
    exercise_history = {}

    for period_data in ordered:
        period = period_data["period"]
        for day_data in period_data["days"]:
            for ex in day_data["exercises"]:
                name    = ex["name"]
                is_comb = ex.get("is_comb", False)
                ex_note = ex.get("note", "")
                weeks_with_data = []
                for w in ex["weeks"]:
                    series_parts = []
                    for idx, s in enumerate(w["series"]):
                        if s["reps"] or s["peso"]:
                            reps     = s["reps"] or "-"
                            peso     = s["peso"] or "-"
                            set_note = f' ("{s["note"]}")' if s.get("note") else ""
                            series_parts.append(f"S{idx+1}:{reps}r/{peso}kg{set_note}")
                    if series_parts:
                        weeks_with_data.append(f"Wk{w['week']}: {' '.join(series_parts)}")

                if weeks_with_data:
                    key = (name, is_comb)
                    if key not in exercise_history:
                        exercise_history[key] = []
                    exercise_history[key].append({
                        "period": period,
                        "data":   " | ".join(weeks_with_data),
                        "note":   ex_note,
                    })

    lines = []
    for (name, is_comb), history in exercise_history.items():
        label = f"**{name}**" + (" *(combinado)*" if is_comb else "")
        lines.append(label)
        for entry in history:
            note_suffix = f'  [Note: "{entry["note"]}"]' if entry["note"] else ""
            lines.append(f"  {entry['period']}: {entry['data']}{note_suffix}")
        lines.append("")
    return "\n".join(lines)


def _group_combined_exercises(exercises):
    """
    Partition a list of exercises into a sequence of groups.

    A group is either:
      - A single exercise with is_comb=False  →  {"comb": False, "exercises": [ex]}
      - A set of exercises with is_comb=True that share the same combined group
                                              →  {"comb": True,  "exercises": [ex, ...]}

    Two strategies depending on available data:

    1. **comb_group IDs present** (from Z-column merges read by the reader):
       Exercises with the same ``comb_group`` value form one group, regardless of
       whether they are contiguous with other combined exercises. This correctly
       handles back-to-back groups (e.g., abdomen triset #1-#4 followed immediately
       by a chest superset #5-#6 — both have is_comb=True with no gap between them).

    2. **Fallback — no comb_group IDs** (old tabs or tabs without merge data):
       Groups consecutive is_comb=True exercises. This may merge adjacent groups
       into one when they are contiguous, but it is the best approximation without
       explicit group boundaries.

    Args:
        exercises: List of exercise dicts (each with at least "is_comb" key).

    Returns:
        List of group dicts with keys "comb" (bool) and "exercises" (list).
    """
    # Use comb_group IDs when any combined exercise has them.
    has_group_ids = any(
        "comb_group" in ex
        for ex in exercises
        if ex.get("is_comb")
    )

    if has_group_ids:
        groups = []
        for ex in exercises:
            is_c = ex.get("is_comb", False)
            gid  = ex.get("comb_group")
            # Extend an existing combined group only when the group ID matches.
            if is_c and gid is not None and groups and groups[-1]["comb"] and groups[-1]["_gid"] == gid:
                groups[-1]["exercises"].append(ex)
            else:
                groups.append({"comb": is_c, "exercises": [ex], "_gid": gid})
        # Strip the internal _gid key before returning.
        return [{"comb": g["comb"], "exercises": g["exercises"]} for g in groups]

    # Fallback: contiguous is_comb=True runs.
    groups = []
    for ex in exercises:
        is_c = ex.get("is_comb", False)
        if is_c and groups and groups[-1]["comb"]:
            groups[-1]["exercises"].append(ex)
        else:
            groups.append({"comb": is_c, "exercises": [ex]})
    return groups


def _week_rep_summary(series: list):
    """Returns 'NxR' string for a list of series dicts, or None if no reps."""
    rep_vals = [s["reps"] for s in series if s.get("reps")]
    if not rep_vals:
        return None
    unique = list(dict.fromkeys(rep_vals))
    return f"{len(series)}x{'/'.join(unique)}"


def _reps_suffix_for_ex(ex: dict) -> str:
    """
    Returns the rep scheme suffix for an exercise across all weeks.
    If all weeks identical: '[3x8 — W1 to W4]'.
    If weeks differ:        '[W1:3x8 | W2:3x8 | W3:3x6 | W4:3x6]'.
    """
    weeks = ex.get("weeks")
    if not weeks:
        return ""
    summaries = [_week_rep_summary(w["series"]) for w in weeks]
    summaries = [s for s in summaries if s]
    if not summaries:
        return ""
    unique_schemes = list(dict.fromkeys(summaries))
    if len(unique_schemes) == 1:
        return f" [{unique_schemes[0]} — W1 to W{len(summaries)}]"
    parts = " | ".join(f"W{i+1}:{s}" for i, s in enumerate(summaries))
    return f" [{parts}]"


def _format_routine_structure(period):
    """
    Formats the exercise structure of a period without execution data.
    Useful for new-routine where the tab doesn't have reps/weights loaded yet.
    Includes target sets × reps from the PDF-parsed data (week 1, rep column).

    Combined groups are rendered as explicit blocks with a header line that
    lists all exercises in the group and a ← REST HERE marker on the last one.
    This lets the AI know exactly where each superset ends (critical for
    assigning rest_s correctly) and avoids treating an entire day as one
    combined sequence.

    Example output for a day with groups [A,B,C,D] / [E,F] / G / H / [I,J]:

        Day 1:
          --- Superset: Abdominal recto largo + Abdominal bolita a dos piernas + Twist ruso + Espinales en colchoneta ---
            #1 Abdominal recto largo [3x8]
            #2 Abdominal bolita a dos piernas [3x8]
            #3 Twist ruso [3x8]
            #4 Espinales en colchoneta [3x10]  ← REST HERE
          --- Superset: Press de pecho con barra + Press de pecho con mancuernas ---
            #5 Press de pecho con barra [3x6]
            #6 Press de pecho con mancuernas [3x8]  ← REST HERE
          #7 Press de pecho en Hammer [3x10]
          #8 Tríceps con polea [3x10]
          --- Superset: Remo al mentón + Vuelos laterales ---
            #9 Remo al mentón [3x6]
            #10 Vuelos laterales [3x8]  ← REST HERE
    """
    def _week_rep_summary(series):
        """Returns 'NxR' for a list of series (e.g., '3x8')."""
        return globals()["_week_rep_summary"](series)

    def _reps_suffix(ex):
        """
        Shows the rep scheme for all 4 weeks.
        If all weeks are identical (most common): '[3x8 — W1 to W4]'.
        If weeks differ: '[W1:3x8 | W2:3x8 | W3:3x6 | W4:3x6]'.
        """
        return _reps_suffix_for_ex(ex)

    lines = []
    for day in period["days"]:
        lines.append(f"Day {day['day']}:")
        groups = _group_combined_exercises(day["exercises"])
        pos = 1  # global exercise counter within the day

        for group in groups:
            if group["comb"]:
                # Render a superset block
                names = " + ".join(ex["name"] for ex in group["exercises"])
                lines.append(f"  --- Superset: {names} ---")
                for i, ex in enumerate(group["exercises"]):
                    suffix = _reps_suffix(ex)
                    is_last = (i == len(group["exercises"]) - 1)
                    rest_marker = "  ← REST HERE" if is_last else ""
                    lines.append(f"    #{pos} {ex['name']}{suffix}{rest_marker}")
                    pos += 1
            else:
                # Single isolated exercise
                ex = group["exercises"][0]
                suffix = _reps_suffix(ex)
                lines.append(f"  #{pos} {ex['name']}{suffix}")
                pos += 1

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


def build_new_routine_prompt(periods, goal, volume_block=None, axial_load_exercises=None):
    """
    New-routine prompt. periods[0] is the freshly-uploaded routine (no real data yet —
    only the structure is used). periods[1:] is the historical context.

    History is capped at 3 prior periods to stay within the 12k TPM limit.
    Older periods add minimal signal for weight suggestions — the most recent
    3 cover all exercises and progression patterns needed.

    Also asks the AI to embed a ```json block with weight/rest suggestions so
    write_suggestions_to_sheet() can parse and apply them directly to the sheet.
    """
    new_period    = periods[0]
    routine_block = _format_routine_structure(new_period)
    prior         = periods[1:4]  # cap at 3 periods — enough signal, fits in 12k TPM
    history_block = _format_exercise_history_compact(prior) if prior else "(no prior history)"
    settled_block = _compute_settled_weights(new_period, prior) if prior else "(sin historial previo)"
    vol_block     = volume_block or "(volumen no calculado)"

    if axial_load_exercises:
        axial_block = ", ".join(axial_load_exercises)
    else:
        axial_block = "(ninguno detectado)"

    result = _load_template("new-routine", goal=goal, period=new_period["period"],
                            routine=routine_block, history=history_block,
                            volume_block=vol_block,
                            axial_load_exercises=axial_block,
                            settled_weights_block=settled_block)
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

def build_weight_suggestions_prompt(
    new_period: dict,
    prior_periods: list,
    goal: str = "hipertrofia",
    axial_load_exercises: list = None,
) -> tuple:
    """
    Build the (system_prompt, user_prompt) pair for the weight suggestions call.

    Uses a global #N index covering all unique exercises across all days.
    Repeated exercises (e.g. the abdomen triset in every day) are defined once
    in the index and referenced by the same key throughout the prompt — no
    redundant name repetition.

    Returns a single (system, user) tuple. One AI call covers the full routine.
    """
    template_path = TEMPLATES_DIR / "weight-suggestions.txt"
    template = template_path.read_text(encoding="utf-8") if template_path.exists() else ""

    index         = _build_global_index(new_period)
    index_block   = _render_global_index_block(index)
    structure_block = _render_period_structure_indexed(new_period, index)
    routine_block = f"{index_block}\n\n{structure_block}"

    settled_block = _render_global_settled_weights_indexed(new_period, prior_periods, index)
    axial_str     = ", ".join(axial_load_exercises) if axial_load_exercises else "ninguno detectado"
    period_label  = new_period.get("period", "")

    user_prompt = template.format(
        routine=routine_block,
        settled_weights_block=settled_block,
        axial_load_exercises=axial_str,
        period=period_label,
        goal=goal,
    )

    system_prompt = (
        "Sos un entrenador de Powerbuilding de élite y analista de rendimiento deportivo. "
        "Analizás números con frialdad matemática. "
        "Hablás con voseo argentino. "
        "Seguís las reglas del prompt al pie de la letra. "
        "Antes de calcular cualquier peso, rastreás el ejercicio en TODO el historial cronológico provisto "
        "para identificar la tendencia real de carga. "
        "Si el rango de repeticiones cambió entre períodos, aplicás estimación de 1RM submáxima "
        "(fórmula: peso × (1 + reps/30)) para adaptar el peso al nuevo rango. "
        "Si hay notas de dolor o molestia en el historial de un ejercicio, penalizás el peso sugerido "
        "un 10% e indicás la advertencia en el campo 'progression_analysis'."
    )

    return system_prompt, user_prompt


def get_weight_suggestions(
    new_period: dict,
    prior_periods: list,
    api_key: str,
    goal: str = "hipertrofia",
    axial_load_exercises: list = None,
) -> list:
    """
    Makes a single structured Gemini call to get weight suggestions for a new routine.

    Uses build_weight_suggestions_prompt() which produces one unified prompt with
    a global #N exercise index. Returns a flat list of weight suggestion dicts —
    same shape as before so validate_suggestions and write_suggestions_to_sheet
    work unchanged.
    """
    system, prompt = build_weight_suggestions_prompt(
        new_period, prior_periods, goal=goal, axial_load_exercises=axial_load_exercises
    )

    client = genai.Client(api_key=api_key)

    log_dir  = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"gemini_{datetime.now().strftime('%Y%m%d')}.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{timestamp}] MODEL: {MODEL_STRUCTURED} [STRUCTURED]\n")
        f.write(f"--- WEIGHT SUGGESTIONS PROMPT ({len(prompt)} chars) ---\n")
        f.write(prompt + "\n")

    print(f"  Sending [weight-suggestions] prompt to Gemini ({MODEL_STRUCTURED})...", flush=True)

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=MODEL_STRUCTURED,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=WeightSuggestionList,
                ),
                contents=prompt,
            )
            items: list[WeightSuggestion] = response.parsed.root
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"--- STRUCTURED RESPONSE ({len(items)} items) ---\n")
                for item in items:
                    f.write(f"  {item.model_dump()}\n")
            return [item.model_dump() for item in items]
        except Exception as e:
            err = str(e)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"--- ERROR (attempt {attempt+1}) ---\n{err}\n")
            if ("429" in err or "RESOURCE_EXHAUSTED" in err or "503" in err or "UNAVAILABLE" in err) and attempt == 0:
                print("  Temporary error — waiting 65s and retrying...")
                time.sleep(65)
                continue
            raise


def translate_to_spanish(text: str, api_key: str) -> str:
    """Translates arbitrary text to Rioplatense Spanish."""
    client = genai.Client(api_key=api_key)
    system_prompt = (
        "Traducís texto al español de Argentina. "
        "Mantenés el significado original, el formato y los números. "
        "No agregás explicaciones ni comentarios."
    )
    return _call_gemini(client, system_prompt, text, max_tokens=2048, thinking_budget=0)


def analyze(periods, api_key, mock=False, mode="global", goal="hipertrofia",
            current_week_data=None, prev_week_data=None, current_week_num=None,
            prev_report=None, volume_block=None, axial_load_exercises=None):
    """
    Generates a training analysis according to the requested mode.

    Args:
        periods:               List of periods (most recent first).
        api_key:               Gemini API key.
        mock:                  If True, returns a test analysis without calling the API.
        mode:                  Analysis mode: 'global', 'new-routine', 'monthly', 'weekly'.
        goal:                  User goal (e.g. 'hypertrophy').
        current_week_data:     Current week data (only for 'weekly' mode).
        prev_week_data:        Previous week data (only for 'weekly' mode, can be None).
        current_week_num:      Current week number 1-based (only for 'weekly' mode).
        prev_report:           Text of the previous analysis for this mode (Markdown).
        volume_block:          Pre-calculated weekly volume string (only for 'new-routine' mode).
        axial_load_exercises:  List of exercise names with axial load (only for 'new-routine' mode).

    Returns:
        String with the analysis in Markdown.
    """
    if mock:
        return _MOCK_OUTPUTS.get(mode, _MOCK_OUTPUTS["global"])

    if mode == "new-routine":
        prompt = build_new_routine_prompt(periods, goal, volume_block=volume_block,
                                          axial_load_exercises=axial_load_exercises)
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

    client = genai.Client(api_key=api_key)
    system = _make_system_prompt(goal)
    print(f"  Sending [{mode}] prompt to Gemini...", flush=True)
    # Disable thinking for prose generation — thinking tokens count against
    # max_output_tokens in gemini-2.5-flash, leaving almost no budget for output.
    return _call_gemini(client, system, prompt, max_tokens=16384, thinking_budget=0)
