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


def _compute_settled_weights(new_period: dict, prior_periods: list) -> str:
    """
    For each exercise in new_period, find its most recent settled weight from
    prior_periods using normalized name matching, with biomechanical pattern
    fallback when no exact name match exists.

    Shows the most recent entry AND (if it has fewer than 3 weeks of data) also
    the most complete entry so the AI has reliable baselines when a recent period
    is sparse.

    Returns a formatted block to inject into the prompt so the AI doesn't need
    to parse exercise names from raw history text.
    """
    import re

    def strip_parens(s):
        return re.sub(r"\s*\(.*?\)", "", s).strip()

    # Build lookup: (normalized_name, is_comb) → [(period_label, settled, n_weeks, is_comb)]
    # Keying by is_comb ensures isolated and combined weights are never cross-compared.
    # ordered from most recent (prior_periods[0]) to oldest
    lookup: dict = {}
    for period_data in prior_periods:
        period_label = period_data["period"][:8]
        for day_data in period_data["days"]:
            for ex in day_data["exercises"]:
                n_weeks, settled = _last_settled_weeks(ex)
                if settled is None:
                    continue
                raw_key   = _normalize_ex_name(strip_parens(ex["name"]))
                canon_key = _canonical_key(ex["name"])
                is_c = ex.get("is_comb", False)
                entry = (period_label, settled, n_weeks, is_c)
                lookup.setdefault((raw_key, is_c), []).append(entry)
                if canon_key != raw_key:
                    lookup.setdefault((canon_key, is_c), []).append(entry)

    lines = ["Pesos de cierre por ejercicio (calculados por el script — usar como baseline):"]
    for day_data in new_period["days"]:
        day_num  = day_data["day"]
        n_in_day = len(day_data["exercises"])
        for pos, ex in enumerate(day_data["exercises"], start=1):
            raw_key   = _normalize_ex_name(strip_parens(ex["name"]))
            canon_key = _canonical_key(ex["name"])
            new_is_c  = ex.get("is_comb", False)
            pos_note  = f" [#{pos}/{n_in_day} del día]"
            # Try canonical key first (alias resolution), then raw key — always same is_comb
            entries = lookup.get((canon_key, new_is_c)) or lookup.get((raw_key, new_is_c))
            isolated_fallback = False
            # If combined and no match found, fall back to isolated history as reference point
            if not entries and new_is_c:
                entries = lookup.get((canon_key, False)) or lookup.get((raw_key, False))
                if entries:
                    isolated_fallback = True
            key = canon_key  # used for partial/archetype fallback below

            if entries:
                # ── Exact name match ──────────────────────────────────────────
                most_recent = entries[0]
                p_label, settled, n_weeks, is_c = most_recent
                comb_note  = " (era combinado)" if is_c else ""
                weeks_note = f" [{n_weeks} semanas de datos]"
                if isolated_fallback:
                    line = (f"  D{day_num} {ex['name']}{pos_note}: sin historial combinado → "
                            f"referencia aislada {p_label} → {settled:.1f} kg{weeks_note} "
                            f"(nuevo en contexto combinado; sugerí ~90% como punto de partida)")
                else:
                    line = (f"  D{day_num} {ex['name']}{pos_note}: {p_label} → {settled:.1f} kg"
                            f"{comb_note}{weeks_note}")

                # If most recent is sparse (<3 weeks), also show the most complete entry
                if n_weeks < 3 and len(entries) > 1:
                    best = max(entries[1:], key=lambda e: e[2])
                    if best[2] > n_weeks:
                        b_label, b_settled, b_weeks, b_comb = best
                        b_comb_note = " (era combinado)" if b_comb else ""
                        line += (f" | referencia más completa: {b_label} → "
                                 f"{b_settled:.1f} kg [{b_weeks} semanas]{b_comb_note}")
                lines.append(line)

            else:
                # ── Partial name match (e.g. "Dominada" matches "Dominada estricta") ──
                partial_entries = None
                for (hist_key, hist_is_c), hist_entries in lookup.items():
                    if hist_is_c == new_is_c and (key.startswith(hist_key) or hist_key.startswith(key)):
                        partial_entries = hist_entries
                        break

                if partial_entries:
                    most_recent = partial_entries[0]
                    p_label, settled, n_weeks, is_c = most_recent
                    comb_note  = " (era combinado)" if is_c else ""
                    weeks_note = f" [{n_weeks} semanas de datos]"
                    line = (f"  D{day_num} {ex['name']}{pos_note}: {p_label} → {settled:.1f} kg"
                            f"{comb_note}{weeks_note}")
                    if n_weeks < 3 and len(partial_entries) > 1:
                        best = max(partial_entries[1:], key=lambda e: e[2])
                        if best[2] > n_weeks:
                            b_label, b_settled, b_weeks, b_comb = best
                            b_comb_note = " (era combinado)" if b_comb else ""
                            line += (f" | referencia más completa: {b_label} → "
                                     f"{b_settled:.1f} kg [{b_weeks} semanas]{b_comb_note}")
                    lines.append(line)

                else:
                    # ── Biomechanical pattern fallback ────────────────────────
                    archetype = _get_archetype(ex["name"])
                    best_match = None  # (period_label, src_name, adjusted_weight, n_weeks, factor)

                    if archetype:
                        for (hist_key, hist_is_c), hist_entries in lookup.items():
                            if hist_is_c != new_is_c:
                                continue  # never cross isolated ↔ combined
                            if _get_archetype(hist_key) == archetype:
                                candidate = hist_entries[0]
                                p_label, settled, n_weeks, _ = candidate
                                factor = _transfer_factor(hist_key, ex["name"])
                                adjusted = round(settled * factor / 2.5) * 2.5
                                if best_match is None or n_weeks > best_match[3]:
                                    best_match = (p_label, hist_key, adjusted, n_weeks, factor)

                    if best_match:
                        p_label, src_name, adjusted, n_weeks, factor = best_match
                        lines.append(
                            f"  D{day_num} {ex['name']}{pos_note}: sin historial exacto → "
                            f"ejercicio equiv. '{src_name}' ({p_label}) "
                            f"→ baseline ajustado {adjusted:.1f} kg [{n_weeks} semanas de datos]"
                        )
                    else:
                        lines.append(f"  D{day_num} {ex['name']}{pos_note}: sin historial previo (ejercicio nuevo sin equivalente)")
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


def _format_routine_structure(period):
    """
    Formats the exercise structure of a period without execution data.
    Useful for new-routine where the tab doesn't have reps/weights loaded yet.
    Includes target sets × reps from the PDF-parsed data (week 1, rep column).
    Combined exercises are marked with (combinado).
    Position within the day is shown (#1, #2, …) so the AI can account for
    accumulated fatigue when suggesting weights.
    """
    lines = []
    for day in period["days"]:
        lines.append(f"Day {day['day']}:")
        for pos, ex in enumerate(day["exercises"], start=1):
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
            lines.append(f"  #{pos} {label}{suffix}")
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

def get_weight_suggestions(
    new_period: dict,
    prior_periods: list,
    api_key: str,
    goal: str = "hipertrofia",
    axial_load_exercises: list = None,
) -> list[dict]:
    """
    Makes a separate structured Gemini call to get weight suggestions for a new routine.

    Uses response_schema=List[WeightSuggestion] so Gemini is forced to produce
    valid typed JSON at the token level — no regex parsing, no hallucinated formats.

    Returns a list of dicts (same shape as parse_suggestions output) so the
    rest of the pipeline (validate_suggestions, write_suggestions_to_sheet) works unchanged.
    """
    template_path = TEMPLATES_DIR / "weight-suggestions.txt"
    template = template_path.read_text(encoding="utf-8") if template_path.exists() else ""

    routine_block  = _format_routine_structure(new_period)
    settled_block  = _compute_settled_weights(new_period, prior_periods)
    axial_str      = ", ".join(axial_load_exercises) if axial_load_exercises else "ninguno detectado"
    period_label   = new_period.get("period", "")

    prompt = template.format(
        routine=routine_block,
        settled_weights_block=settled_block,
        axial_load_exercises=axial_str,
        period=period_label,
        goal=goal,
    )

    client = genai.Client(api_key=api_key)
    system = (
        "Sos un entrenador de Powerbuilding de élite y analista de rendimiento deportivo. "
        "Analizás números con frialdad matemática. "
        "Hablás con voseo argentino. "
        "Seguís las reglas del prompt al pie de la letra. "
        "Antes de calcular cualquier peso, rastreás el ejercicio en TODO el historial cronológico provisto "
        "para identificar la tendencia real de carga. "
        "Si el rango de repeticiones cambió entre períodos, aplicás estimación de 1RM submáxima "
        "(fórmula: peso × (1 + reps/30)) para adaptar el peso al nuevo rango. "
        "Si hay notas de dolor o molestia en el historial de un ejercicio, penalizás el peso sugerido un 10% "
        "e indicás la advertencia en el campo 'progression_analysis'."
    )

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
            # Convert Pydantic objects → plain dicts for the rest of the pipeline
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


def translate_to_spanish(text, api_key):
    """
    Translates the given text to Spanish using Gemini.

    Args:
        text:    The text to translate (Markdown analysis).
        api_key: Gemini API key.

    Returns:
        String with the translated text in Spanish.
    """
    client = genai.Client(api_key=api_key)
    system = (
        "You are a professional translator specializing in Argentine Spanish. "
        "Translate the following text ENTIRELY to Spanish (Argentina). "
        "Every single word must be in Spanish — do not leave any English words or phrases. "
        "Keep markdown formatting intact. Do not add any commentary — only return the translated text."
    )
    return _call_gemini(client, system, text, max_tokens=4096)


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
