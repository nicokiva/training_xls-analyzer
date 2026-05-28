"""
helpers/catalog/catalog.py — Exercise catalog for muscle group classification.

Maintains a local JSON file mapping exercise names (normalized) to:
  - primary muscle group
  - secondary muscle groups (indirect volume)
  - movement pattern

Unknown exercises are auto-classified via a fast Groq call (llama-3.1-8b-instant)
and saved so they are never classified twice.
"""

import json
import unicodedata
from pathlib import Path

CATALOG_PATH = Path(__file__).parent.parent.parent / "exercise_catalog.json"

# ── Normalization ──────────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """Lowercase, strip accents and punctuation for consistent catalog keys."""
    nfkd = unicodedata.normalize("NFKD", name.lower().strip())
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Remove parentheses content and extra whitespace for fuzzy matching
    import re
    ascii_str = re.sub(r"\(.*?\)", "", ascii_str).strip()
    return re.sub(r"\s+", " ", ascii_str)


# ── Catalog I/O ────────────────────────────────────────────────────────────────

def load_catalog() -> dict:
    if CATALOG_PATH.exists():
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {}


def save_catalog(catalog: dict) -> None:
    CATALOG_PATH.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Groq classification ────────────────────────────────────────────────────────

def _classify_via_groq(name: str, api_key: str) -> dict:
    """Ask Groq (fast small model) to classify an exercise. Returns the info dict."""
    from groq import Groq
    client = Groq(api_key=api_key)
    prompt = (
        f'Classify this gym exercise: "{name}"\n\n'
        "Respond ONLY with a JSON object, no markdown, no explanation:\n"
        '{"primary": "<muscle in Spanish>", '
        '"secondary": ["<muscle>", ...], '
        '"pattern": "<pattern>", '
        '"axial_load": <true|false>}\n\n'
        "Primary muscle options: pecho, espalda, bíceps, tríceps, hombros, "
        "cuádriceps, isquiotibiales, glúteos, core\n"
        "Pattern options: horizontal_push, horizontal_pull, vertical_push, "
        "vertical_pull, squat, hip_hinge, lunge, elbow_flexion, "
        "elbow_extension, shoulder_isolation, core\n"
        "axial_load: true if the exercise places compressive load on the spine "
        "with a free barbell (e.g. squat, deadlift, barbell overhead press, "
        "barbell row). false for machines, cables, dumbbells, bodyweight.\n\n"
        "Important rules:\n"
        "- 'Peso muerto' and deadlift variations → primary=espalda, pattern=hip_hinge, axial_load=true\n"
        "- 'Sentadilla' (barbell squat) → primary=cuádriceps, pattern=squat, axial_load=true\n"
        "- 'Prensa' (leg press machine) → primary=cuádriceps, pattern=squat, axial_load=false\n"
        "- 'Remo' (row) → primary=espalda, pattern=horizontal_pull\n"
        "- 'Dominada' (pull-up/chin-up) → primary=espalda, pattern=vertical_pull, axial_load=false\n"
        "- 'Tirón dorsal' (lat pulldown cable) → primary=espalda, pattern=vertical_pull, axial_load=false\n"
        "- 'Flexión de rodillas' (leg curl machine) → primary=isquiotibiales, pattern=hip_hinge, axial_load=false\n"
        "- 'Rotaciones' with torso → primary=core, pattern=core, axial_load=false\n"
        "- 'Extensión de cadera' → primary=glúteos, pattern=hip_hinge, axial_load=false\n"
        "- 'Depresores en polea' (cable lat depression) → primary=espalda, pattern=vertical_pull, axial_load=false\n"
        "- Barbell shoulder press ('Empuje de hombros con barra') → primary=hombros, pattern=vertical_push, axial_load=true\n"
        "- Dumbbell/machine shoulder press → primary=hombros, pattern=vertical_push, axial_load=false\n"
    )
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown fences if present
    import re
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


# ── Public API ─────────────────────────────────────────────────────────────────

def ensure_classified(exercise_names: list[str], api_key: str) -> dict:
    """
    Ensure every exercise in exercise_names is in the catalog.
    Unknown exercises are classified via Groq and saved.
    Returns the (potentially updated) catalog.
    """
    catalog = load_catalog()
    changed = False
    for name in exercise_names:
        key = _normalize(name)
        if key not in catalog:
            print(f"  [catalog] Classifying new exercise: '{name}'...")
            try:
                info = _classify_via_groq(name, api_key)
                catalog[key] = {**info, "_original": name}
                changed = True
                print(f"  [catalog]   → primary={info.get('primary')}, "
                      f"pattern={info.get('pattern')}")
            except Exception as exc:
                print(f"  [catalog] Warning: could not classify '{name}': {exc}")
    if changed:
        save_catalog(catalog)
    return catalog


def get_axial_load_exercises(exercise_names: list[str], catalog: dict) -> list[str]:
    """
    Return the original names of exercises in the list that have axial_load=True.
    Used to build the dynamic safety guardrail in the prompt.
    """
    result = []
    for name in exercise_names:
        key = _normalize(name)
        info = catalog.get(key, {})
        if info.get("axial_load"):
            result.append(name)
    return result


def calculate_volume(period: dict, catalog: dict) -> dict:
    """
    Calculate weekly sets per muscle group (direct + indirect) for a period.

    Uses week 1 series count as the target number of sets per session.
    Each exercise contributes to:
      - primary muscle: counted as "direct"
      - secondary muscles: counted as "indirect"

    Returns:
        {
          "bíceps": {"direct": 3, "indirect": 9, "detail": ["D4: Biceps con polea (3s direct)", ...]},
          ...
        }
    """
    volume: dict = {}

    for day in period["days"]:
        for ex in day["exercises"]:
            key = _normalize(ex["name"])
            info = catalog.get(key)
            if not info:
                continue

            n_sets = len(ex["weeks"][0]["series"]) if ex.get("weeks") else 3
            day_num = day["day"]

            primary   = (info.get("primary") or "").lower().strip()
            secondary = [m.lower().strip() for m in (info.get("secondary") or []) if m]

            if primary:
                entry = volume.setdefault(primary, {"direct": 0, "indirect": 0, "detail": []})
                entry["direct"] += n_sets
                entry["detail"].append(f"D{day_num}: {ex['name']} ({n_sets}s directo)")

            for muscle in secondary:
                entry = volume.setdefault(muscle, {"direct": 0, "indirect": 0, "detail": []})
                entry["indirect"] += n_sets
                entry["detail"].append(f"D{day_num}: {ex['name']} ({n_sets}s indirecto)")

    return volume


def format_volume_block(volume: dict) -> str:
    """Format the volume dict as a readable block to inject into the AI prompt."""
    if not volume:
        return "(volumen no calculado)"

    lines = ["Volumen semanal calculado por el script (series reales del coach):"]
    sorted_muscles = sorted(
        volume.keys(),
        key=lambda m: -(volume[m]["direct"] + volume[m]["indirect"])
    )
    for muscle in sorted_muscles:
        data = volume[muscle]
        direct   = data["direct"]
        indirect = data["indirect"]
        total    = direct + indirect
        detail   = " | ".join(data["detail"])
        lines.append(
            f"  {muscle.capitalize()}: {direct}s directas + {indirect}s indirectas "
            f"= {total}s totales  [{detail}]"
        )
    return "\n".join(lines)
