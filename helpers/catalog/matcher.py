"""
helpers/catalog/matcher.py — Biomechanical exercise matcher.

Given a new exercise and a list of history exercises with known weights, finds
the closest biomechanical match using the shared exercise catalog
(shared/training_shared/exercise_catalog.json).

Scoring model (strict patron_principal gate, then points):
    mecanica:       +3   (movement mechanics — strongest predictor of load transfer)
    vector:         +2   (force direction — horizontal vs vertical vs diagonal)
    estabilizacion: +1   (stabilization demand — bilateral vs unilateral vs corporal)
    ─────────────────────
    Max score:       6

The function is used as a fallback inside _compute_settled_weights() when no
exact name match or partial match is found for a new exercise.
"""

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

# Path to the shared catalog — two levels up from helpers/catalog/ to reach the
# routine-analyzer root, then three levels up to reach the training/ workspace root,
# then into shared/training_shared/.
_SHARED_CATALOG_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "shared"
    / "training_shared"
    / "exercise_catalog.json"
)


def _normalize(name: str) -> str:
    """Lowercase + strip accents for catalog key comparison."""
    nfkd = unicodedata.normalize("NFKD", name.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@lru_cache(maxsize=1)
def load_shared_catalog() -> dict:
    """
    Load the shared exercise catalog and return it as a dict keyed by
    normalized exercise name.

    Returns an empty dict (silently) if the catalog file doesn't exist so the
    rest of the pipeline degrades gracefully rather than crashing.

    The result is cached with lru_cache so the file is read at most once per
    process, regardless of how many times the function is called.
    """
    if not _SHARED_CATALOG_PATH.exists():
        return {}
    entries = json.loads(_SHARED_CATALOG_PATH.read_text(encoding="utf-8"))
    # Key by normalized name so lookups are case- and accent-insensitive.
    return {_normalize(e["ejercicio"]): e for e in entries}


def find_closest_exercise(
    exercise_name: str,
    history: list,
    catalog: dict = None,
) -> tuple:
    """
    Find the closest biomechanically equivalent exercise in a history list.

    Strict rule: only considers history entries that share the same
    patron_principal (primary movement pattern). Among those, scores by
    mecanica, vector and estabilizacion similarity.

    Args:
        exercise_name: Name of the new (target) exercise to find a match for.
        history:       List of dicts with at least:
                         "name"         (str)   — original exercise name
                         "settled_peso" (float) — last settled weight in kg
                       Additional fields ("period", "n_weeks", "is_comb") are
                       preserved in the returned match dict if present.
        catalog:       Pre-loaded shared catalog dict (keyed by normalized name).
                       If None, loads from disk via load_shared_catalog().

    Returns:
        (best_match, score) where:
          best_match — the history entry dict with the highest score, or None
          score      — integer 0-6 (higher = closer match), or -1 if no match found

    Note on ties: when two history exercises have the same score, the first one
    encountered in the list wins. Callers can break ties using "n_weeks" or
    "period" from the returned match dict.
    """
    if catalog is None:
        catalog = load_shared_catalog()

    target_key = _normalize(exercise_name)
    target_meta = catalog.get(target_key)
    if target_meta is None:
        # New exercise not in the shared catalog — cannot score biomechanically.
        return None, -1

    best_match = None
    best_score = -1

    for entry in history:
        hist_key = _normalize(entry["name"])
        hist_meta = catalog.get(hist_key)
        if hist_meta is None:
            continue

        # Strict gate: must share the same primary movement pattern.
        if target_meta["patron_principal"] != hist_meta["patron_principal"]:
            continue

        score = 0
        if target_meta["mecanica"] == hist_meta["mecanica"]:
            score += 3
        if target_meta["vector"] == hist_meta["vector"]:
            score += 2
        if target_meta["estabilizacion"] == hist_meta["estabilizacion"]:
            score += 1

        if score > best_score:
            best_score = score
            best_match = entry

    return best_match, best_score
