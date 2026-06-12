"""
Tests for helpers/catalog/matcher.py — biomechanical exercise matcher.
"""

import pytest
from helpers.catalog.matcher import find_closest_exercise, _normalize


# ---------------------------------------------------------------------------
# Fake catalog — exercises with known biomechanical attributes
# ---------------------------------------------------------------------------

# Catalog is keyed by normalized exercise name (lowercase, accent-stripped).
FAKE_CATALOG = {
    # Chest — horizontal push, bilateral, compound
    "press de pecho con barra en banco plano": {
        "ejercicio": "Press de pecho con barra en banco plano",
        "patron_principal": "Empuje horizontal",
        "vector": "Horizontal",
        "mecanica": "Compuesto",
        "estabilizacion": "Bilateral",
    },
    "press de pecho con mancuernas en banco plano": {
        "ejercicio": "Press de pecho con mancuernas en banco plano",
        "patron_principal": "Empuje horizontal",
        "vector": "Horizontal",
        "mecanica": "Compuesto",
        "estabilizacion": "Bilateral",
    },
    "press de pecho en banco inclinado con barra": {
        "ejercicio": "Press de pecho en banco inclinado con barra",
        "patron_principal": "Empuje horizontal",
        "vector": "Diagonal",
        "mecanica": "Compuesto",
        "estabilizacion": "Bilateral",
    },
    # Shoulders — vertical push, bilateral, compound
    "press militar con barra": {
        "ejercicio": "Press militar con barra",
        "patron_principal": "Empuje vertical",
        "vector": "Vertical",
        "mecanica": "Compuesto",
        "estabilizacion": "Bilateral",
    },
    "press de hombros con mancuernas": {
        "ejercicio": "Press de hombros con mancuernas",
        "patron_principal": "Empuje vertical",
        "vector": "Vertical",
        "mecanica": "Compuesto",
        "estabilizacion": "Bilateral",
    },
    # Legs — squat pattern, bilateral, compound
    "sentadilla clasica": {
        "ejercicio": "Sentadilla clasica",
        "patron_principal": "Squat",
        "vector": "Vertical",
        "mecanica": "Compuesto",
        "estabilizacion": "Bilateral",
    },
    # Arms — elbow flexion, unilateral, isolation
    "biceps con mancuernas": {
        "ejercicio": "Biceps con mancuernas",
        "patron_principal": "Flexion de codo",
        "vector": "Vertical",
        "mecanica": "Aislamiento",
        "estabilizacion": "Unilateral",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(name, settled_peso=60.0, n_weeks=3, period="01/26"):
    """Create a minimal history entry dict."""
    return {
        "name":         name,
        "settled_peso": settled_peso,
        "n_weeks":      n_weeks,
        "period":       period,
        "is_comb":      False,
    }


# ---------------------------------------------------------------------------
# TestFindClosestExercise
# ---------------------------------------------------------------------------

class TestFindClosestExercise:
    """Tests for find_closest_exercise()."""

    def test_exact_match_returns_perfect_score(self):
        """Same exercise in history → score 6 (all three attributes match)."""
        history = [_make_entry("Press de pecho con barra en banco plano", settled_peso=80.0)]
        match, score = find_closest_exercise(
            "Press de pecho con barra en banco plano", history, FAKE_CATALOG
        )
        assert match is not None
        assert score == 6
        assert match["name"] == "Press de pecho con barra en banco plano"

    def test_same_pattern_different_equipment_scores_mechanica_only(self):
        """Same patron, same mecanica, different vector → score 3."""
        # Target: inclined press (Diagonal vector, Compuesto, Bilateral)
        # History: flat press (Horizontal vector, Compuesto, Bilateral)
        # Shared: patron_principal=Empuje horizontal, mecanica=Compuesto, estabilizacion=Bilateral
        # Missing: vector (Diagonal vs Horizontal)
        history = [_make_entry("Press de pecho con barra en banco plano", settled_peso=80.0)]
        match, score = find_closest_exercise(
            "Press de pecho en banco inclinado con barra", history, FAKE_CATALOG
        )
        assert match is not None
        # mecanica (+3) + estabilizacion (+1) = 4 (same patron and bilateral, different vector)
        assert score == 4
        assert match["settled_peso"] == 80.0

    def test_different_patron_returns_no_match(self):
        """Exercises from different patron_principal → gated out, no match returned."""
        # Target: chest press (Empuje horizontal)
        # History: squat (Squat) — completely different pattern
        history = [_make_entry("Sentadilla clasica", settled_peso=100.0)]
        match, score = find_closest_exercise(
            "Press de pecho con barra en banco plano", history, FAKE_CATALOG
        )
        assert match is None
        assert score == -1

    def test_target_not_in_catalog_returns_no_match(self):
        """Exercise not present in catalog → cannot score, returns (None, -1)."""
        history = [_make_entry("Press de pecho con barra en banco plano")]
        match, score = find_closest_exercise(
            "Ejercicio inventado sin catalog", history, FAKE_CATALOG
        )
        assert match is None
        assert score == -1

    def test_history_entry_not_in_catalog_is_skipped(self):
        """History entry missing from catalog is skipped; still finds other matches."""
        history = [
            _make_entry("Ejercicio sin catalog", settled_peso=50.0),
            _make_entry("Press de pecho con mancuernas en banco plano", settled_peso=40.0),
        ]
        match, score = find_closest_exercise(
            "Press de pecho con barra en banco plano", history, FAKE_CATALOG
        )
        assert match is not None
        assert match["name"] == "Press de pecho con mancuernas en banco plano"

    def test_selects_highest_scoring_match(self):
        """When multiple history exercises qualify, the one with the highest score wins."""
        history = [
            # Score 4: same patron + mecanica + estabilizacion but different vector
            _make_entry("Press de pecho en banco inclinado con barra", settled_peso=70.0),
            # Score 6: perfect match (same patron + mecanica + vector + estabilizacion)
            _make_entry("Press de pecho con mancuernas en banco plano", settled_peso=50.0),
        ]
        match, score = find_closest_exercise(
            "Press de pecho con barra en banco plano", history, FAKE_CATALOG
        )
        assert score == 6
        assert match["name"] == "Press de pecho con mancuernas en banco plano"

    def test_empty_history_returns_no_match(self):
        """Empty history list → nothing to compare against."""
        match, score = find_closest_exercise(
            "Press de pecho con barra en banco plano", [], FAKE_CATALOG
        )
        assert match is None
        assert score == -1

    def test_empty_catalog_returns_no_match(self):
        """Empty catalog → target cannot be looked up."""
        history = [_make_entry("Press de pecho con barra en banco plano")]
        match, score = find_closest_exercise(
            "Press de pecho con barra en banco plano", history, {}
        )
        assert match is None
        assert score == -1

    def test_score_range_is_zero_to_six(self):
        """Score is always in [0, 6] when a match is found."""
        history = [_make_entry("Press militar con barra", settled_peso=60.0)]
        match, score = find_closest_exercise(
            "Press de hombros con mancuernas", history, FAKE_CATALOG
        )
        # Same patron (Empuje vertical), same vector, same mecanica, same estabilizacion → 6
        assert 0 <= score <= 6

    def test_match_result_preserves_full_entry_dict(self):
        """Returned match is the original history entry dict (all fields intact)."""
        entry = _make_entry("Press de pecho con mancuernas en banco plano", settled_peso=55.0, n_weeks=4, period="05/26")
        match, _ = find_closest_exercise(
            "Press de pecho con barra en banco plano", [entry], FAKE_CATALOG
        )
        assert match is entry  # same object reference

    def test_score_zero_when_only_patron_matches(self):
        """
        If patron matches but no other attribute does, score is 0.
        A score of 0 still means a match was found (the gate passed).
        """
        # Build a fake catalog with exercises sharing patron but differing in all other attributes
        catalog = {
            "ejercicio a": {
                "patron_principal": "Empuje horizontal",
                "vector": "Horizontal",
                "mecanica": "Compuesto",
                "estabilizacion": "Bilateral",
            },
            "ejercicio b": {
                "patron_principal": "Empuje horizontal",
                "vector": "Diagonal",
                "mecanica": "Aislamiento",
                "estabilizacion": "Unilateral",
            },
        }
        history = [_make_entry("Ejercicio B", settled_peso=30.0)]
        match, score = find_closest_exercise("Ejercicio A", history, catalog)
        assert match is not None
        assert score == 0


# ---------------------------------------------------------------------------
# TestNormalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases(self):
        assert _normalize("Press") == "press"

    def test_strips_accents(self):
        assert _normalize("sentadilla clásica") == "sentadilla clasica"

    def test_strips_leading_trailing_spaces(self):
        assert _normalize("  Bíceps  ") == "biceps"

    def test_combined_transforms(self):
        assert _normalize("Empuje Hombros con Barra (sentado)") == "empuje hombros con barra (sentado)"
