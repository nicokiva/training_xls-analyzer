"""
Tests for helpers/ai/ai.py — pure/prompt-building functions and mock mode.
"""
import pytest
from helpers.ai import (
    _format_exercise_history,
    _format_routine_structure,
    build_global_prompt,
    build_new_routine_prompt,
    build_weekly_prompt,
    analyze,
)
from helpers.ai.ai import _group_combined_exercises


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_series(reps="10", peso="60"):
    return {"reps": reps, "peso": peso}


def _make_week(week_num, has_data=True):
    if has_data:
        return {"week": week_num, "series": [_make_series() for _ in range(3)]}
    return {"week": week_num, "series": [{"reps": "", "peso": ""} for _ in range(3)]}


def _make_exercise(name, weeks_with_data=(1,), is_comb=False):
    weeks = [_make_week(w, has_data=(w in weeks_with_data)) for w in range(1, 5)]
    return {"name": name, "is_comb": is_comb, "weeks": weeks}


def _make_period(period_name, exercises, day=1):
    return {
        "period": period_name,
        "days": [{"day": day, "exercises": exercises}],
    }


PERIOD_A = _make_period("01/01/26-31/01/26", [
    _make_exercise("Sentadilla", (1, 2)),
    _make_exercise("Press plano", (1,)),
])

PERIOD_B = _make_period("01/02/26-28/02/26", [
    _make_exercise("Sentadilla", (1, 2, 3)),
    _make_exercise("Dominada", (1, 2)),
])


# ---------------------------------------------------------------------------
# _format_exercise_history
# ---------------------------------------------------------------------------

class TestFormatExerciseHistory:
    def test_groups_by_exercise_name(self):
        result = _format_exercise_history([PERIOD_A])
        assert "**Sentadilla**" in result
        assert "**Press plano**" in result

    def test_includes_period_name(self):
        result = _format_exercise_history([PERIOD_A])
        assert PERIOD_A["period"] in result

    def test_orders_chronologically(self):
        """Periods are passed newest-first but history should be oldest-first."""
        result = _format_exercise_history([PERIOD_B, PERIOD_A])
        # PERIOD_A (older, index 1 when reversed to oldest-first) should appear before PERIOD_B
        idx_a = result.index(PERIOD_A["period"])
        idx_b = result.index(PERIOD_B["period"])
        assert idx_a < idx_b

    def test_only_weeks_with_data_are_included(self):
        result = _format_exercise_history([PERIOD_A])
        # Press plano only has week 1 data — Wk3 and Wk4 should not appear for it
        # We check that "Wk3" doesn't appear for Press plano block
        lines = result.split("\n")
        press_idx = next(i for i, l in enumerate(lines) if "Press plano" in l)
        # Find lines belonging to press block (until next blank or next exercise)
        press_lines = []
        for line in lines[press_idx + 1:]:
            if not line.strip() or "**" in line:
                break
            press_lines.append(line)
        combined = " ".join(press_lines)
        assert "Wk3" not in combined
        assert "Wk4" not in combined

    def test_empty_periods_returns_empty_string(self):
        result = _format_exercise_history([])
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# build_global_prompt
# ---------------------------------------------------------------------------

class TestBuildGlobalPrompt:
    def test_contains_goal(self):
        prompt = build_global_prompt([PERIOD_A], goal="hipertrofia")
        assert "hipertrofia" in prompt

    def test_contains_exercise_names(self):
        prompt = build_global_prompt([PERIOD_A], goal="fuerza")
        # Names appear as canonical keys in the history block (lowercased, alias-resolved)
        assert "sentadilla" in prompt.lower()
        assert "press plano" in prompt.lower() or "empuje de pecho" in prompt.lower()

    def test_is_string(self):
        prompt = build_global_prompt([PERIOD_A], goal="test")
        assert isinstance(prompt, str)
        assert len(prompt) > 50


# ---------------------------------------------------------------------------
# build_new_routine_prompt
# ---------------------------------------------------------------------------

class TestBuildNewRoutinePrompt:
    def test_contains_new_routine_period(self):
        prompt = build_new_routine_prompt([PERIOD_B, PERIOD_A], goal="hipertrofia")
        assert PERIOD_B["period"] in prompt

    def test_contains_goal(self):
        prompt = build_new_routine_prompt([PERIOD_B, PERIOD_A], goal="fuerza")
        assert "fuerza" in prompt

    def test_contains_exercise_names_from_new_routine(self):
        prompt = build_new_routine_prompt([PERIOD_B, PERIOD_A], goal="test")
        assert "Sentadilla" in prompt

    def test_without_history(self):
        prompt = build_new_routine_prompt([PERIOD_A], goal="test")
        assert "history" in prompt


# ---------------------------------------------------------------------------
# build_weekly_prompt
# ---------------------------------------------------------------------------

class TestBuildWeeklyPrompt:
    def _week_data(self):
        return [{"day": 1, "exercises": [{"name": "Sentadilla", "series": [_make_series()]}]}]

    def test_contains_period_name(self):
        prompt = build_weekly_prompt(PERIOD_A, self._week_data(), None, 1, "test")
        assert PERIOD_A["period"] in prompt

    def test_contains_goal(self):
        prompt = build_weekly_prompt(PERIOD_A, self._week_data(), None, 1, "fuerza")
        assert "fuerza" in prompt

    def test_with_prev_week(self):
        prompt = build_weekly_prompt(PERIOD_A, self._week_data(), self._week_data(), 2, "test")
        assert "previous" in prompt.lower()

    def test_without_prev_week_mentions_primera(self):
        prompt = build_weekly_prompt(PERIOD_A, self._week_data(), None, 1, "test")
        assert "first" in prompt.lower()

    def test_contains_exercise_name(self):
        prompt = build_weekly_prompt(PERIOD_A, self._week_data(), None, 1, "test")
        assert "Sentadilla" in prompt


# ---------------------------------------------------------------------------
# analyze — mock mode
# ---------------------------------------------------------------------------

class TestAnalyzeMock:
    def test_mock_global_returns_string(self):
        result = analyze([], api_key="fake", mock=True, mode="global", goal="test")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_mock_weekly_returns_string(self):
        result = analyze([], api_key="fake", mock=True, mode="weekly", goal="test")
        assert isinstance(result, str)

    def test_mock_monthly_returns_string(self):
        result = analyze([], api_key="fake", mock=True, mode="monthly", goal="test")
        assert isinstance(result, str)

    def test_mock_new_routine_returns_string(self):
        result = analyze([], api_key="fake", mock=True, mode="new-routine", goal="test")
        assert isinstance(result, str)

    def test_mock_unknown_mode_falls_back_to_global(self):
        result = analyze([], api_key="fake", mock=True, mode="nonexistent", goal="test")
        # Falls back to global mock output
        assert isinstance(result, str)
        assert len(result) > 10

    def test_mock_contains_mock_label(self):
        result = analyze([], api_key="fake", mock=True, mode="global", goal="test")
        assert "MOCK" in result or "mock" in result.lower()


# ---------------------------------------------------------------------------
# _group_combined_exercises
# ---------------------------------------------------------------------------

class TestGroupCombinedExercises:
    def _ex(self, name, is_comb):
        return {"name": name, "is_comb": is_comb}

    def test_all_isolated_produces_single_item_groups(self):
        exs = [self._ex("A", False), self._ex("B", False), self._ex("C", False)]
        groups = _group_combined_exercises(exs)
        assert len(groups) == 3
        assert all(not g["comb"] for g in groups)
        assert all(len(g["exercises"]) == 1 for g in groups)

    def test_all_combined_produces_one_group(self):
        exs = [self._ex("A", True), self._ex("B", True), self._ex("C", True)]
        groups = _group_combined_exercises(exs)
        assert len(groups) == 1
        assert groups[0]["comb"] is True
        assert len(groups[0]["exercises"]) == 3

    def test_mixed_sequence_splits_correctly(self):
        # [comb, comb] [isolated] [comb, comb, comb] [isolated]
        exs = [
            self._ex("A", True), self._ex("B", True),
            self._ex("C", False),
            self._ex("D", True), self._ex("E", True), self._ex("F", True),
            self._ex("G", False),
        ]
        groups = _group_combined_exercises(exs)
        assert len(groups) == 4
        assert groups[0] == {"comb": True,  "exercises": [exs[0], exs[1]]}
        assert groups[1] == {"comb": False, "exercises": [exs[2]]}
        assert groups[2] == {"comb": True,  "exercises": [exs[3], exs[4], exs[5]]}
        assert groups[3] == {"comb": False, "exercises": [exs[6]]}

    def test_empty_list_returns_empty(self):
        assert _group_combined_exercises([]) == []

    def test_single_combined_exercise_is_its_own_group(self):
        exs = [self._ex("X", True)]
        groups = _group_combined_exercises(exs)
        assert len(groups) == 1
        assert groups[0]["comb"] is True


# ---------------------------------------------------------------------------
# _format_routine_structure — superset rendering
# ---------------------------------------------------------------------------

class TestFormatRoutineStructure:
    def _make_comb_period(self):
        """Day with: [A,B,C comb] + [D isolated] + [E,F comb]."""
        return {
            "period": "01/01/26-31/01/26",
            "days": [{
                "day": 1,
                "exercises": [
                    _make_exercise("Ejercicio A", is_comb=True),
                    _make_exercise("Ejercicio B", is_comb=True),
                    _make_exercise("Ejercicio C", is_comb=True),
                    _make_exercise("Ejercicio D", is_comb=False),
                    _make_exercise("Ejercicio E", is_comb=True),
                    _make_exercise("Ejercicio F", is_comb=True),
                ]
            }]
        }

    def test_superset_header_contains_all_exercise_names(self):
        result = _format_routine_structure(self._make_comb_period())
        assert "Ejercicio A + Ejercicio B + Ejercicio C" in result

    def test_rest_here_marker_on_last_of_group(self):
        result = _format_routine_structure(self._make_comb_period())
        lines = result.split("\n")
        # Exercise lines start with "#N" — the header line says "Superset:"
        c_line = next(l for l in lines if "Ejercicio C" in l and "#" in l)
        assert "← REST HERE" in c_line
        # Ejercicio A and B must NOT have the marker
        a_line = next(l for l in lines if "Ejercicio A" in l and "#" in l)
        b_line = next(l for l in lines if "Ejercicio B" in l and "#" in l)
        assert "← REST HERE" not in a_line
        assert "← REST HERE" not in b_line

    def test_isolated_exercise_has_no_rest_marker(self):
        result = _format_routine_structure(self._make_comb_period())
        lines = result.split("\n")
        d_line = next(l for l in lines if "Ejercicio D" in l)
        assert "← REST HERE" not in d_line
        assert "Superset" not in d_line

    def test_position_numbers_are_continuous(self):
        result = _format_routine_structure(self._make_comb_period())
        for i in range(1, 7):
            assert f"#{i} " in result

    def test_two_independent_superset_blocks(self):
        result = _format_routine_structure(self._make_comb_period())
        # Both superset headers should appear
        assert "Superset: Ejercicio A" in result
        assert "Superset: Ejercicio E + Ejercicio F" in result

    def test_superset_indented_deeper_than_header(self):
        result = _format_routine_structure(self._make_comb_period())
        lines = result.split("\n")
        header_line = next(l for l in lines if "Superset: Ejercicio A" in l)
        ex_line     = next(l for l in lines if "#1 Ejercicio A" in l)
        # Exercise line should start with more spaces than header line
        header_indent = len(header_line) - len(header_line.lstrip())
        ex_indent     = len(ex_line)     - len(ex_line.lstrip())
        assert ex_indent > header_indent

    def test_day_with_no_combined_exercises(self):
        period = {
            "period": "01/01/26-31/01/26",
            "days": [{"day": 1, "exercises": [
                _make_exercise("Solo A"), _make_exercise("Solo B"),
            ]}]
        }
        result = _format_routine_structure(period)
        assert "Superset" not in result
        assert "#1 Solo A" in result
        assert "#2 Solo B" in result

