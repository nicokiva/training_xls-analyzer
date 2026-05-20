"""
Tests for helpers/ai/ai.py — pure/prompt-building functions and mock mode.
"""
import pytest
from helpers.ai import (
    _format_exercise_history,
    build_global_prompt,
    build_new_routine_prompt,
    build_weekly_prompt,
    analyze,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_series(reps="10", peso="60"):
    return {"reps": reps, "peso": peso}


def _make_week(week_num, has_data=True):
    if has_data:
        return {"week": week_num, "series": [_make_series() for _ in range(3)]}
    return {"week": week_num, "series": [{"reps": "", "peso": ""} for _ in range(3)]}


def _make_exercise(name, weeks_with_data=(1,)):
    weeks = [_make_week(w, has_data=(w in weeks_with_data)) for w in range(1, 5)]
    return {"name": name, "weeks": weeks}


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
        assert "Sentadilla" in prompt
        assert "Press plano" in prompt

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
        assert "no previous history" in prompt


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
