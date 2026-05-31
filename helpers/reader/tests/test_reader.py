"""
Tests for helpers/reader/reader.py — pure functions only (no I/O).
"""
import pytest
from helpers.reader import parse_tab, get_latest_week_indices, extract_week_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(reps="", peso=""):
    return {"reps": reps, "peso": peso}


def _make_week(week_num, series_data):
    """series_data: list of (reps, peso) tuples."""
    return {
        "week": week_num,
        "series": [_make_series(r, p) for r, p in series_data],
    }


def _make_exercise(name, weeks):
    return {"name": name, "weeks": weeks}


def _make_period(period_name, days):
    return {"period": period_name, "days": days}


# ---------------------------------------------------------------------------
# parse_tab
# ---------------------------------------------------------------------------

class TestParseTab:
    def _build_rows(self):
        """Build a minimal valid tab with 1 day, 1 exercise, 4 weeks × 3 series."""
        # Row 0: "Dia 1"
        # Row 1: series numbers (ignored)
        # Row 2: "Rep. Peso" labels (ignored)
        # Row 3: exercise row
        ex_row = ["Sentadilla"]
        # 4 weeks × 3 series × 2 cols = 24 values
        for w in range(4):
            for s in range(3):
                ex_row.append(str(10 + w))   # reps
                ex_row.append(str(60 + w * 5))  # peso
        return [
            ["Dia 1"],
            ["", "1", "", "1", "", "1", "", "2", "", "2", "", "2", "", "3", "", "3", "", "3", "", "4", "", "4", "", "4", ""],
            ["", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso",
             "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso"],
            ex_row,
        ]

    def test_parses_day_number(self):
        rows = self._build_rows()
        days = parse_tab(rows)
        assert len(days) == 1
        assert days[0]["day"] == 1

    def test_parses_exercise_name(self):
        rows = self._build_rows()
        days = parse_tab(rows)
        assert days[0]["exercises"][0]["name"] == "Sentadilla"

    def test_parses_4_weeks(self):
        rows = self._build_rows()
        days = parse_tab(rows)
        ex = days[0]["exercises"][0]
        assert len(ex["weeks"]) == 4

    def test_parses_3_series_per_week(self):
        rows = self._build_rows()
        days = parse_tab(rows)
        ex = days[0]["exercises"][0]
        for w in ex["weeks"]:
            assert len(w["series"]) == 3

    def test_parses_reps_and_peso_correctly(self):
        rows = self._build_rows()
        days = parse_tab(rows)
        ex = days[0]["exercises"][0]
        # Week 0 (index 0): reps = "10", peso = "60"
        assert ex["weeks"][0]["series"][0]["reps"] == "10"
        assert ex["weeks"][0]["series"][0]["peso"] == "60"
        # Week 1 (index 1): reps = "11", peso = "65"
        assert ex["weeks"][1]["series"][0]["reps"] == "11"
        assert ex["weeks"][1]["series"][0]["peso"] == "65"

    def test_empty_rows_returns_empty(self):
        assert parse_tab([]) == []

    def test_multiple_days(self):
        rows = self._build_rows()
        rows.append([])  # blank separator
        # Second day
        ex_row2 = ["Press plano"] + ["8", "80"] * 12
        rows += [
            ["Dia 2"],
            ["", "1"],
            ["", "Rep.", "Peso"],
            ex_row2,
        ]
        days = parse_tab(rows)
        assert len(days) == 2
        assert days[1]["day"] == 2
        assert days[1]["exercises"][0]["name"] == "Press plano"

    def test_week_numbers_are_1_based(self):
        rows = self._build_rows()
        days = parse_tab(rows)
        ex = days[0]["exercises"][0]
        for i, w in enumerate(ex["weeks"]):
            assert w["week"] == i + 1


# ---------------------------------------------------------------------------
# get_latest_week_indices
# ---------------------------------------------------------------------------

class TestGetLatestWeekIndices:
    def test_returns_none_none_when_no_data(self):
        period = _make_period("test", [
            {"day": 1, "exercises": [
                _make_exercise("X", [
                    _make_week(1, [("", ""), ("", ""), ("", "")]),
                    _make_week(2, [("", ""), ("", ""), ("", "")]),
                ])
            ]}
        ])
        assert get_latest_week_indices(period) == (None, None)

    def test_returns_0_none_when_only_week1_has_data(self):
        period = _make_period("test", [
            {"day": 1, "exercises": [
                _make_exercise("X", [
                    _make_week(1, [("10", "60"), ("8", "60"), ("8", "60")]),
                    _make_week(2, [("", ""), ("", ""), ("", "")]),
                    _make_week(3, [("", ""), ("", ""), ("", "")]),
                    _make_week(4, [("", ""), ("", ""), ("", "")]),
                ])
            ]}
        ])
        assert get_latest_week_indices(period) == (0, None)

    def test_returns_1_0_when_weeks_1_and_2_have_data(self):
        period = _make_period("test", [
            {"day": 1, "exercises": [
                _make_exercise("X", [
                    _make_week(1, [("10", "60"), ("8", "60"), ("8", "60")]),
                    _make_week(2, [("9", "62"), ("9", "62"), ("9", "62")]),
                    _make_week(3, [("", ""), ("", ""), ("", "")]),
                    _make_week(4, [("", ""), ("", ""), ("", "")]),
                ])
            ]}
        ])
        assert get_latest_week_indices(period) == (1, 0)

    def test_returns_3_2_when_all_weeks_have_data(self):
        period = _make_period("test", [
            {"day": 1, "exercises": [
                _make_exercise("X", [
                    _make_week(1, [("10", "60"), ("8", "60"), ("8", "60")]),
                    _make_week(2, [("9", "62"), ("9", "62"), ("9", "62")]),
                    _make_week(3, [("8", "65"), ("8", "65"), ("8", "65")]),
                    _make_week(4, [("7", "67"), ("7", "67"), ("7", "67")]),
                ])
            ]}
        ])
        assert get_latest_week_indices(period) == (3, 2)

    def test_empty_days(self):
        period = _make_period("test", [])
        assert get_latest_week_indices(period) == (None, None)

    def test_ongoing_week_is_skipped(self):
        # W1 complete (both days have data), W2 only day 1 has data → ongoing.
        # Should return W1 as current, not W2.
        period = _make_period("test", [
            {"day": 1, "exercises": [
                _make_exercise("X", [
                    _make_week(1, [("10", "60"), ("8", "60"), ("8", "60")]),
                    _make_week(2, [("9", "62"), ("9", "62"), ("9", "62")]),
                ])
            ]},
            {"day": 2, "exercises": [
                _make_exercise("Y", [
                    _make_week(1, [("10", "50"), ("8", "50"), ("8", "50")]),
                    _make_week(2, [("",   ""),   ("",  ""),   ("",  "")]),
                ])
            ]},
        ])
        assert get_latest_week_indices(period) == (0, None)

    def test_both_weeks_complete_with_multiple_days(self):
        period = _make_period("test", [
            {"day": 1, "exercises": [
                _make_exercise("X", [
                    _make_week(1, [("10", "60")]),
                    _make_week(2, [("9",  "62")]),
                ])
            ]},
            {"day": 2, "exercises": [
                _make_exercise("Y", [
                    _make_week(1, [("10", "50")]),
                    _make_week(2, [("9",  "52")]),
                ])
            ]},
        ])
        assert get_latest_week_indices(period) == (1, 0)


# ---------------------------------------------------------------------------
# extract_week_data
# ---------------------------------------------------------------------------

class TestExtractWeekData:
    def _make_full_period(self):
        return _make_period("test", [
            {"day": 1, "exercises": [
                _make_exercise("Sentadilla", [
                    _make_week(1, [("10", "60"), ("10", "60"), ("10", "60")]),
                    _make_week(2, [("9", "62"), ("9", "62"), ("9", "62")]),
                    _make_week(3, [("", ""), ("", ""), ("", "")]),
                    _make_week(4, [("", ""), ("", ""), ("", "")]),
                ]),
                _make_exercise("Press plano", [
                    _make_week(1, [("", ""), ("", ""), ("", "")]),
                    _make_week(2, [("8", "80"), ("8", "80"), ("8", "80")]),
                    _make_week(3, [("", ""), ("", ""), ("", "")]),
                    _make_week(4, [("", ""), ("", ""), ("", "")]),
                ]),
            ]}
        ])

    def test_extracts_exercises_with_data_in_week(self):
        period = self._make_full_period()
        result = extract_week_data(period, 0)  # week index 0 = week 1
        assert len(result) == 1
        assert result[0]["day"] == 1
        assert len(result[0]["exercises"]) == 1
        assert result[0]["exercises"][0]["name"] == "Sentadilla"

    def test_both_exercises_in_week_1_index(self):
        period = self._make_full_period()
        result = extract_week_data(period, 1)  # week index 1 = week 2
        exercises = result[0]["exercises"]
        names = [e["name"] for e in exercises]
        assert "Sentadilla" in names
        assert "Press plano" in names

    def test_empty_when_no_data_in_week(self):
        period = self._make_full_period()
        result = extract_week_data(period, 2)  # week 3 — no data
        assert result == []

    def test_series_data_is_correct(self):
        period = self._make_full_period()
        result = extract_week_data(period, 0)
        series = result[0]["exercises"][0]["series"]
        assert series[0]["reps"] == "10"
        assert series[0]["peso"] == "60"

    def test_out_of_range_week_idx_returns_empty(self):
        period = self._make_full_period()
        result = extract_week_data(period, 10)
        assert result == []


# ---------------------------------------------------------------------------
# Drop set notation in parse_tab
# ---------------------------------------------------------------------------

class TestDropSetParsing:
    def _build_rows_with_drop_set(self, peso_value):
        ex_row = ["Curl de bíceps"]
        for w in range(4):
            for s in range(3):
                ex_row.append("8")
                ex_row.append(peso_value if (w == 0 and s == 0) else "30")
        return [
            ["Dia 1"],
            ["", "1", "", "1", "", "1", "", "2", "", "2", "", "2", "", "3", "", "3", "", "3", "", "4", "", "4", "", "4", ""],
            ["", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso",
             "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso", "Rep.", "Peso"],
            ex_row,
        ]

    def test_drop_set_stores_full_sequence(self):
        rows = self._build_rows_with_drop_set("8 a 42.5 / 4 a 37.5 / 4 a 32.5")
        days = parse_tab(rows)
        s = days[0]["exercises"][0]["weeks"][0]["series"][0]
        assert s["drop_set"] == "8 a 42.5 / 4 a 37.5 / 4 a 32.5"

    def test_drop_set_extracts_starting_weight_as_peso(self):
        rows = self._build_rows_with_drop_set("8 a 42.5 / 4 a 37.5")
        days = parse_tab(rows)
        s = days[0]["exercises"][0]["weeks"][0]["series"][0]
        assert s["peso"] == "42.5"

    def test_normal_peso_has_no_drop_set_key(self):
        rows = self._build_rows_with_drop_set("30")
        days = parse_tab(rows)
        s = days[0]["exercises"][0]["weeks"][0]["series"][0]
        assert "drop_set" not in s
        assert s["peso"] == "30"

    def test_drop_set_peso_is_float_parseable(self):
        rows = self._build_rows_with_drop_set("8 a 10 / 8 a 8 / 8 a 6")
        days = parse_tab(rows)
        s = days[0]["exercises"][0]["weeks"][0]["series"][0]
        assert float(s["peso"]) == 10.0
