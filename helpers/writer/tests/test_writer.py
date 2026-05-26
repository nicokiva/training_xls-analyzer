"""
Tests for helpers/writer/writer.py — pure functions and sheet-write logic.

The write_suggestions_to_sheet function calls the Google Sheets API, so we
use a fake service object instead of a real network call. The fake captures
what batchUpdate() would have written so we can assert on it.
"""
import pytest
from unittest.mock import MagicMock, call

from helpers.writer import (
    parse_suggestions,
    strip_suggestions_block,
    format_suggestions_for_email,
)
from helpers.writer.writer import (
    _normalize,
    _col_letter,
    _peso_col,
    write_suggestions_to_sheet,
    PAUSA_COL,
)
from helpers.reader.reader import N_WEEKS, N_SERIES


# ---------------------------------------------------------------------------
# parse_suggestions
# ---------------------------------------------------------------------------

class TestParseSuggestions:
    def _wrap(self, payload):
        return f"Some prose.\n```json\n{payload}\n```\nMore prose."

    def test_parses_valid_json_list(self):
        json_str = '[{"exercise":"Sentadilla","day":1,"weeks":[60,65,67,70],"rest_s":90,"reason":"base"}]'
        result = parse_suggestions(self._wrap(json_str))
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["exercise"] == "Sentadilla"

    def test_returns_none_when_no_block(self):
        assert parse_suggestions("No JSON here.") is None

    def test_returns_none_on_invalid_json(self):
        assert parse_suggestions("```json\n{bad json\n```") is None

    def test_parses_multiple_suggestions(self):
        json_str = '[{"exercise":"A","day":1,"weeks":[10,11,12,13],"rest_s":60,"reason":"x"},{"exercise":"B","day":2,"weeks":[20,21,22,23],"rest_s":90,"reason":"y"}]'
        result = parse_suggestions(self._wrap(json_str))
        assert len(result) == 2
        assert result[1]["exercise"] == "B"
        assert result[1]["day"] == 2

    def test_parses_float_weights(self):
        json_str = '[{"exercise":"X","day":1,"weeks":[7.5,10,12.5,15],"rest_s":60,"reason":"r"}]'
        result = parse_suggestions(self._wrap(json_str))
        assert result[0]["weeks"][0] == 7.5

    def test_parses_null_weights(self):
        json_str = '[{"exercise":"X","day":1,"weeks":[null,10,12,15],"rest_s":60,"reason":"r"}]'
        result = parse_suggestions(self._wrap(json_str))
        assert result[0]["weeks"][0] is None

    def test_empty_json_array(self):
        result = parse_suggestions(self._wrap("[]"))
        assert result == []


# ---------------------------------------------------------------------------
# strip_suggestions_block
# ---------------------------------------------------------------------------

class TestStripSuggestionsBlock:
    def test_removes_json_block(self):
        text = "Prose before.\n```json\n[{\"key\": 1}]\n```\nProse after."
        result = strip_suggestions_block(text)
        assert "```json" not in result
        assert "Prose before." in result
        assert "Prose after." in result

    def test_no_block_returns_unchanged(self):
        text = "No block here."
        assert strip_suggestions_block(text) == text

    def test_strips_and_trims(self):
        text = "```json\n[]\n```"
        assert strip_suggestions_block(text) == ""


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases(self):
        assert _normalize("Sentadilla") == "sentadilla"

    def test_removes_accents(self):
        assert _normalize("Extensión") == "extension"
        assert _normalize("Rotación") == "rotacion"

    def test_removes_parentheses(self):
        assert _normalize("Press (plano)") == "press"

    def test_strips_whitespace(self):
        assert _normalize("  foo bar  ") == "foo bar"

    def test_collapses_spaces(self):
        assert _normalize("foo   bar") == "foo bar"

    def test_combined_accents_and_parens(self):
        assert _normalize("Extensión de cadera (banco)") == "extension de cadera"


# ---------------------------------------------------------------------------
# _col_letter
# ---------------------------------------------------------------------------

class TestColLetter:
    def test_col_0_is_A(self):
        assert _col_letter(0) == "A"

    def test_col_25_is_Z(self):
        assert _col_letter(25) == "Z"

    def test_col_26_is_AA(self):
        assert _col_letter(26) == "AA"

    def test_col_51_is_AZ(self):
        assert _col_letter(51) == "AZ"

    def test_col_1_is_B(self):
        assert _col_letter(1) == "B"


# ---------------------------------------------------------------------------
# _peso_col
# ---------------------------------------------------------------------------

class TestPesoCol:
    def test_week0_set0(self):
        # Week 0, set 0: col = 1 + 0*(6) + 0*2 + 1 = 2
        assert _peso_col(0, 0) == 2

    def test_week0_set1(self):
        # Week 0, set 1: col = 1 + 0 + 2 + 1 = 4
        assert _peso_col(0, 1) == 4

    def test_week1_set0(self):
        # Week 1, set 0: col = 1 + 1*(3*2) + 0 + 1 = 1 + 6 + 1 = 8
        assert _peso_col(1, 0) == 8

    def test_pausa_col_is_25(self):
        # Pausa is always col Z (index 25) = N_WEEKS * N_SERIES * 2 + 1
        assert PAUSA_COL == N_WEEKS * N_SERIES * 2 + 1
        assert PAUSA_COL == 25


# ---------------------------------------------------------------------------
# format_suggestions_for_email
# ---------------------------------------------------------------------------

class TestFormatSuggestionsForEmail:
    def _make_suggestion(self, name="Sentadilla", day=1, weeks=None, rest_s=90):
        return {
            "exercise": name,
            "day": day,
            "weeks": weeks or [60, 65, 67, 70],
            "rest_s": rest_s,
            "reason": "Progressive overload",
        }

    def test_empty_returns_empty_string(self):
        assert format_suggestions_for_email([]) == ""

    def test_contains_exercise_name(self):
        result = format_suggestions_for_email([self._make_suggestion()])
        assert "Sentadilla" in result

    def test_contains_day_header(self):
        result = format_suggestions_for_email([self._make_suggestion(day=2)])
        assert "Día 2" in result

    def test_contains_rest_time(self):
        result = format_suggestions_for_email([self._make_suggestion(rest_s=120)])
        assert "120s" in result

    def test_groups_by_day(self):
        suggestions = [
            self._make_suggestion("A", day=1),
            self._make_suggestion("B", day=2),
            self._make_suggestion("C", day=1),
        ]
        result = format_suggestions_for_email(suggestions)
        assert "Día 1" in result
        assert "Día 2" in result
        # Day 1 should appear before Day 2
        assert result.index("Día 1") < result.index("Día 2")

    def test_null_rest_shows_dash(self):
        s = self._make_suggestion(rest_s=None)
        s["rest_s"] = None
        result = format_suggestions_for_email([s])
        assert "—" in result

    def test_combined_exercise_gets_comb_class(self):
        s = self._make_suggestion()
        s["is_comb"] = True
        result = format_suggestions_for_email([s])
        assert 'class="comb"' in result

    def test_weights_shown_in_kg(self):
        s = self._make_suggestion(weeks=[50, 55, 57, 60])
        result = format_suggestions_for_email([s])
        assert "50 kg" in result
        assert "55 kg" in result


# ---------------------------------------------------------------------------
# write_suggestions_to_sheet — day-aware lookup
# ---------------------------------------------------------------------------

def _make_fake_service(tab_name, rows, sheet_id=1):
    """
    Builds a fake Google Sheets service that returns `rows` for any get() call
    and captures batchUpdate() calls for assertions.

    Returns (service, captured) where captured["value_updates"] and
    captured["format_requests"] accumulate what was passed to batchUpdate().
    """
    captured = {"value_updates": [], "format_requests": []}

    def _values_get(**kwargs):
        mock = MagicMock()
        mock.execute.return_value = {"values": rows}
        return mock

    def _values_batch_update(spreadsheetId, body):
        captured["value_updates"].extend(body.get("data", []))
        mock = MagicMock()
        mock.execute.return_value = {}
        return mock

    def _spreadsheets_batch_update(spreadsheetId, body):
        captured["format_requests"].extend(body.get("requests", []))
        mock = MagicMock()
        mock.execute.return_value = {}
        return mock

    def _spreadsheets_get(spreadsheetId):
        mock = MagicMock()
        mock.execute.return_value = {
            "sheets": [{"properties": {"title": tab_name, "sheetId": sheet_id}}]
        }
        return mock

    values_mock = MagicMock()
    values_mock.get.side_effect = lambda **kw: _values_get(**kw)
    values_mock.batchUpdate.side_effect = _values_batch_update

    ss_mock = MagicMock()
    ss_mock.get.side_effect = _spreadsheets_get
    ss_mock.values.return_value = values_mock
    ss_mock.batchUpdate.side_effect = _spreadsheets_batch_update

    service = MagicMock()
    service.spreadsheets.return_value = ss_mock

    return service, captured


def _empty_row(name, n_cols=26):
    """An exercise row: col A = name, rest empty."""
    row = [name] + [""] * (n_cols - 1)
    return row


def _filled_row(name, peso_val, n_cols=26):
    """An exercise row with week-1 set-0 already filled."""
    row = _empty_row(name, n_cols)
    row[_peso_col(0, 0)] = str(peso_val)
    return row


class TestWriteSuggestionsToSheet:
    TAB = "25/05/26-..."

    def _rows(self):
        """
        Minimal sheet with 2 days, each with 2 exercises.
        Day 1: Sentadilla, Press plano
        Day 2: Sentadilla (same name, different day), Dominada
        """
        return [
            ["Dia 1"],
            ["", "1", "", "2", "", "3", "", "4", ""],
            ["", "Rep.", "Peso", "Rep.", "Peso"],
            _empty_row("Sentadilla"),
            _empty_row("Press plano"),
            [],
            ["Dia 2"],
            ["", "1"],
            ["", "Rep.", "Peso"],
            _empty_row("Sentadilla"),   # same name, different day
            _empty_row("Dominada"),
        ]

    def test_writes_to_correct_day(self):
        """Suggestions for day=1 and day=2 must land in the right rows."""
        rows = self._rows()
        service, captured = _make_fake_service(self.TAB, rows)

        suggestions = [
            {"exercise": "Sentadilla", "day": 1, "weeks": [60, 65, 67, 70], "rest_s": None},
            {"exercise": "Sentadilla", "day": 2, "weeks": [50, 55, 57, 60], "rest_s": None},
        ]
        write_suggestions_to_sheet(service, "fake-id", self.TAB, suggestions)

        written_values = [u["values"][0][0] for u in captured["value_updates"]]
        # Day 1 Sentadilla → 60kg; Day 2 Sentadilla → 50kg
        assert "60" in written_values
        assert "50" in written_values

    def test_skips_orig_tabs(self):
        service, captured = _make_fake_service("ORIG18/05/26-...", [])
        write_suggestions_to_sheet(service, "fake-id", "ORIG18/05/26-...", [
            {"exercise": "X", "day": 1, "weeks": [10], "rest_s": None}
        ])
        assert captured["value_updates"] == []

    def test_skips_already_filled_cells(self):
        rows = self._rows()
        # Pre-fill week-1 set-0 for Sentadilla day 1
        rows[3] = _filled_row("Sentadilla", 70)

        service, captured = _make_fake_service(self.TAB, rows)
        suggestions = [
            {"exercise": "Sentadilla", "day": 1, "weeks": [60, 65, 67, 70], "rest_s": None},
        ]
        write_suggestions_to_sheet(service, "fake-id", self.TAB, suggestions)

        # The pre-filled cell (week 1, set 0) should NOT be overwritten
        written_ranges = [u["range"] for u in captured["value_updates"]]
        peso_col_0_0 = _col_letter(_peso_col(0, 0))
        assert not any(peso_col_0_0 + "4" in r for r in written_ranges)

    def test_writes_pausa_for_non_combined(self):
        rows = self._rows()
        service, captured = _make_fake_service(self.TAB, rows)

        suggestions = [
            {"exercise": "Press plano", "day": 1, "weeks": [60, 65, 67, 70], "rest_s": 90},
        ]
        write_suggestions_to_sheet(service, "fake-id", self.TAB, suggestions)

        pausa_writes = [u for u in captured["value_updates"] if u["values"][0][0] == "90s"]
        assert len(pausa_writes) == 1

    def test_skips_pausa_for_non_last_combined(self):
        """Non-last [C] exercise must not get a Pausa cell."""
        rows = [
            ["Dia 1"],
            [],
            [],
            _empty_row("[C] Abdominal"),
            _empty_row("[C] Rotaciones"),
            _empty_row("[C] Extension de cadera"),  # last of group
        ]
        service, captured = _make_fake_service(self.TAB, rows)

        suggestions = [
            {"exercise": "Abdominal",          "day": 1, "weeks": [0]*4, "rest_s": None},
            {"exercise": "Rotaciones",          "day": 1, "weeks": [5]*4, "rest_s": None},
            {"exercise": "Extension de cadera", "day": 1, "weeks": [10]*4, "rest_s": 60},
        ]
        write_suggestions_to_sheet(service, "fake-id", self.TAB, suggestions)

        pausa_writes = [u for u in captured["value_updates"] if "s" in str(u["values"][0][0])]
        # Only one pausa write (for the last in the combined group)
        assert len(pausa_writes) == 1
        assert pausa_writes[0]["values"][0][0] == "60s"

    def test_unknown_exercise_logs_and_skips(self, capsys):
        service, captured = _make_fake_service(self.TAB, self._rows())
        suggestions = [
            {"exercise": "NonExistent", "day": 1, "weeks": [50]*4, "rest_s": None},
        ]
        write_suggestions_to_sheet(service, "fake-id", self.TAB, suggestions)
        out = capsys.readouterr().out
        assert "not found" in out
        assert captured["value_updates"] == []

    def test_empty_suggestions_does_nothing(self):
        service, captured = _make_fake_service(self.TAB, self._rows())
        write_suggestions_to_sheet(service, "fake-id", self.TAB, [])
        assert captured["value_updates"] == []
