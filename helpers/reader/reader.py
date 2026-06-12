"""
helpers/reader.py — Reading and parsing tabs from the training routines Google Sheet.

Responsibilities:
  - Authenticate with the Google Sheets API (read-only) using a service account.
  - List all tabs in the spreadsheet in position order.
  - Read the raw cells from each tab.
  - Parse those cells into a usable Python data structure.

Structure of each tab in the spreadsheet:
  Each tab represents a training period (e.g. "18/05/26-14/06/26")
  and contains day blocks with the following layout:

    Row 0: "Dia N"          ← day name (e.g. "Dia 1")
    Row 1: 1 "" 1 "" ...    ← set number (1, 2, 3) repeated per week
    Row 2: Rep. Peso ...    ← column labels, 4 weeks × 3 sets × 2 cols
    Row 3+: exercises       ← col A = name, then alternating reps/weight
    (empty row between days)

  Data columns: 4 weeks × 3 sets × 2 fields (Rep + Peso) = 24 columns
  + 1 name column = 25 total columns (A:Y).
"""

from google.oauth2 import service_account
from googleapiclient.discovery import build

# Read-only permission — sufficient to read the spreadsheet
SCOPES_READ  = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
SCOPES_WRITE = ["https://www.googleapis.com/auth/spreadsheets"]

N_WEEKS = 4    # weeks per period
N_SERIES = 3   # sets per exercise per week


def get_service(credentials_path):
    """Authenticated Google Sheets API client (read-only)."""
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES_READ
    )
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def get_write_service(credentials_path):
    """Authenticated Google Sheets API client (read + write)."""
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES_WRITE
    )
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def list_tabs(service, spreadsheet_id):
    """
    Returns the names of all tabs in the spreadsheet in position order.

    The tab at index 0 is always the most recent (pdf2xls-generator inserts it
    at the front each time it processes a new PDF).

    Args:
        service:        Google Sheets API Resource.
        spreadsheet_id: ID of the spreadsheet (the long part of the Google Sheets URL).

    Returns:
        List of strings with the tab names, e.g.:
        ["18/05/26-14/06/26", "20/04/26-15/05/26", ...]
    """
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    return [s["properties"]["title"] for s in meta["sheets"]]


def read_tab(service, spreadsheet_id, tab_name):
    """
    Reads all cells from a tab and returns them as a list of lists of strings.

    Trailing empty rows are not included by the API. Empty cells within a row
    do appear as empty strings (or may be absent if they are the last in the row).

    Args:
        service:        Google Sheets API Resource.
        spreadsheet_id: ID of the spreadsheet.
        tab_name:       Exact name of the tab to read.

    Returns:
        List of rows, each row is a list of strings.
        E.g.: [["Dia 1", "", ...], ["", "1", "", "1", ...], ...]
    """
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=f"'{tab_name}'")
        .execute()
    )
    return result.get("values", [])


def read_tab_notes(service, spreadsheet_id, tab_name):
    """
    Reads cell notes (not values) for a tab.

    Uses spreadsheets().get() with a fields filter to fetch only notes,
    avoiding the overhead of re-fetching all cell values.

    Args:
        service:        Google Sheets API Resource.
        spreadsheet_id: ID of the spreadsheet.
        tab_name:       Exact name of the tab to read.

    Returns:
        Dict mapping (row_index, col_index) → note text (str).
        Only entries with non-empty notes are included.
        Indices are 0-based and match the row indices from read_tab().
    """
    result = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        ranges=[f"'{tab_name}'"],
        fields="sheets.data.rowData.values.note",
    ).execute()

    notes = {}
    sheets = result.get("sheets", [])
    if not sheets:
        return notes
    data = sheets[0].get("data", [])
    if not data:
        return notes
    for row_idx, row in enumerate(data[0].get("rowData", [])):
        for col_idx, cell in enumerate(row.get("values", [])):
            note = cell.get("note", "").strip()
            if note:
                notes[(row_idx, col_idx)] = note
    return notes


def read_tab_italic_cells(service, spreadsheet_id, tab_name):
    """
    Returns a set of (row_idx, col_idx) for cells that have italic formatting.

    Used to detect AI-suggested peso values written by the writer module.
    Suggested cells are formatted in italic and should be treated as empty
    (not real training data) when building AI prompts.

    Indices are 0-based and match those from read_tab().
    """
    result = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        ranges=[f"'{tab_name}'"],
        fields="sheets.data.rowData.values.userEnteredFormat.textFormat.italic",
    ).execute()

    italic = set()
    sheets = result.get("sheets", [])
    if not sheets:
        return italic
    data = sheets[0].get("data", [])
    if not data:
        return italic
    for row_idx, row in enumerate(data[0].get("rowData", [])):
        for col_idx, cell in enumerate(row.get("values", [])):
            fmt = cell.get("userEnteredFormat", {})
            if fmt.get("textFormat", {}).get("italic"):
                italic.add((row_idx, col_idx))
    return italic


def read_tab_metadata(service, spreadsheet_id, tab_name):
    """
    Fetches notes, italic formatting, yellow-background cells, and Z-column
    merge data for a tab in a single API call.

    Combined exercises are marked in two ways:
      1. Exercise name starts with "[C] " (new convention).
      2. Exercise name cell (col 0) has a yellow background — rgb ≈ (1, 0.95, 0.8)
         (older convention used before the [C] prefix was introduced).

    Z-column merges encode comb_group boundaries: the sheet writer (pdf2xls-generator
    sheets.py) merges the Z-column cells for every exercise in the same combined group
    into a single merged range. Reading these ranges lets us reconstruct which exercises
    belong together — even when two adjacent groups are back-to-back with no gap.

    Returns:
        Tuple (notes, italic_cells, yellow_rows, comb_group_map) where:
          notes:          Dict mapping (row_idx, col_idx) → note text (str).
          italic_cells:   Set of (row_idx, col_idx) for italic-formatted cells.
          yellow_rows:    Set of row_idx where col 0 has a yellow background.
          comb_group_map: Dict mapping row_idx → comb_group_id (int) for rows
                          that are part of a multi-row Z-column merge.
                          Empty dict for tabs generated before merge data existed.
    """
    result = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        ranges=[f"'{tab_name}'"],
        fields=(
            "sheets.data.rowData.values"
            "(note,userEnteredFormat.textFormat.italic,userEnteredFormat.backgroundColor),"
            "sheets.merges"
        ),
    ).execute()

    notes  = {}
    italic = set()
    yellow_rows: set = set()
    sheets = result.get("sheets", [])
    if not sheets:
        return notes, italic, yellow_rows, {}
    data = sheets[0].get("data", [])
    if not data:
        return notes, italic, yellow_rows, {}
    for row_idx, row in enumerate(data[0].get("rowData", [])):
        for col_idx, cell in enumerate(row.get("values", [])):
            note = cell.get("note", "").strip()
            if note:
                notes[(row_idx, col_idx)] = note
            fmt = cell.get("userEnteredFormat", {})
            if fmt.get("textFormat", {}).get("italic"):
                italic.add((row_idx, col_idx))
            # Detect yellow background on the exercise-name cell (col 0).
            # Yellow (combined marker): high red, high green, noticeably lower blue.
            if col_idx == 0:
                bg = fmt.get("backgroundColor", {})
                r = bg.get("red",   1.0)
                g = bg.get("green", 1.0)
                b = bg.get("blue",  1.0)
                if r >= 0.95 and g >= 0.88 and b <= 0.85:
                    yellow_rows.add(row_idx)

    comb_group_map = _build_comb_group_map(sheets[0].get("merges", []))
    return notes, italic, yellow_rows, comb_group_map


# Z-column (Pausa) index — matches PAUSA_COL in pdf2xls-generator/helpers/sheets/sheets.py.
_PAUSA_COL = 25


def _build_comb_group_map(merges: list) -> dict:
    """
    Build a row_index → comb_group_id map from Sheets API merge data.

    The sheet writer merges all Z-column cells belonging to the same combined
    group into a single GridRange. A merge spanning N > 1 rows means those N
    exercise rows form one combined group. Single-row "merges" are just isolated
    exercises with Z-cell formatting — they are ignored.

    Group IDs are sequential integers assigned in top-to-bottom order
    (the topmost group gets id=0, the next gets id=1, …).

    Args:
        merges: List of GridRange dicts from sheets[0]["merges"] (Sheets API).

    Returns:
        Dict {row_idx: group_id} for every row inside a multi-row Z-column merge.
        Empty dict when no Z-column multi-row merges exist (old tabs, or tabs
        generated without the comb_group writer).
    """
    group_map: dict = {}
    # Keep only multi-row merges in the Z-column (startCol=25, endCol=26).
    z_merges = [
        m for m in merges
        if m.get("startColumnIndex") == _PAUSA_COL
        and m.get("endColumnIndex") == _PAUSA_COL + 1
        and (m.get("endRowIndex", 0) - m.get("startRowIndex", 0)) > 1
    ]
    # Sort top-to-bottom so IDs are assigned in document order.
    z_merges.sort(key=lambda m: m["startRowIndex"])
    for group_id, merge in enumerate(z_merges):
        for row in range(merge["startRowIndex"], merge["endRowIndex"]):
            group_map[row] = group_id
    return group_map


def _parse_drop_set_start(raw: str) -> str:
    """
    Extracts the starting weight from a drop set notation like "8 a 42.5 / 4 a 37.5".
    Pattern: "<reps> a <weight> / ..." — weight follows "a ".
    Returns the weight as a string, or the original string if unparseable.
    """
    import re
    # Look for " a <number>" pattern — weight follows "a "
    m = re.search(r"\ba\s+([\d]+(?:[.,][\d]+)?)", raw)
    if m:
        return m.group(1).replace(",", ".")
    # Fallback: first number found
    m = re.search(r"[\d]+(?:[.,][\d]+)?", raw)
    if m:
        return m.group(0).replace(",", ".")
    return raw


def parse_tab(rows, notes=None, italic_cells=None, yellow_rows=None, comb_group_map=None):
    """
    Parses the raw rows from a tab and returns a structured list of days.

    Iterates through the rows looking for day headers ("Dia N"), then reads the
    exercises that follow until an empty row or another header is found.
    For each exercise extracts the reps and weights for 4 weeks × 3 sets.

    Args:
        rows:           List of raw rows (output of read_tab).
        notes:          Optional dict {(row_idx, col_idx): note_text}.
                        Notes on column A are attached to the exercise as "note".
        italic_cells:   Optional set of (row_idx, col_idx). Italic peso cells are
                        treated as empty (AI suggestions, not real training data).
        yellow_rows:    Optional set of row_idx for combined exercises detected by
                        yellow background (older sheet convention).
        comb_group_map: Optional dict {row_idx: group_id} from _build_comb_group_map().
                        When provided, assigns "comb_group" to each combined exercise
                        so adjacent groups in the same day can be distinguished.

    Returns:
        List of days. Each exercise dict has:
          "name"       str   — exercise name (without "[C] " prefix)
          "is_comb"    bool  — True if this exercise is part of a combined group
          "comb_group" int   — group id (only present when comb_group_map is given)
          "note"       str   — coach note (only present when a note exists)
          "weeks"      list  — 4 weeks × 3 series × {"reps", "peso"}
    """
    if notes is None:
        notes = {}
    if yellow_rows is None:
        yellow_rows = set()
    if comb_group_map is None:
        comb_group_map = {}
    days = []
    i = 0

    while i < len(rows):
        row = rows[i]
        if not row:
            i += 1
            continue

        first_cell = row[0].strip() if row[0] else ""

        # Detect day header ("Dia 1", "Dia 2", etc.)
        if first_cell.lower().startswith("dia"):
            try:
                day_num = int(first_cell.split()[-1])
            except ValueError:
                i += 1
                continue

            # Skip the 2 header rows that follow "Dia N":
            # - row of set numbers (1, "", 1, "", ..., 2, "", ...)
            # - row of labels (Rep., Peso, Rep., Peso, ...)
            i += 3
            exercises = []

            while i < len(rows):
                ex_row = rows[i]

                # An empty row or one with empty col A indicates end of this day's block
                if not ex_row or not ex_row[0]:
                    i += 1
                    break

                # If the next row is another "Dia N", stop without advancing
                if ex_row[0].strip().lower().startswith("dia"):
                    break

                # Read the exercise: name + 4 weeks × 3 sets × (reps, weight)
                # Combined exercises are detected in two ways:
                #   1. Name prefixed with "[C] " (current convention).
                #   2. Name cell has a yellow background (older convention).
                raw_name = ex_row[0].strip()
                is_comb  = raw_name.startswith("[C] ") or (i in yellow_rows)
                name     = raw_name[4:] if raw_name.startswith("[C] ") else raw_name

                # Note on column A = coach instruction or general observation for this exercise.
                ex_note = notes.get((i, 0), "").strip()

                weeks = []
                col = 1  # data columns start at col 1 (B)

                for w in range(N_WEEKS):
                    series = []
                    for s in range(N_SERIES):
                        # Access with fallback to "" if the row is shorter than expected
                        reps  = ex_row[col].strip() if col < len(ex_row) else ""
                        # Treat italic peso cells as empty — they are AI suggestions,
                        # not real training data entered by Nicolás.
                        peso_raw = ex_row[col + 1].strip() if (col + 1) < len(ex_row) else ""
                        peso = "" if (italic_cells and (i, col + 1) in italic_cells) else peso_raw
                        # Note on a peso cell = observation about how that specific set went.
                        set_note = notes.get((i, col + 1), "").strip()
                        entry = {"reps": reps, "peso": peso}
                        # Detect drop set notation: "8 a 42.5 / 4 a 37.5 / ..."
                        # means consecutive mini-sets with no rest (drop set).
                        # Store the full sequence and set peso to the starting weight.
                        if "/" in peso:
                            entry["drop_set"] = peso
                            entry["peso"] = _parse_drop_set_start(peso)
                        if set_note:
                            entry["note"] = set_note
                        series.append(entry)
                        col += 2  # advance 2 columns (Rep + Peso)
                    weeks.append({"week": w + 1, "series": series})

                ex_dict = {"name": name, "is_comb": is_comb, "weeks": weeks}
                # Attach comb_group id when the Z-column merge map has data for this row.
                # This lets the prompt builder split adjacent combined groups correctly.
                if i in comb_group_map:
                    ex_dict["comb_group"] = comb_group_map[i]
                if ex_note:
                    ex_dict["note"] = ex_note
                exercises.append(ex_dict)
                i += 1

            days.append({"day": day_num, "exercises": exercises})
        else:
            i += 1

    return days


def get_latest_week_indices(period):
    """
    Detects which weeks have data loaded in the period and returns the
    indices (0-based) of the current (last completed) and previous weeks.

    A week is considered "complete" when all training days in the period
    have at least one exercise with real peso data for that week.
    If the most recent week with data is incomplete (ongoing), it is treated
    as the current in-progress week and is skipped in favour of the last
    fully completed one.

    Args:
        period: Dict with format {"period": str, "days": [...]}.

    Returns:
        Tuple (current_idx, prev_idx) with 0-based indices.
        prev_idx is None if the current week is the first completed week.
        Both are None if there is no completed week in the period.
    """
    total_days = len(period["days"])

    # Build a mapping: week_idx → set of day numbers that have real data.
    days_with_data: dict[int, set] = {}
    for day in period["days"]:
        for ex in day["exercises"]:
            for w in ex["weeks"]:
                # Only count a week as "with data" if there is a peso value.
                # Reps alone are pre-filled from the PDF routine structure and
                # don't indicate that the session was actually performed.
                if any(s["peso"] for s in w["series"]):
                    week_idx = w["week"] - 1  # convert to 0-based
                    days_with_data.setdefault(week_idx, set()).add(day["day"])

    if not days_with_data:
        return (None, None)

    # A week is complete when every training day has data.
    completed = sorted(
        idx for idx, days in days_with_data.items()
        if len(days) >= total_days
    )

    if not completed:
        # No fully completed week yet (period just started).
        return (None, None)

    current = completed[-1]
    prev    = completed[-2] if len(completed) >= 2 else None
    return (current, prev)


def extract_week_data(period, week_idx):
    """
    Extracts the data for a specific week from a period.

    Args:
        period:   Dict with format {"period": str, "days": [...]}.
        week_idx: 0-based index of the week to extract.

    Returns:
        List of days with only the data for that week:
        [{"day": N, "exercises": [{"name": str, "series": [...]}]}, ...]
        Only includes exercises that have at least one data point in that week.
    """
    result = []
    for day in period["days"]:
        exercises = []
        for ex in day["exercises"]:
            if week_idx >= len(ex["weeks"]):
                continue
            week = ex["weeks"][week_idx]
            if any(s["peso"] for s in week["series"]):
                exercises.append({"name": ex["name"], "series": week["series"]})
        if exercises:
            result.append({"day": day["day"], "exercises": exercises})
    return result


def load_all_periods(service, spreadsheet_id):
    """
    Loads and parses all tabs from the spreadsheet.
    Returns periods ordered with index 0 = most recent.

    Notes and italic formatting are fetched together in a single API call per tab
    (via read_tab_metadata) to stay within the Sheets API read quota (60 req/min/user).
    Italic peso cells (AI suggestions) are treated as empty so they are never
    mistaken for real training data.
    """
    tabs = list_tabs(service, spreadsheet_id)
    periods = []
    for tab in tabs:
        if tab.upper().startswith("ORIG"):
            continue
        rows                                   = read_tab(service, spreadsheet_id, tab)
        notes, italic_cells, yellow_rows, comb_group_map = read_tab_metadata(service, spreadsheet_id, tab)
        days = parse_tab(
            rows,
            notes=notes,
            italic_cells=italic_cells,
            yellow_rows=yellow_rows,
            comb_group_map=comb_group_map,
        )
        periods.append({"period": tab, "days": days})
    return periods


def is_active_period(period):
    """
    Returns True if this period is still ongoing (its tab name ends with '-...').
    Active periods don't have an end date yet because the routine is still running.
    """
    return period["period"].endswith("-...")


def get_active_period(periods):
    """
    Returns the currently active period (tab ending in '-...'), or None if not found.
    There should normally be exactly one active period.
    """
    for p in periods:
        if is_active_period(p):
            return p
    return None


def get_last_completed_period(periods):
    """
    Returns the most recently completed period (first tab NOT ending in '-...').
    'Most recently completed' = the first one in the list that has a full Fecha-Fecha name.
    Returns None if all periods are active (shouldn't happen in practice).
    """
    for p in periods:
        if not is_active_period(p):
            return p
    return None


def get_all_open_periods(periods):
    """
    Returns all periods whose tab name ends with '-...' (still running), ordered
    as they appear in the spreadsheet (most recent first, same as periods list).

    Normally there is exactly one open period. When a new routine is uploaded before
    the old one is closed, there are temporarily two. The second-to-last (anteúltimo)
    is the one that just finished and should be closed after running monthly/global.
    """
    return [p for p in periods if is_active_period(p)]


def rename_tab(write_service, spreadsheet_id, old_name, new_name):
    """
    Renames a tab in the spreadsheet (requires write scope).
    Used to close a completed period by replacing '-...' with an end date.

    Args:
        write_service:  Google Sheets API client with write scope.
        old_name:       Current tab title, e.g. '18/05/26-...'
        new_name:       New tab title,     e.g. '18/05/26-23/05/26'
    """
    meta = write_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_id = None
    for s in meta["sheets"]:
        if s["properties"]["title"] == old_name:
            sheet_id = s["properties"]["sheetId"]
            break
    if sheet_id is None:
        raise ValueError(f"Tab '{old_name}' not found in spreadsheet.")

    write_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{
            "updateSheetProperties": {
                "properties": {"sheetId": sheet_id, "title": new_name},
                "fields": "title",
            }
        }]},
    ).execute()
    print(f"  Renamed tab: '{old_name}' → '{new_name}'")
