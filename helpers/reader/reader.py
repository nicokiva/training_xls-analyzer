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
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

N_WEEKS = 4    # weeks per period
N_SERIES = 3   # sets per exercise per week


def get_service(credentials_path):
    """
    Creates and returns the authenticated Google Sheets API client.

    Args:
        credentials_path: Path to the Google service account JSON file.

    Returns:
        Google Sheets API v4 Resource.
    """
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES
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


def parse_tab(rows):
    """
    Parses the raw rows from a tab and returns a structured list of days.

    Iterates through the rows looking for day headers ("Dia N"), then reads the
    exercises that follow until an empty row or another header is found.
    For each exercise extracts the reps and weights for 4 weeks × 3 sets.

    Args:
        rows: List of raw rows (output of read_tab).

    Returns:
        List of days with the following format:
        [
          {
            "day": 1,
            "exercises": [
              {
                "name": "Empuje de pecho con barra",
                "weeks": [
                  {
                    "week": 1,
                    "series": [
                      {"reps": "12", "peso": "60"},
                      {"reps": "10", "peso": "60"},
                      {"reps": "10", "peso": "60"},
                    ]
                  },
                  ... (weeks 2, 3, 4)
                ]
              },
              ...
            ]
          },
          ... (days 2, 3, 4)
        ]
    """
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
                name = ex_row[0].strip()
                weeks = []
                col = 1  # data columns start at col 1 (B)

                for w in range(N_WEEKS):
                    series = []
                    for s in range(N_SERIES):
                        # Access with fallback to "" if the row is shorter than expected
                        reps = ex_row[col].strip() if col < len(ex_row) else ""
                        peso = ex_row[col + 1].strip() if (col + 1) < len(ex_row) else ""
                        series.append({"reps": reps, "peso": peso})
                        col += 2  # advance 2 columns (Rep + Peso)
                    weeks.append({"week": w + 1, "series": series})

                exercises.append({"name": name, "weeks": weeks})
                i += 1

            days.append({"day": day_num, "exercises": exercises})
        else:
            i += 1

    return days


def get_latest_week_indices(period):
    """
    Detects which weeks have data loaded in the period and returns the
    indices (0-based) of the current and previous weeks.

    Useful for weekly mode: the "current week" is the last one with any data,
    and the "previous week" is the immediately prior one.

    Args:
        period: Dict with format {"period": str, "days": [...]}.

    Returns:
        Tuple (current_idx, prev_idx) with 0-based indices.
        prev_idx is None if the current week is the first.
        Both are None if there is no data in the period.
    """
    weeks_with_data = set()
    for day in period["days"]:
        for ex in day["exercises"]:
            for w in ex["weeks"]:
                if any(s["reps"] or s["peso"] for s in w["series"]):
                    weeks_with_data.add(w["week"] - 1)  # convert to 0-based

    if not weeks_with_data:
        return (None, None)

    current = max(weeks_with_data)
    prev = current - 1 if current > 0 else None
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
            if any(s["reps"] or s["peso"] for s in week["series"]):
                exercises.append({"name": ex["name"], "series": week["series"]})
        if exercises:
            result.append({"day": day["day"], "exercises": exercises})
    return result


def load_all_periods(service, spreadsheet_id):
    """
    Loads and parses all tabs from the spreadsheet.
    Returns periods ordered with index 0 = most recent.
    """
    tabs = list_tabs(service, spreadsheet_id)
    periods = []
    for tab in tabs:
        rows = read_tab(service, spreadsheet_id, tab)
        days = parse_tab(rows)
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
