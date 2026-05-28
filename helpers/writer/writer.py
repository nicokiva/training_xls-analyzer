"""
helpers/writer/writer.py — Writes AI weight suggestions back to the Google Sheet.

Responsibilities:
  - Parse the JSON suggestions block embedded in the AI new-routine response.
  - Find the corresponding exercise rows in the target tab.
  - Write suggested peso values to empty peso cells with italic formatting.
  - Write suggested rest times to col Z (Pausa), merging for combined groups.
  - Format suggestions as HTML for the email (grouped by day, with reasons).

Sheet layout (matching reader.py):
  Col 0 (A): exercise name
  For week W (0-indexed) and set S (0-indexed):
    reps col = 1 + W * (N_SERIES * 2) + S * 2
    peso col = 1 + W * (N_SERIES * 2) + S * 2 + 1
  Col Z (index 25): Pausa — suggested rest time (one per exercise or one per combined group)
"""

import json
import re
import unicodedata

from helpers.reader.reader import N_WEEKS, N_SERIES
from helpers.ai.ai import _canonical_key

# Col Z (0-based index 25): suggested rest time per exercise / combined group
PAUSA_COL = N_WEEKS * N_SERIES * 2 + 1  # = 25


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_suggestions(ai_response):
    """
    Extracts the JSON suggestions block from the AI response.

    The AI is asked to embed a ```json ... ``` block in its response.
    Returns the parsed list of suggestion dicts, or None if not found/invalid.

    Each suggestion dict has the shape:
      {
        "exercise": str,
        "day": int,
        "weeks": [float, float, float, float],  # one per week
        "rest_s": int,
        "reason": str
      }
    """
    match = re.search(r"```json\s*([\s\S]*?)\s*```", ai_response)
    if not match:
        return None
    try:
        suggestions = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    # Sanitize rest_s: must be int or None. Reject strings like "—", "null", "90s".
    for s in suggestions:
        val = s.get("rest_s")
        if val is None:
            continue
        if isinstance(val, (int, float)):
            s["rest_s"] = int(val)
        else:
            # Try to extract digits (e.g. "90s" → 90), otherwise null
            digits = re.sub(r"[^\d]", "", str(val))
            s["rest_s"] = int(digits) if digits else None
    return suggestions


def validate_suggestions(suggestions, settled_weights):
    """
    Post-process AI weight suggestions against pre-calculated settled weights.

    If the AI's week-1 suggestion for an exercise deviates by more than 40%
    from the known settled weight, override it and re-scale the progression
    proportionally.  Logs a warning for each correction.

    Args:
        suggestions:     Parsed list from parse_suggestions().
        settled_weights: Dict of normalized exercise name → settled peso (float).
                         Keys should be normalized (lowercase, accent-stripped,
                         parentheses removed) for fuzzy matching.

    Returns:
        The (possibly corrected) suggestions list.
    """
    import unicodedata

    def _norm(s):
        nfkd = unicodedata.normalize("NFKD", s.lower().strip())
        no_acc = "".join(c for c in nfkd if not unicodedata.combining(c))
        return re.sub(r"\s*\(.*?\)", "", no_acc).strip()

    for s in suggestions:
        name     = s.get("exercise", "")
        weeks    = s.get("weeks", [])
        if not weeks or weeks[0] is None:
            continue
        ai_w1 = float(weeks[0])
        if ai_w1 == 0:
            continue  # bodyweight — skip

        # Use canonical key so it matches the keys in settled_weights dict
        key = _canonical_key(name)
        baseline = settled_weights.get(key)

        if baseline is None or baseline == 0:
            continue

        ratio = ai_w1 / baseline
        if ratio < 0.6:
            # AI is more than 40% below baseline — likely hallucinating a low weight, override
            print(f"  [validate] Correcting '{name}': AI={ai_w1:.1f}kg vs "
                  f"baseline={baseline:.1f}kg (ratio={ratio:.2f}) → overriding")
            # Preserve progression: add AI-style increments to the correct base
            increments = [w - weeks[i-1] for i, w in enumerate(weeks) if i > 0
                          and weeks[i] is not None and weeks[i-1] is not None]
            typical_inc = (sum(increments) / len(increments)) if increments else 2.5
            # Use baseline as week-1, keep AI increments (capped at sane range)
            typical_inc = max(1.25, min(5.0, typical_inc))
            new_weeks = [baseline + typical_inc * i for i in range(len(weeks))]
            s["weeks"] = [round(w * 2) / 2 for w in new_weeks]  # round to 0.5
            s["reason"] = (s.get("reason", "") +
                           f" [corregido: baseline script={baseline:.1f}kg]")
    return suggestions


def strip_suggestions_block(ai_response):
    """Removes the ```json ... ``` block from the AI response (for clean email body)."""
    return re.sub(r"```json\s*[\s\S]*?\s*```", "", ai_response).strip()


# ---------------------------------------------------------------------------
# Writing to sheet
# ---------------------------------------------------------------------------

def _normalize(name):
    """Lowercase + strip accents + remove parentheses/punctuation for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", name.lower().strip())
    # Remove accents
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Remove parentheses and their contents, then collapse spaces
    no_parens = re.sub(r"\(.*?\)", "", no_accents)
    return re.sub(r"\s+", " ", no_parens).strip()


def _peso_col(week_idx, set_idx):
    """Returns the 0-based column index for a peso cell."""
    return 1 + week_idx * (N_SERIES * 2) + set_idx * 2 + 1


def write_suggestions_to_sheet(service, spreadsheet_id, tab_name, suggestions, overwrite_italic=False):
    """
    Writes suggested peso values to empty peso cells in the tab, formatted as italic.

    Only writes to cells that are currently empty — never overwrites actual training data.
    When overwrite_italic=True (new-routine mode), also overwrites cells that already
    contain values, since any existing values there are previous AI suggestions (italic),
    not real training data.

    Args:
        service:          Google Sheets API client with write scope.
        spreadsheet_id:   ID of the spreadsheet.
        tab_name:         Name of the tab (e.g. "01/06/26-...").
        suggestions:      List of dicts from parse_suggestions().
        overwrite_italic: If True, overwrite existing cell values (new-routine re-runs).
    """
    if not suggestions:
        return

    # --- Get sheet numeric ID (needed for formatting requests) ---
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_id = None
    for s in meta["sheets"]:
        if s["properties"]["title"] == tab_name:
            sheet_id = s["properties"]["sheetId"]
            break
    if sheet_id is None:
        print(f"  [writer] Tab '{tab_name}' not found in spreadsheet.")
        return

    # --- Read raw rows to find exercise row indices and existing values ---
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=f"'{tab_name}'")
        .execute()
    )
    rows = result.get("values", [])

    # Build (day, name) → (row_index, row) map (col A, skip header/day rows)
    # Keyed by day so exercises that repeat across days (e.g. the abdomen group)
    # map to the correct row for each day, not just the last occurrence.
    # Also record which rows are combined ("[C] " prefix) so we can detect
    # the last exercise in each combined group for the Pausa column.
    name_to_row = {}   # (day, norm_name) → (row_idx, row)
    row_is_comb  = {}  # row_idx → bool
    current_day  = 0
    for row_idx, row in enumerate(rows):
        if not row or not row[0]:
            continue
        cell = row[0].strip()
        if cell.lower().startswith("dia"):
            try:
                current_day = int(cell.split()[1])
            except (IndexError, ValueError):
                pass
            continue
        if cell.lower() in ("rep.", "peso"):
            continue
        is_comb = cell.startswith("[C] ")
        clean   = cell[4:] if is_comb else cell
        name_to_row[(current_day, _normalize(clean))] = (row_idx, row)
        row_is_comb[row_idx] = is_comb

    # Build a set of row indices that are the LAST in their combined group,
    # and a map from any combined row to the FIRST row of its group.
    #
    # Why track the first row?  The sheet merges the Pausa (col Z) cells for
    # the entire combined group (e.g. Z4:Z6 for 3 abdomen rows).  Google Sheets
    # silently ignores writes to any cell inside a merge that isn't the top-left
    # cell, so we must always write the Pausa value to the FIRST row of the group.
    exercise_row_indices = sorted(v[0] for v in name_to_row.values())
    last_of_comb_group   = set()
    comb_group_first     = {}   # row_idx → first row_idx of its combined group
    current_group_start  = None
    for i, ridx in enumerate(exercise_row_indices):
        if row_is_comb.get(ridx):
            if current_group_start is None:
                current_group_start = ridx
            comb_group_first[ridx] = current_group_start
            next_ridx = exercise_row_indices[i + 1] if i + 1 < len(exercise_row_indices) else None
            if next_ridx is None or not row_is_comb.get(next_ridx):
                last_of_comb_group.add(ridx)
                current_group_start = None
        else:
            current_group_start = None

    value_updates = []   # for values().batchUpdate()
    format_requests = [] # for spreadsheets().batchUpdate()

    for suggestion in suggestions:
        ex_name = suggestion.get("exercise", "")
        day     = suggestion.get("day", 0)
        weeks   = suggestion.get("weeks", [])
        rest_s  = suggestion.get("rest_s")
        norm    = _normalize(ex_name)

        # Exact match by (day, name) first, then fuzzy within same day
        match = name_to_row.get((day, norm))
        if match is None:
            for (d, key), val in name_to_row.items():
                if d == day and (norm in key or key in norm):
                    match = val
                    break
        if match is None:
            print(f"  [writer] Exercise not found in sheet: day={day} '{ex_name}'")
            continue

        row_idx, row = match

        # ── Peso cells ──────────────────────────────────────────────────────
        for week_idx, peso in enumerate(weeks[:N_WEEKS]):
            if peso is None:
                continue
            peso_str = str(int(peso)) if peso == int(peso) else str(peso)

            for set_idx in range(N_SERIES):
                col = _peso_col(week_idx, set_idx)
                existing = row[col].strip() if col < len(row) else ""
                if existing and not overwrite_italic:
                    continue

                col_letter = _col_letter(col)
                cell_a1 = f"'{tab_name}'!{col_letter}{row_idx + 1}"

                value_updates.append({"range": cell_a1, "values": [[peso_str]]})
                format_requests.append({
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": row_idx,
                            "endRowIndex": row_idx + 1,
                            "startColumnIndex": col,
                            "endColumnIndex": col + 1,
                        },
                        "cell": {"userEnteredFormat": {"textFormat": {"italic": True}}},
                        "fields": "userEnteredFormat.textFormat.italic",
                    }
                })

        # ── Pausa (col Z) ────────────────────────────────────────────────────
        # For individual exercises: write rest_s to their own row.
        # For combined groups: only fire once (on the LAST exercise, which is when
        #   the AI sends a non-null rest_s), but write to the FIRST row of the group.
        #   The sheet merges col Z across all combined rows; Google Sheets silently
        #   discards writes to any cell inside a merge that isn't the top-left cell.
        if rest_s is not None:
            is_comb = row_is_comb.get(row_idx, False)
            should_write = (not is_comb) or (row_idx in last_of_comb_group)
            if should_write:
                # For combined groups, write to the first (top-left) row of the merge.
                pausa_row = comb_group_first.get(row_idx, row_idx) if is_comb else row_idx
                pausa_src_row = rows[pausa_row] if pausa_row < len(rows) else []
                existing_z = pausa_src_row[PAUSA_COL].strip() if PAUSA_COL < len(pausa_src_row) else ""
                if not existing_z or overwrite_italic:
                    cell_z = f"'{tab_name}'!{_col_letter(PAUSA_COL)}{pausa_row + 1}"
                    value_updates.append({"range": cell_z, "values": [[f"{rest_s}s"]]})
                    format_requests.append({
                        "repeatCell": {
                            "range": {
                                "sheetId": sheet_id,
                                "startRowIndex": pausa_row,
                                "endRowIndex": pausa_row + 1,
                                "startColumnIndex": PAUSA_COL,
                                "endColumnIndex": PAUSA_COL + 1,
                            },
                            "cell": {"userEnteredFormat": {"textFormat": {"italic": True}}},
                            "fields": "userEnteredFormat.textFormat.italic",
                        }
                    })

    if not value_updates:
        print("  [writer] No empty cells to fill.")
        return

    # Write values
    service.spreadsheets().values().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={
            "valueInputOption": "RAW",
            "data": value_updates,
        },
    ).execute()

    # Apply italic formatting
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": format_requests},
    ).execute()

    print(f"  [writer] Wrote {len(value_updates)} suggested peso cell(s) to '{tab_name}'.")


def _col_letter(col_idx):
    """Converts a 0-based column index to a spreadsheet column letter (A, B, ..., Z, AA, ...)."""
    result = ""
    col_idx += 1  # 1-based
    while col_idx:
        col_idx, remainder = divmod(col_idx - 1, 26)
        result = chr(65 + remainder) + result
    return result


# ---------------------------------------------------------------------------
# Email formatting
# ---------------------------------------------------------------------------

def format_suggestions_for_email(suggestions):
    """
    Formats the suggestions as an HTML section grouped by day.
    Returns an HTML string (safe to embed in the Markdown body — the markdown
    library passes raw HTML blocks through unchanged).
    """
    if not suggestions:
        return ""

    # Group by day
    by_day = {}
    for s in suggestions:
        by_day.setdefault(s.get("day", 0), []).append(s)

    parts = ["\n\n---\n\n## Pesos y pausas sugeridos para arrancar\n\n"]

    for day in sorted(by_day):
        parts.append(f"### Día {day}\n\n")
        parts.append(
            '<table class="sug-table">'
            "<thead><tr>"
            "<th>Ejercicio</th>"
            "<th>Sem 1</th><th>Sem 2</th><th>Sem 3</th><th>Sem 4</th>"
            "<th>Pausa</th>"
            "<th>Tempo</th>"
            "<th>RIR W4</th>"
            "<th>Desafío del mes</th>"
            "<th>Progresión histórica</th>"
            "<th>Por qué</th>"
            "</tr></thead><tbody>"
        )
        for s in by_day[day]:
            name    = s.get("exercise", "")
            is_comb = s.get("is_comb", False)
            weeks   = s.get("weeks", [None] * 4)
            rest_s  = s.get("rest_s")
            reason  = s.get("reason", "")
            tempo   = s.get("tempo", "Controlado")
            challenge = s.get("challenge", "")
            rir_w4    = s.get("rir_w4", "")
            progression = s.get("progression_analysis", "")

            w = [(f"{v} kg" if v is not None else "—") for v in weeks]
            rest_str = f"{rest_s}s" if rest_s else "—"
            name_class = "comb" if is_comb else "name"
            comb_note  = " <em>(combinado)</em>" if is_comb else ""

            parts.append(
                f"<tr>"
                f'<td class="{name_class}">{name}{comb_note}</td>'
                f'<td class="num">{w[0]}</td>'
                f'<td class="num">{w[1]}</td>'
                f'<td class="num">{w[2]}</td>'
                f'<td class="num">{w[3]}</td>'
                f'<td class="num">{rest_str}</td>'
                f'<td class="tempo">{tempo}</td>'
                f'<td class="rir">{rir_w4}</td>'
                f'<td class="chall">{challenge}</td>'
                f'<td class="prog">{progression}</td>'
                f'<td class="why">{reason}</td>'
                f"</tr>"
            )
        parts.append("</tbody></table>\n\n")

    return "".join(parts)
