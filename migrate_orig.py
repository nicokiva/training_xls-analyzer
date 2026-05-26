#!/usr/bin/env python3
"""
migrate_orig.py — Copies data and notes from the ORIG backup tab to the new active tab.

Usage:
    python3 migrate_orig.py [--dry-run] [--audit-only]

Options:
    --dry-run      Show what would be written without actually writing anything.
    --audit-only   Same as --dry-run: only print the diff, no writes.

What it does:
  1. Finds the ORIG tab (starts with "ORIG") and the active tab (ends with "-...").
  2. Reads all exercise data from ORIG (including text values like "8 a 28 / 8 a 20 / 8 a 15").
  3. Reads all cell notes from ORIG.
  4. Compares cell by cell: for each ORIG value that is non-empty, checks whether the
     corresponding cell in the new tab has real (non-italic) data.
  5. Writes missing values, clears their italic formatting (makes them non-italic = real data).
  6. Writes missing notes.

Abdomen rule:
  Abdomen exercises appear in all 4 days of both ORIG and new tab.
  Each new day maps to a specific ORIG day (determined by which non-abdomen exercises overlap).
  Abdomen data is copied from the corresponding ORIG day only — cell by cell, only the weeks
  that were actually done. Days with no ORIG abdomen data are left empty.

Exercise name matching:
  Uses a RENAMES dict to translate ORIG names to new-tab names, with fuzzy fallback.
"""

import argparse
import os
import re
import sys
import unicodedata

from dotenv import load_dotenv

from helpers.reader.reader import (
    get_service,
    get_write_service,
    list_tabs,
    read_tab,
    read_tab_notes,
    read_tab_italic_cells,
    parse_tab,
    N_WEEKS,
    N_SERIES,
)

load_dotenv()

SHEETS_ID   = os.getenv("SHEETS_ID")
CREDENTIALS = os.getenv("CREDENTIALS")

# ---------------------------------------------------------------------------
# Exercise name translation: ORIG name → new tab name (both normalized)
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Lowercase + remove accents + remove parentheses + collapse spaces."""
    nfkd    = unicodedata.normalize("NFKD", name.lower().strip())
    no_acc  = "".join(c for c in nfkd if not unicodedata.combining(c))
    no_par  = re.sub(r"\(.*?\)", "", no_acc)
    no_sym  = re.sub(r"[°]", "", no_par)
    return re.sub(r"\s+", " ", no_sym).strip()


# Maps _norm(orig_name) → _norm(new_name).
# Only entries where the name actually changed are needed; same-name exercises
# will be matched directly.
RENAMES: dict[str, str] = {
    _norm("Press plano con barra"):               _norm("Empuje de pecho con barra en banco plano"),
    _norm("Press inclinado con barra"):            _norm("Empuje de pecho con barra en banco inclinado"),
    _norm("Peck Deck"):                            _norm("Peck Deck"),
    _norm("Triceps prono con polea"):              _norm("Triceps con polea"),
    _norm("Empuje hammer"):                        _norm("Empuje de pecho en Hammer"),
    _norm("Press militar sentado con barra"):      _norm("Empuje de hombros con barra"),
    _norm("Press militar parado con mancuernas"):  _norm("Empuje de hombros con mancuernas"),
    _norm("Flexion de rodillas sentado"):          _norm("Flexion de rodillas en maquina"),
    _norm("Estocada en banco"):                    _norm("Estocada en banco"),
    _norm("Prensa 45"):                            _norm("Prensa Hammer 45"),
    _norm("Extension de rodillas"):                _norm("Extension de rodillas en maquina"),
    _norm("Tiron supino con polea"):               _norm("Tiron dorsal en polea con agarre supino"),
    _norm("Biceps en polea"):                      _norm("Biceps con polea"),
    _norm("Abdominal recto largo"):                _norm("Abdominal recto largo"),
    # Same-name exercises (included explicitly so we can track them)
    _norm("Rotaciones de pie con disco"):          _norm("Rotaciones de pie con disco"),
    _norm("Extension de cadera en banco"):         _norm("Extension de cadera en banco"),
    _norm("Sentadilla clasica"):                   _norm("Sentadilla clasica"),
    _norm("Peso muerto"):                          _norm("Peso muerto"),
    _norm("Dominada estricta"):                    _norm("Dominada estricta"),
    _norm("Depresores en polea"):                  _norm("Depresores en polea"),
    _norm("Remo en polea baja"):                   _norm("Remo en polea baja"),
    _norm("Remo al menton"):                       _norm("Remo al menton"),
    _norm("Vuelos laterales con mancuernas"):      _norm("Vuelos laterales con mancuernas"),
}

# Reverse: _norm(new_name) → _norm(orig_name)
_REV_RENAMES: dict[str, str] = {v: k for k, v in RENAMES.items()}

# Abdomen exercise names (normalized, without parentheses content)
ABDOMEN_NORMS = {
    _norm("Abdominal recto largo"),
    _norm("Rotaciones de pie con disco"),
    _norm("Extension de cadera en banco"),
}


def _orig_norm_for_new(new_name: str) -> str:
    """Given a new-tab exercise name, return the normalized ORIG name to look up."""
    n = _norm(new_name)
    return _REV_RENAMES.get(n, n)


def _col_letter(col_idx: int) -> str:
    """0-based column index → spreadsheet column letter (A, B, ..., Z, AA, ...)."""
    result = ""
    col_idx += 1
    while col_idx:
        col_idx, rem = divmod(col_idx - 1, 26)
        result = chr(65 + rem) + result
    return result


def _peso_col(week_idx: int, set_idx: int) -> int:
    """0-based column of a peso cell: week_idx and set_idx are both 0-based."""
    return 1 + week_idx * (N_SERIES * 2) + set_idx * 2 + 1


def _reps_col(week_idx: int, set_idx: int) -> int:
    """0-based column of a reps cell."""
    return 1 + week_idx * (N_SERIES * 2) + set_idx * 2


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(dry_run: bool = False):
    print("Connecting to Google Sheets...")
    svc       = get_service(CREDENTIALS)
    write_svc = None if dry_run else get_write_service(CREDENTIALS)

    tabs = list_tabs(svc, SHEETS_ID)
    orig_tab = next((t for t in tabs if t.startswith("ORIG")), None)
    new_tab  = next((t for t in tabs if re.match(r"^\d{2}/\d{2}/\d{2}-\.\.\.$", t)), None)

    if not orig_tab:
        print("ERROR: No ORIG tab found.")
        sys.exit(1)
    if not new_tab:
        print("ERROR: No active tab (ending in '-...') found.")
        sys.exit(1)

    print(f"ORIG tab : {orig_tab}")
    print(f"New tab  : {new_tab}")

    # ── Read data ────────────────────────────────────────────────────────────
    print("\nReading ORIG...")
    orig_rows  = read_tab(svc, SHEETS_ID, orig_tab)
    orig_notes = read_tab_notes(svc, SHEETS_ID, orig_tab)

    print("Reading new tab...")
    new_rows   = read_tab(svc, SHEETS_ID, new_tab)
    new_italic = read_tab_italic_cells(svc, SHEETS_ID, new_tab)

    # ── Compute new_day → orig_day mapping ──────────────────────────────────
    # For each new day, find the ORIG day with the most non-abdomen exercise matches.
    # This handles day reordering between the old and new routines.
    orig_days_parsed = parse_tab(orig_rows)
    day_map: dict[int, int] = {}  # new_day → orig_day
    for new_day in parse_tab(new_rows):
        nd = new_day["day"]
        new_names = {
            RENAMES.get(_norm(ex["name"]), _norm(ex["name"]))
            for ex in new_day["exercises"]
            if _norm(ex["name"]) not in ABDOMEN_NORMS
        }
        best_score, best_od = 0, None
        for orig_day in orig_days_parsed:
            od = orig_day["day"]
            orig_names = {
                RENAMES.get(_norm(ex["name"]), _norm(ex["name"]))
                for ex in orig_day["exercises"]
                if _norm(ex["name"]) not in ABDOMEN_NORMS
            }
            score = len(new_names & orig_names)
            if score > best_score:
                best_score, best_od = score, od
        day_map[nd] = best_od
        print(f"  Day mapping: New D{nd} ← ORIG D{best_od} (matched {best_score} exercises)")

    # Build ORIG abdomen index: (orig_day, norm_name) → exercise dict
    orig_abd: dict[tuple, dict] = {}
    for orig_day in orig_days_parsed:
        for ex in orig_day["exercises"]:
            n = _norm(ex["name"])
            if n in ABDOMEN_NORMS:
                orig_abd[(orig_day["day"], n)] = ex

    # ── Build ORIG exercise index ────────────────────────────────────────────
    # _norm(name) → list of (orig_day_num, exercise_dict)
    orig_by_norm: dict[str, list] = {}
    for day in parse_tab(orig_rows):
        for ex in day["exercises"]:
            k = _norm(ex["name"])
            orig_by_norm.setdefault(k, []).append((day["day"], ex))

    # ── Build new-tab row index ──────────────────────────────────────────────
    # (new_day_num, _norm(name)) → row_idx in new_rows
    new_row_index: dict[tuple, int] = {}
    current_day = 0
    for row_idx, row in enumerate(new_rows):
        if not row or not row[0]:
            continue
        cell = row[0].strip()
        if cell.lower().startswith("dia"):
            try:
                current_day = int(cell.split()[-1])
            except ValueError:
                pass
            continue
        if cell.lower() in ("rep.", "peso"):
            continue
        clean = cell[4:] if cell.startswith("[C] ") else cell
        new_row_index[(current_day, _norm(clean))] = row_idx

    # ── Get sheet numeric ID for formatting requests ─────────────────────────
    sheet_id = None
    if not dry_run:
        meta = write_svc.spreadsheets().get(spreadsheetId=SHEETS_ID).execute()
        for s in meta["sheets"]:
            if s["properties"]["title"] == new_tab:
                sheet_id = s["properties"]["sheetId"]
                break
        if sheet_id is None:
            print(f"ERROR: Tab '{new_tab}' not found in spreadsheet metadata.")
            sys.exit(1)

    # ── Collect writes ────────────────────────────────────────────────────────
    # For each exercise in new tab, determine source ORIG exercise and copy.
    value_updates   = []  # for values().batchUpdate()
    format_requests = []  # for spreadsheets().batchUpdate() — clear italic
    note_updates    = []  # (new_row_idx, new_col_idx, note_text)
    audit_rows      = []  # (action, new_day, new_name, week, set, value)

    new_days_parsed = parse_tab(new_rows)

    for new_day in new_days_parsed:
        nd = new_day["day"]
        for new_ex in new_day["exercises"]:
            new_name = new_ex["name"]
            new_norm = _norm(new_name)
            orig_norm = _orig_norm_for_new(new_name)

            orig_occurrences = orig_by_norm.get(orig_norm, [])

            # Fuzzy fallback: try substring match
            if not orig_occurrences:
                for k, v in orig_by_norm.items():
                    if orig_norm in k or k in orig_norm:
                        orig_occurrences = v
                        print(f"  [fuzzy] {new_name!r} → matched ORIG {v[0][1]['name']!r}")
                        break

            if not orig_occurrences:
                print(f"  [WARN] No ORIG match for: Day {nd} {new_name!r}")
                continue

            is_abd = orig_norm in ABDOMEN_NORMS

            if is_abd:
                # Abdomen: use the ORIG day that corresponds to this new day (cell-by-cell).
                od = day_map.get(nd)
                orig_ex_entry = orig_abd.get((od, orig_norm)) if od else None
                if orig_ex_entry is None:
                    continue  # no abdomen data for this ORIG day — leave empty
                orig_day_num, orig_ex = od, orig_ex_entry
            else:
                # Non-abdomen: exactly one ORIG occurrence
                orig_day_num, orig_ex = orig_occurrences[0]

            # Find new-tab row index for this (day, exercise)
            new_row_idx = new_row_index.get((nd, new_norm))
            if new_row_idx is None:
                print(f"  [WARN] Row not found in new tab: Day {nd} {new_name!r}")
                continue

            new_row_data = new_rows[new_row_idx] if new_row_idx < len(new_rows) else []

            # ── Peso cells ──────────────────────────────────────────────────
            for w_idx, orig_week in enumerate(orig_ex["weeks"][:N_WEEKS]):
                for s_idx, orig_ser in enumerate(orig_week["series"][:N_SERIES]):
                    orig_peso = orig_ser["peso"].strip() if orig_ser["peso"] else ""
                    if not orig_peso:
                        continue  # nothing to copy

                    col = _peso_col(w_idx, s_idx)
                    existing = new_row_data[col].strip() if col < len(new_row_data) else ""
                    is_italic = (new_row_idx, col) in new_italic

                    # Write if: empty, or currently an AI suggestion (italic)
                    if existing and not is_italic:
                        # Already has real data — skip
                        continue

                    cell_a1 = f"'{new_tab}'!{_col_letter(col)}{new_row_idx + 1}"
                    audit_rows.append(
                        f"  WRITE D{nd} {new_name!r} W{w_idx+1} S{s_idx+1} "
                        f"col={_col_letter(col)}{new_row_idx+1}: {orig_peso!r}"
                        + (" (was italic AI)" if is_italic else "")
                    )
                    if not dry_run:
                        value_updates.append({"range": cell_a1, "values": [[orig_peso]]})
                        format_requests.append({
                            "repeatCell": {
                                "range": {
                                    "sheetId": sheet_id,
                                    "startRowIndex": new_row_idx,
                                    "endRowIndex": new_row_idx + 1,
                                    "startColumnIndex": col,
                                    "endColumnIndex": col + 1,
                                },
                                "cell": {"userEnteredFormat": {"textFormat": {"italic": False}}},
                                "fields": "userEnteredFormat.textFormat.italic",
                            }
                        })

            # ── Reps cells (copy too) ────────────────────────────────────────
            for w_idx, orig_week in enumerate(orig_ex["weeks"][:N_WEEKS]):
                for s_idx, orig_ser in enumerate(orig_week["series"][:N_SERIES]):
                    orig_reps = orig_ser["reps"].strip() if orig_ser["reps"] else ""
                    if not orig_reps:
                        continue

                    col = _reps_col(w_idx, s_idx)
                    existing = new_row_data[col].strip() if col < len(new_row_data) else ""
                    is_italic = (new_row_idx, col) in new_italic
                    if existing and not is_italic:
                        continue

                    cell_a1 = f"'{new_tab}'!{_col_letter(col)}{new_row_idx + 1}"
                    if not dry_run:
                        value_updates.append({"range": cell_a1, "values": [[orig_reps]]})
                        format_requests.append({
                            "repeatCell": {
                                "range": {
                                    "sheetId": sheet_id,
                                    "startRowIndex": new_row_idx,
                                    "endRowIndex": new_row_idx + 1,
                                    "startColumnIndex": col,
                                    "endColumnIndex": col + 1,
                                },
                                "cell": {"userEnteredFormat": {"textFormat": {"italic": False}}},
                                "fields": "userEnteredFormat.textFormat.italic",
                            }
                        })

    # ── Notes ─────────────────────────────────────────────────────────────────
    # Read existing notes in new tab so we don't flag already-copied notes.
    existing_new_notes = read_tab_notes(svc, SHEETS_ID, new_tab)

    # Build ORIG row → (day, exercise_norm) for reverse lookup
    orig_row_to_ex: dict[int, tuple[int, str]] = {}
    cur_day = 0
    for row_idx, row in enumerate(orig_rows):
        if not row or not row[0]:
            continue
        cell = row[0].strip()
        if cell.lower().startswith("dia"):
            try:
                cur_day = int(cell.split()[-1])
            except ValueError:
                pass
            continue
        if cell.lower() in ("rep.", "peso"):
            continue
        clean = cell[4:] if cell.startswith("[C] ") else cell
        orig_row_to_ex[row_idx] = (cur_day, _norm(clean))

    for (orig_row_idx, orig_col_idx), note_text in orig_notes.items():
        if orig_row_idx not in orig_row_to_ex:
            continue
        orig_day_num, orig_ex_norm = orig_row_to_ex[orig_row_idx]
        new_norm_name = RENAMES.get(orig_ex_norm, orig_ex_norm)

        # Find matching new-tab row for this (day, exercise)
        # Note: day number may differ (days were reordered), so search all days
        matching_new_rows = []
        for (nd, nn), nr in new_row_index.items():
            if nn == new_norm_name or nn == orig_ex_norm:
                matching_new_rows.append((nd, nr))

        for nd, new_row_idx in matching_new_rows:
            # Skip if note already exists in new tab with same text
            existing = existing_new_notes.get((new_row_idx, orig_col_idx), "")
            if existing.strip() == note_text.strip():
                continue  # already there, nothing to do
            note_updates.append((new_row_idx, orig_col_idx, note_text, nd, new_norm_name))
            audit_rows.append(
                f"  NOTE D{nd} row={new_row_idx+1} col={_col_letter(orig_col_idx)}: {note_text!r}"
            )

    # ── Print audit ────────────────────────────────────────────────────────────
    if audit_rows:
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Changes to apply ({len(audit_rows)}):")
        for line in audit_rows:
            print(line)
    else:
        print("\n✓ Nothing to migrate — new tab is already in sync with ORIG.")
        return

    if dry_run:
        print("\n[DRY RUN] No writes performed.")
        return

    # ── Write values ─────────────────────────────────────────────────────────
    if value_updates:
        print(f"\nWriting {len(value_updates)} cell value(s)...")
        write_svc.spreadsheets().values().batchUpdate(
            spreadsheetId=SHEETS_ID,
            body={
                "valueInputOption": "USER_ENTERED",
                "data": value_updates,
            },
        ).execute()

    # ── Apply non-italic format ───────────────────────────────────────────────
    if format_requests:
        print(f"Clearing italic on {len(format_requests)} cell(s)...")
        write_svc.spreadsheets().batchUpdate(
            spreadsheetId=SHEETS_ID,
            body={"requests": format_requests},
        ).execute()

    # ── Write notes ───────────────────────────────────────────────────────────
    if note_updates:
        print(f"Writing {len(note_updates)} note(s)...")
        # Build updateCells requests for notes
        note_requests = []
        for (new_row_idx, orig_col_idx, note_text, nd, name) in note_updates:
            note_requests.append({
                "updateCells": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": new_row_idx,
                        "endRowIndex": new_row_idx + 1,
                        "startColumnIndex": orig_col_idx,
                        "endColumnIndex": orig_col_idx + 1,
                    },
                    "rows": [{"values": [{"note": note_text}]}],
                    "fields": "note",
                }
            })
        write_svc.spreadsheets().batchUpdate(
            spreadsheetId=SHEETS_ID,
            body={"requests": note_requests},
        ).execute()

    print("\n✓ Migration complete.")


def main():
    parser = argparse.ArgumentParser(description="Migrate ORIG tab data to the active tab.")
    parser.add_argument("--dry-run",    action="store_true", help="Show what would change without writing.")
    parser.add_argument("--audit-only", action="store_true", help="Alias for --dry-run.")
    args = parser.parse_args()

    if not SHEETS_ID or not CREDENTIALS:
        print("ERROR: SHEETS_ID and CREDENTIALS must be set in .env")
        sys.exit(1)

    run(dry_run=args.dry_run or args.audit_only)


if __name__ == "__main__":
    main()
