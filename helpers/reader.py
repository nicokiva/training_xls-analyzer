"""
helpers/reader.py — Lee y parsea los tabs del Google Sheets de rutinas.

Cada tab tiene este layout por bloque de día:
  Row 0: "Dia N"
  Row 1: series headers  (1, "", 1, "", ..., 2, "", ...)
  Row 2: labels          ("Rep.", "Peso", ...)
  Row 3+: exercises      (name, reps_s1, peso_s1, reps_s2, ...)
  blank rows between days
"""

from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

N_WEEKS = 4
N_SERIES = 3


def get_service(credentials_path):
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES
    )
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def list_tabs(service, spreadsheet_id):
    """Retorna los nombres de los tabs en orden de posición (índice 0 = más reciente)."""
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    return [s["properties"]["title"] for s in meta["sheets"]]


def read_tab(service, spreadsheet_id, tab_name):
    """Lee todas las celdas de un tab y retorna una lista de filas (listas de strings)."""
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=f"'{tab_name}'")
        .execute()
    )
    return result.get("values", [])


def parse_tab(rows):
    """
    Parsea las filas crudas de un tab y devuelve una lista de días:

    [
      {
        "day": 1,
        "exercises": [
          {
            "name": "Empuje de pecho con barra",
            "weeks": [
              {"week": 1, "series": [{"reps": "12", "peso": "60"}, ...]},
              ...
            ]
          },
          ...
        ]
      },
      ...
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

        # Detectar fila de día ("Dia 1", "Dia 2", etc.)
        if first_cell.lower().startswith("dia"):
            try:
                day_num = int(first_cell.split()[-1])
            except ValueError:
                i += 1
                continue

            # Saltar las 2 filas de encabezado (series + etiquetas Rep/Peso)
            i += 3
            exercises = []

            while i < len(rows):
                ex_row = rows[i]

                # Fin del bloque de día: fila vacía o nueva fila "Dia N"
                if not ex_row or not ex_row[0]:
                    i += 1
                    # Si la siguiente fila no vacía empieza con "Dia", salimos
                    break

                if ex_row[0].strip().lower().startswith("dia"):
                    break

                name = ex_row[0].strip()
                weeks = []
                col = 1

                for w in range(N_WEEKS):
                    series = []
                    for s in range(N_SERIES):
                        reps = ex_row[col].strip() if col < len(ex_row) else ""
                        peso = ex_row[col + 1].strip() if (col + 1) < len(ex_row) else ""
                        series.append({"reps": reps, "peso": peso})
                        col += 2
                    weeks.append({"week": w + 1, "series": series})

                exercises.append({"name": name, "weeks": weeks})
                i += 1

            days.append({"day": day_num, "exercises": exercises})
        else:
            i += 1

    return days


def load_all_periods(service, spreadsheet_id):
    """
    Carga todos los tabs del spreadsheet en orden (más reciente primero).
    Retorna lista de dicts: {"period": "18/05/26-14/06/26", "days": [...]}
    """
    tabs = list_tabs(service, spreadsheet_id)
    periods = []
    for tab in tabs:
        rows = read_tab(service, spreadsheet_id, tab)
        days = parse_tab(rows)
        periods.append({"period": tab, "days": days})
    return periods
