"""
helpers/reader.py — Lectura y parseo de los tabs del Google Sheets de rutinas.

Responsabilidades:
  - Autenticar con la API de Google Sheets (read-only) usando una service account.
  - Listar todos los tabs del spreadsheet en orden de posición.
  - Leer las celdas crudas de cada tab.
  - Parsear esas celdas en una estructura de datos Python usable.

Estructura de cada tab en el spreadsheet:
  Cada tab representa un período de entrenamiento (ej: "18/05/26-14/06/26")
  y contiene bloques de días con el siguiente layout:

    Fila 0: "Dia N"          ← nombre del día (ej: "Dia 1")
    Fila 1: 1 "" 1 "" ...    ← número de serie (1, 2, 3) repetido por semana
    Fila 2: Rep. Peso ...    ← etiquetas de columna, 4 semanas × 3 series × 2 cols
    Fila 3+: ejercicios      ← col A = nombre, luego reps/peso alternados
    (fila vacía entre días)

  Columnas de datos: 4 semanas × 3 series × 2 campos (Rep + Peso) = 24 columnas
  + 1 columna de nombre = 25 columnas totales (A:Y).
"""

from google.oauth2 import service_account
from googleapiclient.discovery import build

# Permiso de solo lectura — suficiente para leer el spreadsheet
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

N_WEEKS = 4    # semanas por período
N_SERIES = 3   # series por ejercicio por semana


def get_service(credentials_path):
    """
    Crea y retorna el cliente autenticado de la API de Google Sheets.

    Args:
        credentials_path: Ruta al archivo JSON de la service account de Google.

    Returns:
        Resource de la API de Google Sheets v4.
    """
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES
    )
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def list_tabs(service, spreadsheet_id):
    """
    Retorna los nombres de todos los tabs del spreadsheet en orden de posición.

    El tab en índice 0 es siempre el más reciente (pdf2xls-generator lo inserta
    al frente cada vez que procesa un PDF nuevo).

    Args:
        service:        Resource de la API de Google Sheets.
        spreadsheet_id: ID del spreadsheet (la parte larga de la URL de Google Sheets).

    Returns:
        Lista de strings con los nombres de los tabs, ej:
        ["18/05/26-14/06/26", "20/04/26-15/05/26", ...]
    """
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    return [s["properties"]["title"] for s in meta["sheets"]]


def read_tab(service, spreadsheet_id, tab_name):
    """
    Lee todas las celdas de un tab y las retorna como lista de listas de strings.

    Las filas vacías al final no son incluidas por la API. Las celdas vacías
    dentro de una fila sí aparecen como strings vacíos (o pueden estar ausentes
    si son las últimas de la fila).

    Args:
        service:        Resource de la API de Google Sheets.
        spreadsheet_id: ID del spreadsheet.
        tab_name:       Nombre exacto del tab a leer.

    Returns:
        Lista de filas, cada fila es una lista de strings.
        Ej: [["Dia 1", "", ...], ["", "1", "", "1", ...], ...]
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
    Parsea las filas crudas de un tab y devuelve una lista estructurada de días.

    Recorre las filas buscando encabezados de día ("Dia N"), luego lee los
    ejercicios que siguen hasta encontrar una fila vacía u otro encabezado.
    Para cada ejercicio extrae las reps y pesos de las 4 semanas × 3 series.

    Args:
        rows: Lista de filas crudas (output de read_tab).

    Returns:
        Lista de días con el siguiente formato:
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
                  ... (semanas 2, 3, 4)
                ]
              },
              ...
            ]
          },
          ... (días 2, 3, 4)
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

        # Detectar encabezado de día ("Dia 1", "Dia 2", etc.)
        if first_cell.lower().startswith("dia"):
            try:
                day_num = int(first_cell.split()[-1])
            except ValueError:
                i += 1
                continue

            # Saltar las 2 filas de encabezado que siguen al "Dia N":
            # - fila de números de serie (1, "", 1, "", ..., 2, "", ...)
            # - fila de etiquetas (Rep., Peso, Rep., Peso, ...)
            i += 3
            exercises = []

            while i < len(rows):
                ex_row = rows[i]

                # Una fila vacía o con col A vacía indica fin del bloque de este día
                if not ex_row or not ex_row[0]:
                    i += 1
                    break

                # Si la siguiente fila es otro "Dia N", terminar sin avanzar
                if ex_row[0].strip().lower().startswith("dia"):
                    break

                # Leer el ejercicio: nombre + 4 semanas × 3 series × (reps, peso)
                name = ex_row[0].strip()
                weeks = []
                col = 1  # las columnas de datos empiezan en col 1 (B)

                for w in range(N_WEEKS):
                    series = []
                    for s in range(N_SERIES):
                        # Acceder con fallback a "" si la fila es más corta de lo esperado
                        reps = ex_row[col].strip() if col < len(ex_row) else ""
                        peso = ex_row[col + 1].strip() if (col + 1) < len(ex_row) else ""
                        series.append({"reps": reps, "peso": peso})
                        col += 2  # avanzar 2 columnas (Rep + Peso)
                    weeks.append({"week": w + 1, "series": series})

                exercises.append({"name": name, "weeks": weeks})
                i += 1

            days.append({"day": day_num, "exercises": exercises})
        else:
            i += 1

    return days


def load_all_periods(service, spreadsheet_id):
    """
    Carga y parsea todos los tabs del spreadsheet.

    Llama a list_tabs, luego por cada tab llama a read_tab y parse_tab,
    y retorna la lista de períodos en el mismo orden que los tabs
    (índice 0 = más reciente).

    Args:
        service:        Resource de la API de Google Sheets.
        spreadsheet_id: ID del spreadsheet.

    Returns:
        Lista de períodos:
        [
          {"period": "18/05/26-14/06/26", "days": [...]},
          {"period": "20/04/26-15/05/26", "days": [...]},
          ...
        ]
    """
    tabs = list_tabs(service, spreadsheet_id)
    periods = []
    for tab in tabs:
        rows = read_tab(service, spreadsheet_id, tab)
        days = parse_tab(rows)
        periods.append({"period": tab, "days": days})
    return periods

