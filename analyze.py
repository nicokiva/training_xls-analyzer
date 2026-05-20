#!/usr/bin/env python3
"""
analyze.py — Entry point del analizador de rutinas de entrenamiento.

Modos disponibles (--mode):
  global       (default) Análisis completo de todo el historial. Detecta tendencias
               y evalúa si se está cumpliendo el objetivo a largo plazo.
  new-routine  Post pdf2xls: analiza la nueva rutina contra el historial.
               ¿Sirve para el objetivo? ¿Qué cambiaría?
  monthly      Balance del mes más reciente. ¿Cómo fue? ¿Se cumplió el objetivo?
  weekly       Compara la semana actual con la anterior (cron domingos).

Flujo:
  1. Conecta a Google Sheets con la service account.
  2. Carga los períodos necesarios según el modo.
  3. Construye el prompt apropiado y llama a Groq.
  4. Guarda el análisis en un archivo .md y opcionalmente lo manda por email.

Uso mínimo (config en .env):
    python3 analyze.py
    python3 analyze.py --mode new-routine
    python3 analyze.py --mode weekly --mock
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from helpers.reader import get_service, load_all_periods, get_latest_week_indices, extract_week_data
from helpers.ai import analyze
from helpers.mailer import send_analysis

load_dotenv()

HASH_FILE_TEMPLATE = ".last_data_hash_{mode}"

EMAIL_SUBJECTS = {
    "global":      lambda d: f"Análisis global — {d}",
    "new-routine": lambda d: f"Nueva rutina — ¿Sirve para el objetivo?",
    "monthly":     lambda d: f"Balance mensual — {d}",
    "weekly":      lambda d: f"Semana de entrenamiento — {d}",
}


def compute_hash(data):
    """Genera un hash MD5 del contenido serializado para detectar cambios."""
    content = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()


def has_changed(data, mode):
    """
    Retorna True si los datos cambiaron desde la última ejecución del modo dado.
    Actualiza el archivo de hash si hubo cambios.
    """
    hash_file = Path(HASH_FILE_TEMPLATE.format(mode=mode.replace("-", "_")))
    current_hash = compute_hash(data)
    if hash_file.exists() and hash_file.read_text().strip() == current_hash:
        return False
    hash_file.write_text(current_hash)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered gym training analysis."
    )
    parser.add_argument("--mode",
        choices=["global", "new-routine", "monthly", "weekly"],
        default="global",
        help="Analysis mode (default: global)")
    parser.add_argument("--sheets-id",      default=os.getenv("SHEETS_ID"))
    parser.add_argument("--credentials",    default=os.getenv("CREDENTIALS"))
    parser.add_argument("--api-key",        default=os.getenv("GROQ_API_KEY"))
    parser.add_argument("--goal",           default=os.getenv("GOAL", "hipertrofia"),
        help="Training objective injected into all prompts (default: hipertrofia)")
    parser.add_argument("--output",         default=None)
    parser.add_argument("--mock",           action="store_true")
    parser.add_argument("--max-periods",    type=int, default=None)
    parser.add_argument("--email-to",       default=os.getenv("EMAIL_TO"))
    parser.add_argument("--email-from",     default=os.getenv("EMAIL_FROM"))
    parser.add_argument("--email-password", default=os.getenv("EMAIL_PASSWORD"))
    args = parser.parse_args()

    if not args.sheets_id or not args.credentials or not args.api_key:
        print("Error: --sheets-id, --credentials and --api-key are required (or set them in .env).")
        sys.exit(1)

    today = datetime.now().strftime("%d/%m/%Y")
    output_path = args.output or f"analysis_{args.mode}_{datetime.now().strftime('%Y%m%d')}.md"

    print(f"Mode: {args.mode} | Goal: {args.goal}")
    print("Connecting to Google Sheets...")
    service = get_service(args.credentials)

    print("Loading training periods...")
    periods = load_all_periods(service, args.sheets_id)

    if not periods:
        print("No data found in the spreadsheet.")
        sys.exit(1)

    if args.max_periods:
        periods = periods[:args.max_periods]

    print(f"Found {len(periods)} period(s): {', '.join(p['period'] for p in periods)}")

    # --- Preparar datos según el modo y detectar cambios ---

    current_week_data = None
    prev_week_data    = None
    current_week_num  = None

    if args.mode == "weekly":
        current_period = periods[0]
        current_idx, prev_idx = get_latest_week_indices(current_period)

        if current_idx is None:
            print("No data found in the current period for weekly analysis.")
            sys.exit(1)

        current_week_num  = current_idx + 1  # convertir a 1-based para mostrar
        current_week_data = extract_week_data(current_period, current_idx)
        prev_week_data    = extract_week_data(current_period, prev_idx) if prev_idx is not None else None

        change_data = current_week_data
    elif args.mode == "new-routine":
        change_data = periods[0]["days"]  # solo la nueva rutina
    elif args.mode == "monthly":
        change_data = periods[0]
    else:
        change_data = periods

    if not args.mock and not has_changed(change_data, args.mode):
        print("No changes since last run. Nothing to do.")
        sys.exit(0)

    print("Analyzing with Groq...")
    analysis = analyze(
        periods,
        args.api_key,
        mock=args.mock,
        mode=args.mode,
        goal=args.goal,
        current_week_data=current_week_data,
        prev_week_data=prev_week_data,
        current_week_num=current_week_num,
    )

    Path(output_path).write_text(analysis, encoding="utf-8")
    print(f"Analysis saved to: {output_path}")

    if args.email_to:
        if not args.email_from or not args.email_password:
            print("Error: --email-from and --email-password are required to send email.")
            sys.exit(1)
        subject = EMAIL_SUBJECTS.get(args.mode, EMAIL_SUBJECTS["global"])(today)
        print(f"Sending email to {args.email_to}...")
        send_analysis(args.email_from, args.email_password, args.email_to, subject, analysis)
        print("Email sent.")


if __name__ == "__main__":
    main()

