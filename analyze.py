#!/usr/bin/env python3
"""
analyze.py — Entry point del analizador de rutinas de entrenamiento.

Flujo completo:
  1. Se conecta a Google Sheets usando una service account de Google.
  2. Lee todos los tabs del spreadsheet (cada tab = un período de entrenamiento).
  3. Parsea los datos: ejercicios, repeticiones y pesos por semana/serie.
  4. Arma un prompt ejercicio-céntrico y lo manda a Groq (LLaMA 3).
  5. Guarda el análisis generado en un archivo Markdown.

Uso:
    python3 analyze.py \\
        --sheets-id SPREADSHEET_ID \\
        --credentials /path/to/service_account.json \\
        --api-key gsk_... \\
        [--output analysis.md] \\
        [--max-periods N] \\
        [--mock]
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from helpers.reader import get_service, load_all_periods
from helpers.ai import analyze
from helpers.mailer import send_analysis

# Cargar variables del .env antes de leer args (los CLI args tienen prioridad)
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Generate an AI-powered analysis of gym training progressions."
    )
    parser.add_argument("--sheets-id",   default=os.getenv("SHEETS_ID"),      help="Google Sheets spreadsheet ID")
    parser.add_argument("--credentials", default=os.getenv("CREDENTIALS"),    help="Path to Google service account JSON")
    parser.add_argument("--api-key",     default=os.getenv("GROQ_API_KEY"),   help="Groq API key (console.groq.com)")
    parser.add_argument("--output",      default=None,                         help="Output .md file (default: analysis_YYYYMMDD.md)")
    parser.add_argument("--mock",        action="store_true",                  help="Skip Groq API calls and write a fake analysis (for testing)")
    parser.add_argument("--max-periods", type=int, default=None,               help="Limit analysis to the N most recent periods (default: all)")
    parser.add_argument("--email-to",       default=os.getenv("EMAIL_TO"),       help="Send analysis to this email address")
    parser.add_argument("--email-from",     default=os.getenv("EMAIL_FROM"),     help="Gmail address to send from")
    parser.add_argument("--email-password", default=os.getenv("EMAIL_PASSWORD"), help="Gmail App Password (16 chars)")
    args = parser.parse_args()

    if not args.sheets_id or not args.credentials or not args.api_key:
        print("Error: --sheets-id, --credentials and --api-key are required (or set them in .env).")
        sys.exit(1)

    output_path = args.output or f"analysis_{datetime.now().strftime('%Y%m%d')}.md"

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

    print("Analyzing with Groq...")
    analysis = analyze(periods, args.api_key, mock=args.mock)

    Path(output_path).write_text(analysis, encoding="utf-8")
    print(f"Analysis saved to: {output_path}")

    if args.email_to:
        if not args.email_from or not args.email_password:
            print("Error: --email-from and --email-password are required to send email.")
            sys.exit(1)
        print(f"Sending email to {args.email_to}...")
        subject = f"Training Analysis — {datetime.now().strftime('%d/%m/%Y')}"
        send_analysis(args.email_from, args.email_password, args.email_to, subject, analysis)
        print("Email sent.")


if __name__ == "__main__":
    main()
