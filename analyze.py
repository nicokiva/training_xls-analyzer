#!/usr/bin/env python3
"""
analyze.py — Analiza las progresiones de rutinas de entrenamiento usando IA.

Lee todos los tabs del Google Sheets (más reciente primero), genera un análisis
completo con Claude y lo guarda en un archivo Markdown.

Uso:
    python3 analyze.py \\
        --sheets-id SPREADSHEET_ID \\
        --credentials /path/to/credentials.json \\
        --api-key ANTHROPIC_API_KEY \\
        [--output analysis.md]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from helpers.reader import get_service, load_all_periods
from helpers.ai import analyze


def main():
    parser = argparse.ArgumentParser(
        description="Generate an AI-powered analysis of gym training progressions."
    )
    parser.add_argument("--sheets-id", required=True, help="Google Sheets spreadsheet ID")
    parser.add_argument("--credentials", required=True, help="Path to Google service account JSON")
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .md file (default: analysis_YYYYMMDD.md)",
    )
    args = parser.parse_args()

    output_path = args.output or f"analysis_{datetime.now().strftime('%Y%m%d')}.md"

    print("Connecting to Google Sheets...")
    service = get_service(args.credentials)

    print("Loading training periods...")
    periods = load_all_periods(service, args.sheets_id)

    if not periods:
        print("No data found in the spreadsheet.")
        sys.exit(1)

    print(f"Found {len(periods)} period(s): {', '.join(p['period'] for p in periods)}")

    print("Analyzing with Claude...")
    analysis = analyze(periods, args.api_key)

    Path(output_path).write_text(analysis, encoding="utf-8")
    print(f"Analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
