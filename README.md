# routine-analyzer

Analyzes gym training progressions from a Google Sheets spreadsheet using Gemini (Google AI).

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com) → **Get API key**.

## Usage

```bash
python3 analyze.py \
  --sheets-id 1z4N0o6C1zBx7U_Y-G0h6dkqstgyz5dDCQp7MsAVf2WE \
  --credentials /path/to/credentials.json \
  --api-key AIza... \
  --output analysis.md
```

The script reads all tabs from the spreadsheet (newest first), sends the data to Gemini, and saves the analysis as a Markdown file.
