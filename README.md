# routine-analyzer

Analyzes gym training progressions from a Google Sheets spreadsheet using Groq (LLaMA 3).

## Setup

```bash
pip install -r requirements.txt
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) → **API Keys** → **Create API Key**.

For Gmail, create an App Password at [myaccount.google.com](https://myaccount.google.com) → Security → App Passwords.

## Usage

**Analysis only (saved to .md):**
```bash
python3 analyze.py \
  --sheets-id 1z4N0o6C1zBx7U_Y-G0h6dkqstgyz5dDCQp7MsAVf2WE \
  --credentials ../../rutinas-entrenamiento-496600-cfbbb2bb0b5c.json \
  --api-key gsk_... \
  --output analysis.md
```

**Analysis + send by email:**
```bash
python3 analyze.py \
  --sheets-id 1z4N0o6C1zBx7U_Y-G0h6dkqstgyz5dDCQp7MsAVf2WE \
  --credentials ../../rutinas-entrenamiento-496600-cfbbb2bb0b5c.json \
  --api-key gsk_... \
  --email-to vos@gmail.com \
  --email-from vos@gmail.com \
  --email-password "xxxx xxxx xxxx xxxx" \
  --output analysis.md
```

**Test without spending tokens:**
```bash
python3 analyze.py \
  --sheets-id 1z4N0o6C1zBx7U_Y-G0h6dkqstgyz5dDCQp7MsAVf2WE \
  --credentials ../../rutinas-entrenamiento-496600-cfbbb2bb0b5c.json \
  --api-key dummy \
  --mock
```

## Options

| Flag | Description |
|------|-------------|
| `--sheets-id` | Google Sheets spreadsheet ID |
| `--credentials` | Path to Google service account JSON |
| `--api-key` | Groq API key |
| `--output` | Output .md file (default: `analysis_YYYYMMDD.md`) |
| `--max-periods` | Limit to N most recent periods (default: all) |
| `--mock` | Skip Groq API, write fake output (for testing) |
| `--email-to` | Recipient email address |
| `--email-from` | Gmail address to send from |
| `--email-password` | Gmail App Password (16 chars) |

