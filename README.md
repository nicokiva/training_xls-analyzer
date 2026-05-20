# routine-analyzer

AI-powered gym training analyzer. Reads Google Sheets training data, sends it to Groq (LLaMA 3.3 70b), translates the output to Spanish, saves the result to a `.md` file and emails it.

## What it does

The script supports four analysis modes:

| Mode | Trigger | What it does |
|---|---|---|
| `global` | Manual / pre-upload | Full history, stagnation detection |
| `monthly` | Cron day 1 / pre-upload | How was the month? |
| `new-routine` | Post PDF upload | Does the new routine fit the goal? |
| `weekly` | Cron Sundays | This week vs last week |

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the values:

```
SHEETS_ID=<your Google Sheets spreadsheet ID>
CREDENTIALS=<path to service account JSON>
GROQ_API_KEY=<your Groq API key>
EMAIL_TO=<recipient email>
EMAIL_FROM=<your Gmail address>
EMAIL_PASSWORD=<Gmail App Password>
GOAL=<your training goal, e.g. "gain muscle mass">
```

- Free Groq API key: [console.groq.com](https://console.groq.com) → **API Keys** → **Create API Key**
- Gmail App Password: [myaccount.google.com](https://myaccount.google.com) → Security → App Passwords

## Usage

```bash
# Single mode
python3 analyze.py --mode global
python3 analyze.py --mode weekly
python3 analyze.py --mode new-routine --mock   # test without Groq

# Consume pending events from events.db (triggered by pdf2xls-generator)
python3 analyze.py
```

## All options

| Flag | Default | Description |
|---|---|---|
| `--mode` | (none) | If omitted, consumes pending events from `events.db` |
| `--goal` | from `.env` | Training goal injected into prompts |
| `--mock` | false | Skip Groq, use fake output |
| `--max-periods` | all | Limit to N most recent periods |
| `--sheets-id` | from `.env` | Google Sheets ID |
| `--credentials` | from `.env` | Path to service account JSON |
| `--api-key` | from `.env` | Groq API key |
| `--email-to` | from `.env` | Recipient email address |
| `--email-from` | from `.env` | Gmail address to send from |
| `--email-password` | from `.env` | Gmail App Password (16 chars) |

## Prompt templates

Prompts live in `templates/*.txt` with `{placeholders}`. Edit them without touching any Python code:

- `system.txt` — base AI instructions
- `global.txt`, `monthly.txt`, `new-routine.txt`, `weekly.txt`, `weekly_first.txt`

## Event queue integration

When `pdf2xls-generator` uploads a PDF it publishes events to `../events.db`. Running `python3 analyze.py` (no `--mode`) consumes those pending events automatically and runs the appropriate analysis for each one.

Event type constants are shared via the `training-shared` package — no magic strings in either project.

## Project structure

```
analyze.py           ← entry point
templates/           ← editable prompt templates (.txt)
analyses/            ← output .md files (auto-created)
helpers/
  ai/                ← Groq API, prompt builders, translation
  reader/            ← Google Sheets auth and data loading
  mailer/            ← Gmail sending
  events/            ← SQLite event consumer
```
