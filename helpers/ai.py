"""
helpers/ai.py — Llama a Claude (Anthropic) para analizar las progresiones.
"""

import anthropic

MODEL = "claude-opus-4-5"

SYSTEM_PROMPT = """You are a professional fitness coach and data analyst.
You will receive structured gym training data across multiple periods and your job is to analyze progressions, identify trends, and provide actionable recommendations.
Respond in the same language the user writes in.
Be specific, reference actual exercise names and numbers from the data."""


def build_prompt(periods):
    """Construye el prompt con todos los datos de los períodos."""
    lines = [
        "Analizá las siguientes rutinas de entrenamiento registradas a lo largo del tiempo.",
        "Los datos están ordenados del período más reciente al más antiguo.",
        "Incluí en tu análisis:",
        "- Progresión de pesos y repeticiones por ejercicio",
        "- Tendencias generales (mejoras, estancamientos, retrocesos)",
        "- Ejercicios con mayor progreso",
        "- Recomendaciones concretas para los próximos ciclos",
        "",
    ]

    for period_data in periods:
        period = period_data["period"]
        lines.append(f"## Período: {period}")
        for day_data in period_data["days"]:
            lines.append(f"\n### Día {day_data['day']}")
            for ex in day_data["exercises"]:
                lines.append(f"\n**{ex['name']}**")
                for w in ex["weeks"]:
                    week_str = f"  Semana {w['week']}: "
                    series_parts = []
                    for s in w["series"]:
                        reps = s["reps"] or "-"
                        peso = s["peso"] or "-"
                        series_parts.append(f"S{w['series'].index(s)+1}: {reps} reps / {peso} kg")
                    lines.append(week_str + " | ".join(series_parts))
        lines.append("")

    return "\n".join(lines)


def analyze(periods, api_key):
    """Llama a Claude con los datos y retorna el análisis como string."""
    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(periods)

    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text
