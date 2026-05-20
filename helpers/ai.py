"""
helpers/ai.py — Llama a Gemini (Google) para analizar las progresiones.
"""

import google.generativeai as genai

MODEL = "gemini-1.5-flash"

SYSTEM_PROMPT = """Sos un coach de fitness profesional y analista de datos.
Vas a recibir datos estructurados de entrenamiento en el gimnasio a lo largo de varios períodos.
Tu trabajo es analizar progresiones, identificar tendencias y dar recomendaciones concretas.
Respondé en el mismo idioma en que está escrito el mensaje.
Sé específico, referenciá nombres reales de ejercicios y números del data."""


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
    """Llama a Gemini con los datos y retorna el análisis como string."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT)
    prompt = build_prompt(periods)
    response = model.generate_content(prompt)
    return response.text
