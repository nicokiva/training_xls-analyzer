"""
helpers/ai.py — Llama a Groq (LLaMA 3) para analizar las progresiones.
"""

from groq import Groq

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """Sos un coach de fitness profesional y analista de datos.
Vas a recibir datos estructurados de entrenamiento en el gimnasio a lo largo de varios períodos.
Tu trabajo es analizar progresiones, identificar tendencias y dar recomendaciones concretas.
Respondé en el mismo idioma en que está escrito el mensaje.
Sé específico, referenciá nombres reales de ejercicios y números del data."""


def build_prompt(periods):
    """Construye el prompt con los datos de los períodos, omitiendo celdas vacías."""
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
        period_lines = [f"## Período: {period}"]
        has_data = False

        for day_data in period_data["days"]:
            day_lines = [f"\n### Día {day_data['day']}"]

            for ex in day_data["exercises"]:
                ex_lines = []
                for w in ex["weeks"]:
                    series_parts = []
                    for idx, s in enumerate(w["series"]):
                        if s["reps"] or s["peso"]:
                            reps = s["reps"] or "-"
                            peso = s["peso"] or "-"
                            series_parts.append(f"S{idx+1}: {reps}r/{peso}kg")
                    if series_parts:
                        ex_lines.append(f"  Sem{w['week']}: {' | '.join(series_parts)}")

                if ex_lines:
                    has_data = True
                    day_lines.append(f"\n**{ex['name']}**")
                    day_lines.extend(ex_lines)

            if len(day_lines) > 1:
                period_lines.extend(day_lines)

        if has_data:
            lines.extend(period_lines)
            lines.append("")

    return "\n".join(lines)


def _create_client(api_key):
    return Groq(api_key=api_key)


def _call_groq(client, prompt, system=SYSTEM_PROMPT):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content


def analyze(periods, api_key, batch_size=1, mock=False):
    """
    Procesa los períodos uno por uno y genera un análisis final consolidado.
    Con mock=True omite los llamados a Groq y devuelve texto de prueba.
    """
    if mock:
        lines = ["# Análisis de rutinas (MOCK)\n"]
        for p in periods:
            lines.append(f"## {p['period']}\n")
            for day in p["days"]:
                lines.append(f"### Día {day['day']}: {len(day['exercises'])} ejercicios parseados.")
            lines.append("")
        return "\n".join(lines)
    client = _create_client(api_key)
    batch_analyses = []

    for i in range(0, len(periods), batch_size):
        batch = periods[i:i + batch_size]
        labels = " / ".join(p["period"] for p in batch)
        print(f"  Analyzing: {labels}...", flush=True)
        prompt = build_prompt(batch)
        if not prompt.strip().endswith("---") and len(prompt) < 200:
            print(f"    (no data, skipping)")
            continue
        result = _call_groq(client, prompt)
        batch_analyses.append(f"### {labels}\n\n{result}")

    if not batch_analyses:
        return "No hay datos cargados para analizar."

    if len(batch_analyses) == 1:
        return batch_analyses[0]

    # Síntesis final en chunks de 4 para no exceder el límite
    print("  Generating final synthesis...", flush=True)
    chunk_size = 4
    summaries = []
    for i in range(0, len(batch_analyses), chunk_size):
        chunk = batch_analyses[i:i + chunk_size]
        synthesis_prompt = (
            "Resumí en pocas líneas las tendencias clave de estos períodos de entrenamiento "
            "(pesos, progresiones, ejercicios destacados):\n\n"
            + "\n\n---\n\n".join(chunk)
        )
        summaries.append(_call_groq(client, synthesis_prompt))

    final_prompt = (
        "Con base en estos resúmenes de entrenamiento, generá un análisis final completo con:\n"
        "- Tendencias a largo plazo\n"
        "- Ejercicios con mayor y menor progreso\n"
        "- Recomendaciones concretas para los próximos ciclos\n\n"
        + "\n\n---\n\n".join(summaries)
    )
    return _call_groq(client, final_prompt)
