"""
helpers/ai.py — Integración con Groq para análisis de progresiones de entrenamiento.

Responsabilidades:
  - Transformar los datos de períodos en un prompt ejercicio-céntrico.
  - Llamar al modelo LLaMA 3 via Groq API con ese prompt.
  - Retornar el análisis como string listo para guardar.

Diseño del prompt:
  En lugar de analizar período por período, el prompt agrupa los datos
  por ejercicio y muestra la evolución cronológica de cada uno. Esto le
  permite a la IA detectar tendencias a largo plazo en vez de comparar
  snapshots aislados.

  Ejemplo de lo que ve la IA:
    **Empuje de pecho con barra en banco plano**
      15/07/25-17/08/25: Sem1: S1:12r/60kg S2:10r/60kg | Sem2: S1:12r/65kg
      18/08/25-12/09/25: Sem1: S1:12r/65kg S2:10r/70kg | ...
      ...

  Solo se incluyen celdas con datos cargados (reps o peso no vacíos),
  lo que reduce drásticamente el tamaño del prompt comparado con enviar
  toda la grilla cruda.
"""

from groq import Groq

# Modelo de Groq a usar. llama-3.3-70b-versatile ofrece el mejor balance
# entre calidad de análisis y velocidad en el free tier.
MODEL = "llama-3.3-70b-versatile"

# Instrucción de sistema que define el rol y comportamiento del modelo.
# Le explica explícitamente el significado de cada campo del spreadsheet
# para que el análisis sea correcto.
SYSTEM_PROMPT = """Sos un coach de fitness profesional y analista de datos.

Vas a recibir datos de entrenamiento de un gimnasio a lo largo de varios períodos.

Cómo interpretar los datos:
- "Rep." = repeticiones REALIZADAS por el usuario en esa serie (no las esperadas).
- "Peso" = peso usado en kg. A veces contiene notas de texto en lugar de o además del número
  (ej: "8 agarre / 2 sin agarre", "3 con 3kg / 5 sin peso"). Estas notas son observaciones
  importantes del usuario sobre cómo fue la serie — tenelas en cuenta en el análisis.
- Si "Peso" es "0" o vacío, el ejercicio se hizo con peso corporal o sin carga externa.
- Los datos están ordenados: Semana 1 → 2 → 3 → 4, con 3 series por semana.

Tu trabajo:
- Analizar la progresión global de cada ejercicio a lo largo del tiempo.
- Identificar tendencias reales (mejoras de peso, más reps, notas que indican dificultad o facilidad).
- Detectar estancamientos o retrocesos con evidencia concreta de los números.
- Dar recomendaciones específicas y accionables para los próximos ciclos.

Respondé en español. Sé concreto y referenciá ejercicios y números reales de los datos."""


def _create_client(api_key):
    """Crea y retorna el cliente autenticado de Groq."""
    return Groq(api_key=api_key)


def _call_groq(client, prompt):
    """
    Hace una llamada al modelo de Groq y retorna el texto de la respuesta.

    Args:
        client: Instancia de Groq autenticada.
        prompt: Texto del mensaje a enviar al modelo.

    Returns:
        String con la respuesta generada por el modelo.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content


def build_prompt(periods):
    """
    Construye el prompt para la IA a partir de los datos de todos los períodos.

    En vez de estructurar el prompt por período (cronograma), lo estructura
    por ejercicio (progresión). Cada ejercicio aparece una sola vez con todas
    sus entradas históricas ordenadas de más antiguo a más reciente.

    Solo se incluyen semanas/series que tengan al menos un dato cargado
    (reps o peso), para no desperdiciar tokens en celdas vacías.

    Args:
        periods: Lista de dicts con formato:
            [{"period": "18/05/26-14/06/26", "days": [...]}, ...]
            (orden: más reciente primero, como viene del spreadsheet)

    Returns:
        String listo para enviar como mensaje a la IA.
    """
    # La IA necesita ver la progresión de más antiguo a más reciente
    # para detectar tendencias correctamente. Los períodos llegan en orden
    # inverso (más reciente primero), por eso se invierte acá.
    ordered = list(reversed(periods))

    # Construir un dict { nombre_ejercicio: [ {period, data}, ... ] }
    # agrupando todas las apariciones de cada ejercicio a través del tiempo.
    exercise_history = {}
    for period_data in ordered:
        period = period_data["period"]
        for day_data in period_data["days"]:
            for ex in day_data["exercises"]:
                name = ex["name"]

                # Construir string compacto con solo los datos cargados.
                # Formato: "Sem1: S1:12r/60kg S2:10r/60kg | Sem2: S1:12r/65kg"
                weeks_with_data = []
                for w in ex["weeks"]:
                    series_parts = []
                    for idx, s in enumerate(w["series"]):
                        if s["reps"] or s["peso"]:
                            reps = s["reps"] or "-"
                            peso = s["peso"] or "-"
                            series_parts.append(f"S{idx+1}:{reps}r/{peso}kg")
                    if series_parts:
                        weeks_with_data.append(f"Sem{w['week']}: {' '.join(series_parts)}")

                if weeks_with_data:
                    if name not in exercise_history:
                        exercise_history[name] = []
                    exercise_history[name].append({
                        "period": period,
                        "data": " | ".join(weeks_with_data),
                    })

    # Armar el texto del prompt con instrucciones + datos por ejercicio
    lines = [
        "Analizá la progresión completa de los siguientes ejercicios a lo largo del tiempo.",
        "Los datos están ordenados cronológicamente (más antiguo → más reciente).",
        "Generá un análisis global (no período a período) que incluya:",
        "- Tendencia general de cada ejercicio (mejora, estancamiento, retroceso)",
        "- Ejercicios con mayor y menor progreso",
        "- Observaciones sobre volumen y consistencia",
        "- Recomendaciones concretas para los próximos ciclos",
        "",
    ]

    for name, history in exercise_history.items():
        lines.append(f"**{name}**")
        for entry in history:
            lines.append(f"  {entry['period']}: {entry['data']}")
        lines.append("")

    return "\n".join(lines)


def analyze(periods, api_key, mock=False, **kwargs):
    """
    Genera un análisis global de la progresión de todos los ejercicios.

    Construye un prompt ejercicio-céntrico con todos los datos históricos
    y hace una sola llamada a Groq para obtener el análisis completo.

    Args:
        periods:  Lista de períodos con sus datos de ejercicios.
                  Formato: [{"period": str, "days": [...]}, ...]
        api_key:  API key de Groq (obtenida en console.groq.com).
        mock:     Si True, saltea la llamada a la API y retorna un texto
                  de prueba. Útil para testear el flujo sin gastar tokens.
        **kwargs: Ignorado (para compatibilidad futura).

    Returns:
        String con el análisis en Markdown.
    """
    if mock:
        exercise_count = sum(
            len(day["exercises"])
            for p in periods for day in p["days"]
        )
        return (
            f"# Análisis de rutinas (MOCK)\n\n"
            f"Se analizarían {len(periods)} períodos con un total de "
            f"{exercise_count} registros de ejercicios.\n"
        )

    client = _create_client(api_key)
    prompt = build_prompt(periods)
    print("  Sending full progression data to Groq...", flush=True)
    return _call_groq(client, prompt)

