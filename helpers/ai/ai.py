"""
helpers/ai.py — Integración con Groq para análisis de progresiones de entrenamiento.

Responsabilidades:
  - Transformar los datos de períodos en prompts específicos por modo.
  - Llamar al modelo LLaMA 3 via Groq API con esos prompts.
  - Retornar el análisis como string listo para guardar.

Modos disponibles:
  - global:       Análisis completo de todo el historial. Detecta tendencias,
                  estancamientos y evalúa si se está cumpliendo el objetivo.
  - new-routine:  Analiza la nueva rutina (recién generada, sin datos de ejecución)
                  contra el historial. ¿Sirve para el objetivo? ¿Qué cambiaría?
  - monthly:      Balance mensual del período más reciente con ejecución completa.
                  ¿Cómo fue el mes? ¿Se cumplió el objetivo?
  - weekly:       Compara la semana actual con la anterior en el período activo.
                  ¿Fue una buena semana? ¿Mejoró?

Diseño del prompt ejercicio-céntrico (modo global/monthly):
  En lugar de analizar período por período, agrupa los datos por ejercicio
  y muestra la evolución cronológica. Esto permite detectar tendencias a
  largo plazo en vez de comparar snapshots aislados.
"""

from groq import Groq

MODEL = "llama-3.3-70b-versatile"

_BASE_SYSTEM_PROMPT = """Sos un coach de fitness profesional y analista de datos de entrenamiento.

Cómo interpretar los datos:
- "Rep." = repeticiones REALIZADAS por el usuario en esa serie (no las esperadas).
- "Peso" = peso usado en kg. A veces contiene notas de texto en lugar de o además del número
  (ej: "8 agarre / 2 sin agarre", "3 con 3kg / 5 sin peso"). Estas notas son observaciones
  importantes del usuario sobre cómo fue la serie — tenelas en cuenta en el análisis.
- Si "Peso" es "0" o vacío, el ejercicio se hizo con peso corporal o sin carga externa.
- Los datos están ordenados: Semana 1 → 2 → 3 → 4, con 3 series por semana.

Respondé en español. Sé concreto y referenciá ejercicios y números reales de los datos.
No uses frases genéricas — cada observación debe estar respaldada por datos específicos."""


def _make_system_prompt(goal):
    return f"{_BASE_SYSTEM_PROMPT}\n\nObjetivo del usuario: **{goal}**."


def _create_client(api_key):
    """Crea y retorna el cliente autenticado de Groq."""
    return Groq(api_key=api_key)


def _call_groq(client, system_prompt, user_prompt):
    """
    Hace una llamada al modelo de Groq y retorna el texto de la respuesta.

    Args:
        client:        Instancia de Groq autenticada.
        system_prompt: Instrucción de sistema con rol y contexto.
        user_prompt:   Datos y pregunta a enviar al modelo.

    Returns:
        String con la respuesta generada por el modelo.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_exercise_history(periods):
    """
    Genera el bloque de texto con el historial ejercicio-céntrico.
    Agrupa todas las apariciones de cada ejercicio ordenadas cronológicamente.
    Solo incluye semanas/series con al menos un dato cargado.
    """
    ordered = list(reversed(periods))
    exercise_history = {}

    for period_data in ordered:
        period = period_data["period"]
        for day_data in period_data["days"]:
            for ex in day_data["exercises"]:
                name = ex["name"]
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
                        "data":   " | ".join(weeks_with_data),
                    })

    lines = []
    for name, history in exercise_history.items():
        lines.append(f"**{name}**")
        for entry in history:
            lines.append(f"  {entry['period']}: {entry['data']}")
        lines.append("")
    return "\n".join(lines)


def _format_routine_structure(period):
    """
    Formatea la estructura de ejercicios de un período sin datos de ejecución.
    Útil para new-routine donde el tab todavía no tiene reps/pesos cargados.
    """
    lines = []
    for day in period["days"]:
        lines.append(f"Día {day['day']}:")
        for ex in day["exercises"]:
            lines.append(f"  - {ex['name']}")
        lines.append("")
    return "\n".join(lines)


def _format_week_data(week_data, week_label):
    """Formatea los datos de una semana (output de extract_week_data)."""
    lines = [f"**{week_label}**"]
    for day in week_data:
        lines.append(f"  Día {day['day']}:")
        for ex in day["exercises"]:
            series_str = "  ".join(
                f"S{i+1}:{s['reps'] or '-'}r/{s['peso'] or '-'}kg"
                for i, s in enumerate(ex["series"])
                if s["reps"] or s["peso"]
            )
            lines.append(f"    {ex['name']}: {series_str}")
    lines.append("")
    return "\n".join(lines)


def build_global_prompt(periods, goal):
    """
    Prompt para el modo global: análisis completo de todo el historial.

    Args:
        periods: Todos los períodos disponibles (más reciente primero).
        goal:    Objetivo del usuario (ej: "hipertrofia").

    Returns:
        String del prompt listo para enviar a la IA.
    """
    history_block = _format_exercise_history(periods)
    return (
        f"Analizá la progresión completa de los siguientes ejercicios a lo largo del tiempo.\n"
        f"Los datos están ordenados cronológicamente (más antiguo → más reciente).\n\n"
        f"Generá un análisis global orientado al objetivo de **{goal}** que incluya:\n"
        f"- Tendencia general de cada ejercicio (mejora, estancamiento, retroceso)\n"
        f"- Ejercicios con mayor y menor progreso\n"
        f"- Evaluación de si el historial apunta a cumplir el objetivo de {goal}\n"
        f"- Señales de estancamiento con evidencia concreta\n"
        f"- Recomendaciones concretas para los próximos ciclos\n\n"
        f"{history_block}"
    )


def build_new_routine_prompt(periods, goal):
    """
    Prompt para el modo new-routine: evalúa la rutina recién generada.

    El primer período es la nueva rutina (sin datos de ejecución todavía).
    El resto son el historial de períodos ejecutados.

    Args:
        periods: Lista con al menos 1 período. periods[0] = nueva rutina.
        goal:    Objetivo del usuario.

    Returns:
        String del prompt listo para enviar a la IA.
    """
    new_period = periods[0]
    history    = periods[1:]

    routine_block  = _format_routine_structure(new_period)
    history_block  = _format_exercise_history(history) if history else "(sin historial previo)"

    return (
        f"Se acaba de generar una nueva rutina de entrenamiento.\n"
        f"El objetivo del usuario es: **{goal}**.\n\n"
        f"## Nueva rutina ({new_period['period']})\n\n"
        f"{routine_block}\n"
        f"## Historial de entrenamiento previo\n\n"
        f"{history_block}\n"
        f"Analizá:\n"
        f"1. ¿Esta rutina es adecuada para el objetivo de {goal}? ¿Por qué?\n"
        f"2. ¿Hay ejercicios que no aportan al objetivo o podrían mejorarse?\n"
        f"3. ¿Qué cambios concretos harías a esta rutina dado el historial del usuario?\n"
        f"4. ¿Hay patrones del historial que esta rutina aprovecha bien o ignora?\n"
    )


def build_monthly_prompt(periods, goal):
    """
    Prompt para el modo monthly: balance del mes más reciente.

    El primer período es el mes a analizar (con datos de ejecución completos).
    El resto son el historial anterior para contexto.

    Args:
        periods: Lista de períodos. periods[0] = mes a analizar.
        goal:    Objetivo del usuario.

    Returns:
        String del prompt listo para enviar a la IA.
    """
    current_period = periods[0]
    history        = periods[1:]

    current_block = _format_exercise_history([current_period])
    history_block = _format_exercise_history(history) if history else "(sin historial previo)"

    return (
        f"Hacé un balance del mes de entrenamiento **{current_period['period']}**.\n"
        f"El objetivo del usuario es: **{goal}**.\n\n"
        f"## Datos del mes\n\n"
        f"{current_block}\n"
        f"## Historial anterior (contexto)\n\n"
        f"{history_block}\n"
        f"Analizá:\n"
        f"1. ¿Se cumplió el objetivo de {goal} este mes? ¿Qué evidencia hay en los números?\n"
        f"2. ¿Qué ejercicios progresaron bien? ¿Cuáles se estancaron o retrocedieron?\n"
        f"3. ¿Cómo fue la consistencia y el volumen comparado con meses anteriores?\n"
        f"4. ¿Qué ajustes recomendás para el próximo mes?\n"
    )


def build_weekly_prompt(period, current_week_data, prev_week_data, current_week_num, goal):
    """
    Prompt para el modo weekly: compara la semana actual con la anterior.

    Args:
        period:            Período activo (dict con "period" key).
        current_week_data: Output de extract_week_data para la semana actual.
        prev_week_data:    Output de extract_week_data para la semana anterior.
                           Puede ser None si es la primera semana del período.
        current_week_num:  Número de semana actual (1-based, para mostrar en el prompt).
        goal:              Objetivo del usuario.

    Returns:
        String del prompt listo para enviar a la IA.
    """
    current_block = _format_week_data(current_week_data, f"Semana {current_week_num} (actual)")

    if prev_week_data:
        prev_block = _format_week_data(prev_week_data, f"Semana {current_week_num - 1} (anterior)")
        comparison = (
            f"## Semana anterior\n\n{prev_block}\n"
            f"## Semana actual\n\n{current_block}\n"
            f"Comparando con la semana anterior, analizá:\n"
            f"1. ¿Mejoró el rendimiento general esta semana?\n"
            f"2. ¿Qué ejercicios mejoraron (más peso o más reps)? ¿Cuáles bajaron?\n"
            f"3. ¿Fue una buena semana para el objetivo de {goal}?\n"
            f"4. ¿Qué ajustes recomendás para la semana que viene?\n"
        )
    else:
        comparison = (
            f"## Semana actual (primera del período)\n\n{current_block}\n"
            f"Es la primera semana del período, no hay semana anterior para comparar.\n"
            f"Analizá:\n"
            f"1. ¿Cómo arrancó el período en relación al objetivo de {goal}?\n"
            f"2. ¿Hay algo llamativo en los números de esta primera semana?\n"
            f"3. ¿Qué recomendás para la semana que viene?\n"
        )

    return (
        f"Análisis semanal del período **{period['period']}**.\n"
        f"Objetivo del usuario: **{goal}**.\n\n"
        f"{comparison}"
    )


# ---------------------------------------------------------------------------
# Mock outputs
# ---------------------------------------------------------------------------

_MOCK_OUTPUTS = {
    "global": """\
# Análisis global (MOCK)

> ⚠️ Análisis de prueba — los datos son reales pero el análisis es inventado.

## Tendencias generales

- **Press plano con barra**: progresión sostenida de ~60kg a ~75kg a lo largo del año. ✅
- **Sentadilla clásica**: estancamiento en semanas 2-3, sin variación de peso en los últimos 2 períodos. ⚠️
- **Dominada estricta**: retroceso leve, bajó de 8 reps a 6 en el último período. ❌

## Evaluación del objetivo (hipertrofia)

El volumen total aumentó un 15% en 6 meses. La progresión de carga en tren superior es compatible con hipertrofia. Tren inferior muestra estancamiento que limita el objetivo.

## Recomendaciones

1. Aumentar carga en sentadilla — 2 períodos sin cambios.
2. Revisar técnica en dominadas antes de subir el volumen.
3. Mantener la progresión en press plano, está funcionando bien.

---
*Corré sin `--mock` para obtener el análisis real generado por IA.*""",

    "new-routine": """\
# Nueva rutina (MOCK)

> ⚠️ Análisis de prueba — los datos son reales pero el análisis es inventado.

## ¿Sirve para hipertrofia?

La rutina tiene buena estructura: 4 días con separación de grupos musculares clara. La distribución de ejercicios compuestos + aislamiento es compatible con hipertrofia.

## Puntos fuertes

- Press plano + inclinado cubre bien el pecho en distintos ángulos.
- Sentadilla como ejercicio base de pierna es ideal para hipertrofia.

## Cambios sugeridos

1. Reemplazar "Curl predicador" por "Curl martillo" — el historial muestra más consistencia con agarre neutro.
2. Agregar un ejercicio de isquiotibiales (peso muerto rumano) — el historial no los trabaja en 3 períodos.

---
*Corré sin `--mock` para obtener el análisis real generado por IA.*""",

    "monthly": """\
# Balance mensual (MOCK)

> ⚠️ Análisis de prueba — los datos son reales pero el análisis es inventado.

## ¿Se cumplió el objetivo de hipertrofia?

Parcialmente. El volumen fue alto (semanas 1-3) pero cayó en semana 4, probablemente por fatiga acumulada.

## Progresiones del mes

- ✅ Press plano: +5kg respecto al mes anterior en semana 3.
- ✅ Remo con barra: +2 reps promedio en todas las semanas.
- ⚠️ Sentadilla: peso estable, sin progresión.

## Recomendaciones para el próximo mes

1. Planificar deload en semana 4 — la caída de rendimiento es recurrente.
2. Aumentar carga en sentadilla al menos un 5%.

---
*Corré sin `--mock` para obtener el análisis real generado por IA.*""",

    "weekly": """\
# Análisis semanal (MOCK)

> ⚠️ Análisis de prueba — los datos son reales pero el análisis es inventado.

## ¿Mejoró respecto a la semana anterior?

Sí, en general. 4 de 6 ejercicios principales mejoraron en peso o reps.

## Detalle

- ✅ Press plano: 70kg → 72.5kg en serie 1. Buena progresión.
- ✅ Dominadas: 6 → 7 reps en todas las series.
- ⚠️ Sentadilla: igual que la semana anterior (60kg × 10).
- ❌ Curl de bíceps: bajó 1 rep en series 2 y 3 — posible fatiga.

## Para la semana que viene

1. Intentar 75kg en press plano en la primera serie.
2. Agregar 2.5kg en sentadilla.
3. Descansá bien antes del día de bíceps.

---
*Corré sin `--mock` para obtener el análisis real generado por IA.*""",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(periods, api_key, mock=False, mode="global", goal="hipertrofia",
            current_week_data=None, prev_week_data=None, current_week_num=None):
    """
    Genera un análisis de entrenamiento según el modo solicitado.

    Args:
        periods:          Lista de períodos (más reciente primero).
        api_key:          API key de Groq.
        mock:             Si True, retorna un análisis de prueba sin llamar a la API.
        mode:             Modo de análisis: 'global', 'new-routine', 'monthly', 'weekly'.
        goal:             Objetivo del usuario (ej: 'hipertrofia').
        current_week_data: Datos de la semana actual (solo modo 'weekly').
        prev_week_data:   Datos de la semana anterior (solo modo 'weekly', puede ser None).
        current_week_num: Número de semana actual 1-based (solo modo 'weekly').

    Returns:
        String con el análisis en Markdown.
    """
    if mock:
        return _MOCK_OUTPUTS.get(mode, _MOCK_OUTPUTS["global"])

    if mode == "new-routine":
        prompt = build_new_routine_prompt(periods, goal)
    elif mode == "monthly":
        prompt = build_monthly_prompt(periods, goal)
    elif mode == "weekly":
        prompt = build_weekly_prompt(
            periods[0], current_week_data, prev_week_data, current_week_num, goal
        )
    else:
        prompt = build_global_prompt(periods, goal)

    client = _create_client(api_key)
    system = _make_system_prompt(goal)
    print(f"  Sending [{mode}] prompt to Groq...", flush=True)
    return _call_groq(client, system, prompt)


