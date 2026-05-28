"""
Pydantic models for structured Gemini output.

Using response_schema forces Gemini to produce type-safe JSON at the token level —
no regex parsing, no hallucinated formats.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, RootModel


class WeightSuggestion(BaseModel):
    exercise: str = Field(
        description="Nombre exacto del ejercicio, idéntico carácter por carácter al recibido en el input."
    )
    day: int = Field(
        description="Número de día de la rutina (1 al 4)."
    )
    weeks: List[float] = Field(
        description="Exactamente 4 números con los kilos para W1, W2, W3 y W4."
    )
    rest_s: Optional[int] = Field(
        None,
        description=(
            "Segundos de descanso como entero (ej. 90, 120). "
            "null si el ejercicio es parte de una superserie/combinado y NO es el último del grupo."
        ),
    )
    reason: str = Field(
        description="Una línea de justificación técnica basada en el historial del atleta."
    )
    progression_analysis: str = Field(
        description=(
            "Análisis de la tendencia histórica del atleta en este ejercicio a lo largo de TODOS los períodos provistos. "
            "Debe mencionar la carga más alta registrada, si la tendencia es ascendente/estancada/descendente, "
            "y si hubo cambio de rango de repeticiones entre períodos (en ese caso, aplicar estimación de 1RM). "
            "Ejemplo: 'Sube de 55kg→67.5kg en 3 meses (6 rep). Nuevo rango 8 rep → estimo 1RM ~76kg → salida W1 60kg.' "
            "Si el ejercicio es nuevo, indicar el ejercicio equivalente del historial usado como referencia."
        )
    )
    rir_w4: str = Field(
        description=(
            "RIR objetivo para la última serie de W4. "
            "Axial/barra libre (Sentadilla, Peso muerto, Press con barra): siempre 'RIR 1-2'. "
            "Máquinas guiadas e isométricos: 'RIR 0 (fallo técnico)'. "
            "Poleas de aislamiento (tríceps, bíceps, depresores): 'RIR 0 + Drop Set'. "
            "Ejercicios de abdomen: 'RIR 1'."
        )
    )
    tempo: str = Field(
        description=(
            "Cadencia de ejecución en formato 'E-P-C-P' (excéntrica-pausa abajo-concéntrica-pausa arriba). "
            "Usá 'Controlado' para ejercicios sin énfasis especial. "
            "Priorizá tempos lentos en excéntrica (ej. '4-0-1-0') para ejercicios de brazos y deltoides."
        )
    )
    challenge: str = Field(
        description=(
            "Desafío técnico o de intensidad para el mes. Debe ser específico y alcanzable. "
            "Ejemplos: 'Drop Set en última serie de W4', 'Batir récord histórico en W4', 'Mantener tempo 4-0-1-0 sin bajar el peso'. "
            "PROHIBIDO sugerir Drop Set o fallo en ejercicios axiales/barra libre (Sentadilla, Peso muerto, Press plano con barra). "
            "Si no hay un desafío real, escribí 'Consolidar técnica y ramp progresivo'."
        )
    )


class WeightSuggestionList(RootModel[List[WeightSuggestion]]):
    """Wrapper so the SDK can handle a top-level JSON array via response_schema."""
    pass
