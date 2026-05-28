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


class WeightSuggestionList(RootModel[List[WeightSuggestion]]):
    """Wrapper so the SDK can handle a top-level JSON array via response_schema."""
    pass
