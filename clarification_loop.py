"""
Parte 2: Bucle de aclaración (si falta info)

Objetivo
--------
Pedir solo la información que mejora el filtrado. Se generan 3–5 preguntas máximas
basadas en `dudas_priorizadas` y en las `features_actuales`. Se registran las
respuestas, se re-analiza la consulta (Sección 1, idempotente) y se decide si la
suficiencia alcanzó el umbral para avanzar a la Sección 3.

Preferencias del proyecto
-------------------------
- Nombres de variables en inglés.
- Docstrings/comentarios en español.
- Salidas con prints útiles en español.

Integración
-----------
- Este módulo puede importarse desde `main.py`.
- No requiere cambios en `ingest_csv_pgvector.py`.

Interfaces clave
----------------
- `generate_clarifying_questions(...)` → List[str]
- `apply_answers_update_features(...)` → Dict[str, Any]
- `clarification_loop(...)` → ClarificationResult (dataclass)

"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# =============================
# Configuración y tipos de dato
# =============================

@dataclass
class ClarificationResult:
    """Resultado del bucle de aclaración.

    Atributos
    ---------
    questions : List[str]
        Preguntas generadas para el usuario (máx. 5).
    features_updated : Dict[str, Any]
        Diccionario de features actualizado en base a respuestas.
    sufficiency_score : float
        Puntuación [0.0–1.0] que representa cuán suficiente está la información
        para un filtrado robusto.
    sufficiency_reached : bool
        True si la suficiencia alcanza o supera el umbral configurado.
    rationale : str
        Explicación breve de por qué se generaron esas preguntas / puntaje.
    """

    questions: List[str]
    features_updated: Dict[str, Any]
    sufficiency_score: float
    sufficiency_reached: bool
    rationale: str


DEFAULT_SUFFICIENCY_THRESHOLD: float = 0.72
MAX_QUESTIONS: int = 5
MIN_QUESTIONS: int = 3


# ==================================
# Heurísticas para generar preguntas
# ==================================

def _score_feature_impact(missing_feature: str) -> float:
    """Heurística: Asigna un impacto (0–1) por feature faltante.

    Notas
    -----
    - Campos típicos de jurisprudencia con alto impacto en filtrado:
      * materia, fuero, jurisdiccion, tribunal, anio, tipo_proceso,
        etapa, tipo_fallo, instancia, sala.
    - También claves de búsqueda: partes, numero_expediente.
    """
    high = {"materia", "fuero", "jurisdiccion", "tribunal", "anio", "instancia", "sala"}
    medium = {"tipo_proceso", "etapa", "tipo_fallo", "numero_expediente", "partes"}

    if missing_feature in high:
        return 1.0
    if missing_feature in medium:
        return 0.65
    return 0.4


def _question_for_feature(name: str) -> str:
    """Devuelve una pregunta concreta y no ambigua para un feature dado."""
    templates = {
        "materia": "¿La materia es Laboral, Civil, Penal, Comercial, Familia u otra?",
        "fuero": "¿En qué fuero se tramitó la causa (e.g., laboral, civil, penal)?",
        "jurisdiccion": "¿Podés precisar la jurisdicción (p. ej., Tucumán, CABA, PBA)?",
        "tribunal": "¿Recordás el tribunal o sala (p. ej., Cámara del Trabajo Sala II)?",
        "instancia": "¿En qué instancia está el fallo (1ª instancia, Cámara, Corte)?",
        "sala": "Si aplica, ¿qué sala intervino?",
        "anio": "¿De qué año aproximado es la sentencia (p. ej., 2019–2024)?",
        "tipo_proceso": "¿Qué tipo de proceso es (despido, daños, alimentos, etc.)?",
        "tipo_fallo": "¿Qué tipo de resolución buscás (sentencia, auto, medida cautelar)?",
        "etapa": "¿En qué etapa procesal estaba la causa (p. ej., sentencia definitiva)?",
        "numero_expediente": "¿Tenés el número de expediente completo o parcial?",
        "partes": "¿Recordás al menos un nombre de parte (actor/demandado)?",
    }
    return templates.get(name, f"¿Podés precisar el valor de '{name}' para refinar la búsqueda?")


# =============================
# Cálculo de suficiencia/score
# =============================

def _estimate_sufficiency(features: Dict[str, Any]) -> Tuple[float, str]:
    """Calcula un score de suficiencia (0–1) y una breve justificación.

    Criterio simple (ajustable):
    - Peso por cobertura de features clave (high) y medios (medium).
    - Penalización si faltan 3+ features clave.
    - Bonificación si hay `numero_expediente` o múltiples partes.
    """
    present = {k for k, v in features.items() if v not in (None, "", [], {}, "UNKNOWN")}
    high = {"materia", "fuero", "jurisdiccion", "tribunal", "anio", "instancia", "sala"}
    medium = {"tipo_proceso", "etapa", "tipo_fallo", "numero_expediente", "partes"}

    present_high = len(high & present)
    present_medium = len(medium & present)

    # base por cobertura
    score = 0.12 * present_high + 0.07 * present_medium

    # bonificaciones útiles
    if "numero_expediente" in present:
        score += 0.18
    if "partes" in present:
        parts_val = features.get("partes")
        if isinstance(parts_val, (list, tuple)) and len(parts_val) >= 2:
            score += 0.12
        else:
            score += 0.06

    # penalización por faltantes críticos
    missing_high = len(high - present)
    if missing_high >= 3:
        score -= 0.08

    # clamp 0–1
    score = max(0.0, min(1.0, score))

    rationale = (
        f"Cobertura alta={present_high}, media={present_medium}, "
        f"missing_high={missing_high}. "
        f"Expediente={'sí' if 'numero_expediente' in present else 'no'}, "
        f"partes={'sí' if 'partes' in present else 'no'}."
    )
    return score, rationale


# =============================
# API pública del bucle
# =============================

def generate_clarifying_questions(
    dudas_priorizadas: List[str],
    current_features: Dict[str, Any],
    max_questions: int = MAX_QUESTIONS,
) -> List[str]:
    """Genera 3–5 preguntas que realmente mejoren el filtrado.

    Parámetros
    ----------
    dudas_priorizadas : List[str]
        Lista de nombres de features (en orden de prioridad) que podrían faltar o
        necesitar precisión.
    current_features : Dict[str, Any]
        Estado actual de features extraídos en la Sección 1.
    max_questions : int
        Máximo de preguntas (por defecto 5).

    Retorna
    -------
    List[str]
        Preguntas concretas y accionables. Entre 3 y 5 si es posible.
    """
    missing = []
    for feat in dudas_priorizadas:
        val = current_features.get(feat)
        if val in (None, "", [], {}, "UNKNOWN"):
            missing.append(feat)

    # Ordenar por impacto
    missing_sorted = sorted(missing, key=_score_feature_impact, reverse=True)
    n = max(MIN_QUESTIONS, min(max_questions, len(missing_sorted)))

    questions = [_question_for_feature(name) for name in missing_sorted[:n]]
    return questions


def apply_answers_update_features(
    current_features: Dict[str, Any],
    answers: Dict[str, Any],
) -> Dict[str, Any]:
    """Aplica respuestas del usuario para actualizar features.

    Notas
    -----
    - Normaliza valores básicos (e.g., `anio` a int si es posible).
    - Simplifica strings (strip, lower donde corresponda).
    """
    updated = dict(current_features)

    for k, v in answers.items():
        if v is None:
            continue
        if k == "anio":
            try:
                updated[k] = int(str(v).strip())
            except Exception:
                updated[k] = v
            continue
        if isinstance(v, str):
            val = v.strip()
            # normalizaciones ligeras por campo
            if k in {"materia", "fuero", "jurisdiccion", "tribunal", "instancia", "sala", "tipo_proceso", "tipo_fallo", "etapa"}:
                updated[k] = val.lower()
            else:
                updated[k] = val
        else:
            updated[k] = v

    return updated


def clarification_loop(
    dudas_priorizadas: List[str],
    current_features: Dict[str, Any],
    sufficiency_threshold: float = DEFAULT_SUFFICIENCY_THRESHOLD,
    provided_answers: Optional[Dict[str, Any]] = None,
    reanalyze_fn: Optional[callable] = None,
) -> ClarificationResult:
    """Ejecuta el bucle de aclaración y decide si se alcanzó la suficiencia.

    Parámetros
    ----------
    dudas_priorizadas : List[str]
        Features ordenadas por importancia a precisar.
    current_features : Dict[str, Any]
        Estado actual (Sección 1).
    sufficiency_threshold : float
        Umbral de suficiencia para avanzar a Sección 3.
    provided_answers : Optional[Dict[str, Any]]
        Respuestas ya conocidas (modo programático / API). Si es None, se retornan
        preguntas y el caller deberá recolectar respuestas.
    reanalyze_fn : Optional[callable]
        Función de re-análisis de Sección 1 para idempotencia. Firma esperada:
        `reanalyze_fn(features: Dict[str, Any]) -> Dict[str, Any]`.

    Retorna
    -------
    ClarificationResult
        Con preguntas, features actualizadas y decisión de suficiencia.
    """
    # 1) Generar preguntas
    questions = generate_clarifying_questions(dudas_priorizadas, current_features)

    # 2) Si ya tenemos respuestas (API programática), aplicar y re-analizar
    if provided_answers is not None:
        updated = apply_answers_update_features(current_features, provided_answers)
        if reanalyze_fn is not None:
            try:
                updated = reanalyze_fn(updated)
            except Exception as e:
                print(f"⚠️ Error en reanalyze_fn (Sección 1): {e}")
        score, rationale = _estimate_sufficiency(updated)
        reached = score >= sufficiency_threshold
        return ClarificationResult(
            questions=[],  # ya respondidas en modo programático
            features_updated=updated,
            sufficiency_score=score,
            sufficiency_reached=reached,
            rationale=rationale,
        )

    # 3) Sin respuestas: devolver preguntas y un estado provisional
    score, rationale = _estimate_sufficiency(current_features)
    reached = score >= sufficiency_threshold
    return ClarificationResult(
        questions=questions,
        features_updated=current_features,
        sufficiency_score=score,
        sufficiency_reached=reached,
        rationale=rationale,
    )


# =============================
# Ejemplo de integración en main
# =============================
if __name__ == "__main__":
    # ⚠️ Solo para demo por consola; en producción usar desde `main.py` del proyecto.
    base_features = {
        "materia": None,
        "fuero": None,
        "jurisdiccion": "tucumán",
        "tribunal": None,
        "anio": None,
        "instancia": None,
        "tipo_proceso": None,
        "partes": [],
        "numero_expediente": None,
    }
    dudas = [
        "materia",
        "fuero",
        "tribunal",
        "anio",
        "instancia",
        "tipo_proceso",
        "numero_expediente",
        "partes",
    ]

    result = clarification_loop(dudas, base_features)
    print("\n➡️ Preguntas sugeridas (demo):")
    for i, q in enumerate(result.questions, 1):
        print(f"{i}. {q}")
    print(f"\nSuficiencia actual: {result.sufficiency_score:.2f} (umbral {DEFAULT_SUFFICIENCY_THRESHOLD})")
    print(f"Rationale: {result.rationale}")
