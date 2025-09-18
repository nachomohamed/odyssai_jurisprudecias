"""
Parche para `main.py`:
- Agrega Sección 1 (extracción básica de features desde el user_input)
- Integra el bucle de aclaración (clarification_loop)
- Maneja estado de preguntas/respuestas en Streamlit

Instrucciones rápidas:
1) Guardar `clarification_loop.py` (del canvas anterior) junto a `main.py`.
2) Abrir `main.py` y aplicar:
   - Import nuevo
   - Funciones nuevas: extract_features_basic, parse_answer_like, reanalizar_seccion1, maybe_refine_query
   - Estado y flujo en la zona de manejo del `user_input`

El parche está organizado en bloques con comentarios: `# >>> ADD` y `# >>> MODIFY`.
"""

# =============================
# 1) IMPORTS —>>> ADD debajo de tus imports
# =============================
# from clarification_loop import clarification_loop

# EXTRA imports que usa Sección 1
# import re
# from typing import Any, Optional, Tuple


# =============================
# 2) SECCIÓN 1: extracción básica —>>> ADD (por ejemplo debajo de "Research helpers")
# =============================
# Mapeos ligeros para detectar valores por palabras clave
# _MATERIA_LEX = {
#     "laboral": "laboral",
#     "civil": "civil",
#     "penal": "penal",
#     "comercial": "comercial",
#     "familia": "familia",
# }
# _FUERO_LEX = {
#     "laboral": "laboral",
#     "civil": "civil",
#     "penal": "penal",
#     "contencioso": "contencioso",
# }
# _JURIS_LEX = {
#     "tucuman": "tucumán",
#     "caba": "caba",
#     "buenos aires": "pba",
# }

# def extract_features_basic(user_query: str) -> Tuple[dict, list]:
#     """Heurística mínima para armar `features_actuales` y `dudas_priorizadas`.
#     - Busca año (1980–2030), número de expediente (patrones simples), y claves por palabras.
#     - Devuelve features con defaults None y una lista ordenada de dudas prioritarias.
#     """
#     q = (user_query or "").lower()
#     feats = {
#         "materia": None,
#         "fuero": None,
#         "jurisdiccion": None,
#         "tribunal": None,
#         "instancia": None,
#         "sala": None,
#         "anio": None,
#         "tipo_proceso": None,
#         "tipo_fallo": None,
#         "etapa": None,
#         "numero_expediente": None,
#         "partes": [],
#     }
#     # año
#     m = re.search(r"\b(19\d{2}|20[0-3]\d)\b", q)
#     if m:
#         try:
#             feats["anio"] = int(m.group(1))
#         except:
#             pass
#     # nro expediente (muy laxo)
#     m = re.search(r"expediente\s*[:#-]?\s*([\w./-]{5,})", q)
#     if m:
#         feats["numero_expediente"] = m.group(1).strip()
#     # materia / fuero / jurisdicción
#     for k, v in _MATERIA_LEX.items():
#         if k in q:
#             feats["materia"] = v
#             break
#     for k, v in _FUERO_LEX.items():
#         if k in q:
#             feats["fuero"] = v
#             break
#     for k, v in _JURIS_LEX.items():
#         if k in q:
#             feats["jurisdiccion"] = v
#             break
#     # tipo_proceso (ejemplos comunes)
#     if "despido" in q:
#         feats["tipo_proceso"] = "despido"
#     elif "alimentos" in q:
#         feats["tipo_proceso"] = "alimentos"
#     elif "daños" in q or "danos" in q:
#         feats["tipo_proceso"] = "daños y perjuicios"
#     # dudas priorizadas (orden alto→medio)
#     dudas = [
#         "materia", "fuero", "jurisdiccion", "tribunal", "instancia", "sala",
#         "anio", "tipo_proceso", "tipo_fallo", "etapa", "numero_expediente", "partes"
#     ]
#     return feats, dudas

# def parse_answer_like(text: str) -> dict:
#     """Parser simple de respuestas tipo: "materia=laboral; anio=2023; jurisdiccion=tucuman"""
#     d = {}
#     for chunk in re.split(r"[,;\n]+", text or ""):
#         if "=" in chunk:
#             k, v = chunk.split("=", 1)
#             k, v = k.strip().lower(), v.strip()
#             if k:
#                 d[k] = v
#     return d

# def reanalizar_seccion1(features: dict) -> dict:
#     """Idempotente para compatibilidad; acá podés re-normalizar si querés."""
#     return features

# def maybe_refine_query(user_query: str, features: dict) -> str:
#     """Opcional: agregar hints textuales al query para el retriever."""
#     hints = []
#     for k in ["materia", "fuero", "jurisdiccion", "tribunal", "tipo_proceso", "anio", "numero_expediente"]:
#         v = features.get(k)
#         if v:
#             hints.append(f"{k}:{v}")
#     return user_query + (" " + " ".join(hints) if hints else "")


# =============================
# 3) ESTADO en Streamlit —>>> ADD donde definís session_state
# =============================
# if "clar_pending" not in st.session_state:
#     st.session_state.clar_pending = False
# if "clar_questions" not in st.session_state:
#     st.session_state.clar_questions = []
# if "clar_features" not in st.session_state:
#     st.session_state.clar_features = {}
# if "clar_dudas" not in st.session_state:
#     st.session_state.clar_dudas = []


# =============================
# 4) FLUJO principal —>>> MODIFY en el bloque que procesa `user_input`
# =============================
# Reemplazar el bloque actual del intent=="research" por este patrón:

# if intent == "research":
#     # (A) Si estamos en medio del bucle de aclaración, tratamos el input como respuestas
#     if st.session_state.clar_pending:
#         answers = parse_answer_like(user_input)
#         from clarification_loop import clarification_loop  # import local si no lo pusiste arriba
#         res = clarification_loop(
#             dudas_priorizadas=st.session_state.clar_dudas,
#             current_features=st.session_state.clar_features,
#             sufficiency_threshold=0.72,
#             provided_answers=answers,
#             reanalyze_fn=reanalizar_seccion1,
#         )
#         st.session_state.clar_features = res.features_updated
#         if res.sufficiency_reached:
#             st.session_state.clar_pending = False
#             refined = maybe_refine_query(user_input, res.features_updated)
#             run_research(refined)
#         else:
#             # seguimos preguntando (máximo 3–5 preguntas)
#             qs = res.questions or []
#             if not qs:
#                 from clarification_loop import generate_clarifying_questions
#                 qs = generate_clarifying_questions(st.session_state.clar_dudas, res.features_updated)
#             st.session_state.clar_questions = qs
#             st.chat_message("assistant").markdown("Necesito algunos datos para afinar la búsqueda. Respondé así: `clave=valor; clave=valor`.")
#             for q in qs:
#                 st.chat_message("assistant").markdown(q)
#             st.stop()
#     else:
#         # (B) Primera pasada: extraemos features y disparamos el bucle
#         feats, dudas = extract_features_basic(user_input)
#         from clarification_loop import clarification_loop
#         res = clarification_loop(dudas, feats, sufficiency_threshold=0.72, reanalyze_fn=reanalizar_seccion1)
#         st.session_state.clar_features = res.features_updated
#         st.session_state.clar_dudas = dudas
#         if res.sufficiency_reached:
#             refined = maybe_refine_query(user_input, res.features_updated)
#             run_research(refined)
#         else:
#             st.session_state.clar_pending = True
#             st.session_state.clar_questions = res.questions
#             st.chat_message("assistant").markdown("Para afinar la búsqueda, ¿podés aclarar esto?")
#             for q in res.questions:
#                 st.chat_message("assistant").markdown(q)
#             st.chat_message("assistant").markdown("Respondé en una línea con el formato: `materia=...; fuero=...; jurisdiccion=...; anio=...`.")
#             st.stop()

# Listo. Con esto, cuando la info es insuficiente, el asistente pregunta 3–5 cosas máximas
# y espera tus respuestas para continuar con la búsqueda en Postgres.
