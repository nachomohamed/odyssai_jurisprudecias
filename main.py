# main.py
import json
import uuid
import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple
import datetime as _dt


# LLM / Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Estructuras y validación
from pydantic import BaseModel, Field, constr
from typing import List as TList

# Postgres / pgvector
import psycopg2
from pgvector import Vector
from pgvector.psycopg2 import register_vector


# LangChain docs
from langchain_core.documents import Document

# Clarification loop
from aclaracion import clarification_loop

# =============================
# 1) IMPORTS —>>> ADD debajo de tus imports
# =============================
from clarification_loop import clarification_loop
import re
from typing import Any, Optional, Tuple# EXTRA imports que usa Sección 1



def _to_str(x):
    if x is None:
        return ""
    if isinstance(x, (_dt.date, _dt.datetime)):
        return x.isoformat()  # o x.strftime("%Y-%m-%d")
    return str(x)

def _fmt_date(x):
    if isinstance(x, (_dt.date, _dt.datetime)):
        return x.strftime("%Y-%m-%d")
    return str(x) if x is not None else "-"

def _fmt_score(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "—"


# =============================
# Configuración base
# =============================
st.set_page_config(page_title="⚖️ Jurisprudencia Assistant", layout="wide")
st.title("⚖️ Jurisprudencia Assistant")

OPENAI_KEY = "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"
PG_DSN="host=localhost port=5433 dbname=jurisprudencias user=postgres password=1a2b.3c4d"


# =============================
# Estado global
# =============================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []  # [{role, content}]
if "picked_docs" not in st.session_state:
    # docs elegidos por el LLM: [{"metadata": {...}, "page_content": "..."}]
    st.session_state.picked_docs: List[Dict] = []
if "last_mode" not in st.session_state:
    st.session_state.last_mode: str = "research"
# =============================
# 3) ESTADO en Streamlit —>>> ADD donde definís session_state
# =============================
if "clar_pending" not in st.session_state:
    st.session_state.clar_pending = False
if "clar_questions" not in st.session_state:
    st.session_state.clar_questions = []
if "clar_features" not in st.session_state:
    st.session_state.clar_features = {}
if "clar_dudas" not in st.session_state:
    st.session_state.clar_dudas = []


# =============================
# Modelos
# =============================
# Router barato para intención
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_KEY)
# Modelo principal
main_llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_KEY)


# =============================
# Retriever (Postgres + pgvector)
# =============================
class PgRetriever:
    """
    Retriever que consulta directamente Postgres (pgvector) en tus tablas:
      - juris.JURIS_CHUNKS (EMBEDDING vector)
      - juris.JURISPRUDENCES (metadatos)
    Ajustá nombres de columnas/tabla si difieren.
    """
    def __init__(self, dsn: str, embeddings: OpenAIEmbeddings, default_k: int = 10, default_fetch_k: int = 50):
        self.dsn = dsn
        self.embeddings = embeddings
        self.default_k = default_k
        self.default_fetch_k = default_fetch_k



    def get_relevant_documents(self, query: str, k: int = None, fetch_k: int = None) -> List[Document]:
        k = k or self.default_k          # ej: 10
        fetch_k = fetch_k or self.default_fetch_k  # ej: 50

        # 1) Embed de la consulta (mismo modelo que el indexado)
        qvec = self.embeddings.embed_query(query)  # -> list[float]


        sql = """
        WITH ranked AS (
        SELECT
            c.doc_id,
            c.section,
            c.content AS page_content,
            p.id, p.title, p.court, p.chamber, p.jurisdiction, p.fuero,
            p.matter, p.case_type, p.caratula, p.summary, p.ratio,
            p.decision_date, p.register_code, p.docket_number,
            p.decision_number, p.source_url,
            (c.embedding <-> %s) AS distance,
            ROW_NUMBER() OVER (PARTITION BY c.doc_id ORDER BY c.embedding <-> %s) AS rn
        FROM juris.juris_chunks c
        JOIN juris.jurisprudences p ON p.id = c.doc_id
        )
        SELECT
        doc_id, section, page_content,
        id, title, court, chamber, jurisdiction, fuero,
        matter, case_type, caratula, summary, ratio,
        decision_date, register_code, docket_number,
        decision_number, source_url,
        distance
        FROM ranked
        WHERE rn = 1
        ORDER BY distance
        LIMIT %s;
        """

        docs: List[Document] = []
        with psycopg2.connect(self.dsn) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(sql, (Vector(qvec), Vector(qvec), fetch_k))
                rows = cur.fetchall()
                cols = [d.name for d in cur.description]

        # normalización básica de distancia → score (1 = más similar)
        dist_vals = [dict(zip(cols, r))["distance"] for r in rows] or [0.0]
        dmin, dmax = min(dist_vals), max(dist_vals)
        def to_score(d):
            return 1.0 if dmax == dmin else (1.0 - (d - dmin) / (dmax - dmin))

        for r in rows[:k]:
            row = dict(zip(cols, r))
            meta = {
                "id": row.get("id"),
                "doc_id": row.get("doc_id"),
                "title": row.get("title"),
                "court": row.get("court"),
                "chamber": row.get("chamber"),
                "jurisdiction": row.get("jurisdiction"),
                "fuero": row.get("fuero"),
                "matter": row.get("matter"),
                "case_type": row.get("case_type"),
                "caratula": row.get("caratula"),
                "summary": row.get("summary"),
                "ratio": row.get("ratio"),
                "decision_date": _fmt_date(row.get("decision_date")),
                "register_code": row.get("register_code"),
                "docket_number": row.get("docket_number"),
                "decision_number": row.get("decision_number"),
                "source_url": row.get("source_url"),
                # extras
                "section": row.get("section"),
                "distance": row.get("distance"),
                "score": round(to_score(row.get("distance") or 0.0), 3),
            }
            docs.append(Document(page_content=row.get("page_content") or "", metadata=meta))
        return docs


    # compat mínima con vs.as_retriever(...)
    def as_retriever(self, **kwargs):
        return self


@st.cache_resource
def load_retriever_pg():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # MISMO modelo usado al indexar
        openai_api_key=OPENAI_KEY
    )
    return PgRetriever(PG_DSN, embeddings)

retriever = load_retriever_pg()


# =============================
# Helpers UI
# =============================
DISPLAY_ORDER = [
    "caratula",
    "tribunal_principal", "tribunal_sala",
    "tipo_causa",
    "nro_expediente", "nro_sentencia", "registro",
    "fecha_sentencia",
    "sumario",
    "texto",
]

def render_kv_table(meta: dict):
    """Tabla Columna | Contenido + expanders para sumario/texto."""
    meta = meta or {}
    rows, seen = [], set()

    for k in DISPLAY_ORDER:
        if k in meta and meta[k] not in (None, "", "nan"):
            if k in ["sumario", "texto"]:
                continue
            rows.append({"Columna": k, "Contenido": str(meta[k])})
            seen.add(k)

    for k, v in meta.items():
        if k in seen or k in ["sumario", "texto"]:
            continue
        if v not in (None, "", "nan"):
            rows.append({"Columna": k, "Contenido": str(v)})

    if rows:
        st.table(pd.DataFrame(rows))

    if meta.get("sumario"):
        with st.expander("sumario", expanded=True):
            st.write(str(meta["sumario"]))
    if meta.get("texto"):
        with st.expander("texto", expanded=False):
            st.write(str(meta["texto"]))

def show_active_docs():
    # No-op: decidimos no mostrar panel lateral ni expanders.
    return
    
    # """Panel con las jurisprudencias activas para conversar sobre ellas."""
    # docs = st.session_state.picked_docs
    # if not docs:
    #     return
    # with st.expander(f"📚 Jurisprudencias activas ({len(docs)})", expanded=False):
    #     for i, d in enumerate(docs, start=1):
    #         meta = d["metadata"] or {}
    #         titulo = meta.get("caratula") or meta.get("titulo") or f"Jurisprudencia {i}"
    #         trib = meta.get("tribunal_principal") or meta.get("tribunal") or ""
    #         fecha = meta.get("fecha_sentencia") or meta.get("fecha") or ""
    #         header = f"**{titulo}**" + (f" — {trib}" if trib else "") + (f" — {fecha}" if fecha else "")
    #         st.markdown(f"{i}. {header}")
    #         with st.expander(f"Ver detalles: {titulo}", expanded=False):
    #             render_kv_table(meta)


# =============================
# Router de intención (LLM)
# =============================
def classify_intent(user_message: str) -> str:
    hist_pairs = [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-12:]]
    history_text = "\n".join(hist_pairs)

    system = (
        "You are an intention classifier for a legal assistant. "
        "Decide if the user wants to SEARCH/RETRIEVE new jurisprudences (label: research), "
        "or CHAT/DISCUSS about already retrieved jurisprudences (label: chat). "
        "Return EXACTLY one token: research OR chat."
    )
    user = (
        f"Conversation so far:\n{history_text}\n\n"
        f"New user message:\n{user_message}\n\n"
        "Return: research OR chat"
    )
    out = router_llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])
    label = (out.content if hasattr(out, "content") else str(out)).strip().lower()
    if label not in ("research", "chat"):
        label = "research"
    if label == "chat" and not st.session_state.picked_docs:
        label = "research"
    return label


# =============================
# Esquema estructurado para Top-3
# =============================
class PickItem(BaseModel):
    uid: constr(strip_whitespace=True, min_length=1)
    bullets: TList[constr(strip_whitespace=True, min_length=1)] = Field(
        ..., description="Viñetas concretas (hechos, normas/art, jurisdicción/fecha, resultado)"
    )
    resumen: constr(strip_whitespace=True, min_length=1)

class Top3Response(BaseModel):
    intro: constr(strip_whitespace=True, min_length=1)
    items: TList[PickItem] = Field(..., min_items=3, max_items=3)


# =============================
# Research helpers
# =============================
def choose_uid(doc_meta: dict, extract: str) -> str:
    titulo = (doc_meta.get("caratula") or doc_meta.get("titulo") or "Jurisprudencia").strip()
    trib = (doc_meta.get("tribunal_principal") or doc_meta.get("tribunal") or "").strip()
    fecha = (doc_meta.get("fecha_sentencia") or doc_meta.get("fecha") or "").strip()
    tipo = (doc_meta.get("tipo_causa") or "").strip()
    descriptor = " — ".join([s for s in map(_to_str, [titulo, trib, fecha, tipo]) if s])
    key = (descriptor + " | " + extract[:200]).strip()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

def _distinct_strings(lst: List[str]) -> bool:
    norm = [(" ".join((s or "").lower().split())) for s in lst]
    return len(set(norm)) == len(norm)

def llm_pick_top3_and_explain(user_query: str, candidates: List[Dict], retry_hint: str = "") -> Tuple[str, List[Dict]]:
    """
    Usa un esquema Pydantic para obligar a devolver JSON válido con EXACTAMENTE 3 ítems.
    Con fallback a JSON mode si el proveedor no respeta del todo el esquema.
    """
    # 1) Modelo estructurado (JSON validado contra Top3Response)
    structured_llm = main_llm.with_structured_output(Top3Response)

    system = (
        "Eres un asistente jurídico. Recibirás hasta 10 fallos candidatos. "
        "Debes elegir EXACTAMENTE 3. Sé específico y no repitas explicaciones entre casos. "
        "Para cada fallo: viñetas claras sobre hechos clave, normas/artículos (p. ej. art. 242 LCT si figura), "
        "jurisdicción/instancia/fecha y resultado/criterio. Incluye al menos una cita corta (≤12 palabras) del extracto. "
        "No inventes."
    )
    user = (
        f"Consulta del abogado:\n{user_query}\n\n"
        "Fallos candidatos (uid, descriptor, extracto parcial):\n"
        f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
        "TAREA:\n"
        "1) Selecciona los 3 fallos más relevantes (no más ni menos).\n"
        "2) Devuelve un objeto con 'intro' y 'items' (exactamente 3), "
        "   donde cada item tiene 'uid', 'bullets' (lista) y 'resumen'.\n"
        f"{retry_hint}"
    )

    try:
        parsed: Top3Response = structured_llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ])
        intro = parsed.intro.strip()
        result = [{"uid": it.uid, "bullets": it.bullets, "resumen": it.resumen} for it in parsed.items]
        return intro, result

    except Exception:
        # Fallback: “JSON mode” + validación manual
        json_mode_llm = ChatOpenAI(
            model=main_llm.model,
            temperature=main_llm.temperature,
            openai_api_key=OPENAI_KEY,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        out = json_mode_llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ])
        text = out.content if hasattr(out, "content") else str(out)

        try:
            data = json.loads(text)
            intro = (data.get("intro") or "").strip()
            items = (data.get("items") or [])[:3]
            result = []
            for it in items:
                uid = (it.get("uid") or "").strip()
                bullets = it.get("bullets") or []
                resumen = (it.get("resumen") or "").strip()
                if uid and bullets and resumen:
                    result.append({"uid": uid, "bullets": bullets, "resumen": resumen})
            if len(result) != 3 or not intro:
                raise ValueError("Esquema incompleto en fallback")
            return intro, result
        except Exception:
            return "", []

# =============================
# 2) SECCIÓN 1: extracción básica —>>> ADD (por ejemplo debajo de "Research helpers")
# =============================
# Mapeos ligeros para detectar valores por palabras clave
_MATERIA_LEX = {
    "laboral": "laboral",
    "civil": "civil",
    "penal": "penal",
    "comercial": "comercial",
    "familia": "familia",
}
_FUERO_LEX = {
    "laboral": "laboral",
    "civil": "civil",
    "penal": "penal",
    "contencioso": "contencioso",
}
_JURIS_LEX = {
    "tucuman": "tucumán",
    "caba": "caba",
    "buenos aires": "pba",
}

def extract_features_basic(user_query: str) -> Tuple[dict, list]:
    """Heurística mínima para armar `features_actuales` y `dudas_priorizadas`.
    - Busca año (1980–2030), número de expediente (patrones simples), y claves por palabras.
    - Devuelve features con defaults None y una lista ordenada de dudas prioritarias.
    """
    q = (user_query or "").lower()
    feats = {
        "materia": None,
        "fuero": None,
        "jurisdiccion": None,
        "tribunal": None,
        "instancia": None,
        "sala": None,
        "anio": None,
        "tipo_proceso": None,
        "tipo_fallo": None,
        "etapa": None,
        "numero_expediente": None,
        "partes": [],
    }
    # año
    m = re.search(r"\b(19\d{2}|20[0-3]\d)\b", q)
    if m:
        try:
            feats["anio"] = int(m.group(1))
        except:
            pass
    # nro expediente (muy laxo)
    m = re.search(r"expediente\s*[:#-]?\s*([\w./-]{5,})", q)
    if m:
        feats["numero_expediente"] = m.group(1).strip()
    # materia / fuero / jurisdicción
    for k, v in _MATERIA_LEX.items():
        if k in q:
            feats["materia"] = v
            break
    for k, v in _FUERO_LEX.items():
        if k in q:
            feats["fuero"] = v
            break
    for k, v in _JURIS_LEX.items():
        if k in q:
            feats["jurisdiccion"] = v
            break
    # tipo_proceso (ejemplos comunes)
    if "despido" in q:
        feats["tipo_proceso"] = "despido"
    elif "alimentos" in q:
        feats["tipo_proceso"] = "alimentos"
    elif "daños" in q or "danos" in q:
        feats["tipo_proceso"] = "daños y perjuicios"
    # dudas priorizadas (orden alto→medio)
    dudas = [
        "materia", "fuero", "jurisdiccion", "tribunal", "instancia", "sala",
        "anio", "tipo_proceso", "tipo_fallo", "etapa", "numero_expediente", "partes"
    ]
    return feats, dudas

def parse_answer_like(text: str) -> dict:
    """Parser simple de respuestas tipo: "materia=laboral; anio=2023; jurisdiccion=tucuman"""
    d = {}
    for chunk in re.split(r"[,;\n]+", text or ""):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            k, v = k.strip().lower(), v.strip()
            if k:
                d[k] = v
    return d

def reanalizar_seccion1(features: dict) -> dict:
    """Idempotente para compatibilidad; acá podés re-normalizar si querés."""
    return features

def maybe_refine_query(user_query: str, features: dict) -> str:
    """Opcional: agregar hints textuales al query para el retriever."""
    hints = []
    for k in ["materia", "fuero", "jurisdiccion", "tribunal", "tipo_proceso", "anio", "numero_expediente"]:
        v = features.get(k)
        if v:
            hints.append(f"{k}:{v}")
    return user_query + (" " + " ".join(hints) if hints else "")


# =============================
# Research con diagnósticos y fallbacks
# =============================
def run_research(user_query: str):
    # 0) Recuperar candidatos
    try:
        candidate_docs = retriever.get_relevant_documents(user_query)
    except Exception as e:
        st.error(f"❌ Error al recuperar documentos desde Postgres: {e}")
        return

    if not candidate_docs:
        st.warning("⚠️ No encontré jurisprudencias relevantes.")
        return

    # === PRELISTADO (previo al filtrado por LLM): Top 10 por similitud ===
    pre_blocks = []
    seen = set()  # por si el retriever aún puede devolver duplicados por doc_id
    pre_list = []
    for d in candidate_docs:
        m = d.metadata or {}
        did = m.get("doc_id")
        if did in seen:
            continue
        seen.add(did)
        pre_list.append(d)
        if len(pre_list) == 10:
            break

    for i, d in enumerate(pre_list, start=1):
        m = d.metadata or {}
        titulo = m.get("caratula") or m.get("title") or f"Jurisprudencia {i}"
        tribunal = m.get("court") or m.get("tribunal_principal") or "-"
        juris = m.get("jurisdiction") or "-"
        fecha = _fmt_date(m.get("decision_date") or m.get("fecha_sentencia"))
        score = _fmt_score(m.get("score"))
        pre_blocks.append(
            f"**{i}. {titulo}**\n"
            f"- Tribunal: {tribunal} | Jurisdicción: {juris} | Fecha: {fecha}\n"
            f"- **Puntaje**: {score}"
        )

    # Mostramos el prelistado en el chat
    pre_text = "**Jurisprudencias buscadas previo al filtrado (Top 10 por similitud)**\n\n" + "\n\n".join(pre_blocks)
    st.session_state.messages.append({"role": "assistant", "content": pre_text})
    st.chat_message("assistant").markdown(pre_text)

    # 1) Armar candidatos para el LLM + mapa uid→doc (usamos primeros 10)
    candidates = []
    uid_to_doc = {}
    for d in pre_list:
        m = d.metadata or {}
        titulo = m.get("caratula") or m.get("title") or "Jurisprudencia"
        tribunal = m.get("court") or m.get("tribunal_principal") or ""
        fecha = m.get("decision_date") or m.get("fecha_sentencia") or ""
        tipo = m.get("case_type") or m.get("tipo_causa") or ""
        descriptor = " — ".join([s for s in map(_to_str, [titulo, tribunal, fecha, tipo]) if s])
        extracto = (getattr(d, "page_content", "") or "")[:1600]
        uid = choose_uid(m, extracto)
        candidates.append({"uid": uid, "descriptor": descriptor, "extracto": extracto})
        uid_to_doc[uid] = d

    # 2) LLM elige 3 (estructurado + fallback)
    intro, picked = llm_pick_top3_and_explain(user_query, candidates)
    if not picked or not _distinct_strings([p.get("resumen","") for p in picked]):
        intro2, picked2 = llm_pick_top3_and_explain(
            user_query, candidates,
            retry_hint="Asegura diferencias concretas entre casos (hechos, normas, resultado, jurisdicción/fecha)."
        )
        if picked2:
            intro = intro2 or intro
            picked = picked2

    # 3) Fallback duro si quedó vacío
    if not picked:
        picked = [
            {"uid": c["uid"], "bullets": ["• Coincidencia por texto y normas."], "resumen": "Pertinente por similitud fáctica."}
            for c in candidates[:3]
        ]
        intro = intro or "Seleccioné los fallos más pertinentes disponibles."

    # 4) Guardar “activos” para chat posterior (sin expanders)
    st.session_state.picked_docs = [
        {"metadata": uid_to_doc[u["uid"]].metadata, "page_content": uid_to_doc[u["uid"]].page_content}
        for u in picked if u["uid"] in uid_to_doc
    ]

    # 5) Mostrar SELECCIÓN FINAL (Top 3) en el chat (mínimo + puntaje)
    bloques = []
    for i, item in enumerate(picked, start=1):
        d = uid_to_doc.get(item["uid"])
        if not d:
            continue
        m = d.metadata or {}
        titulo = m.get("caratula") or m.get("title") or f"Jurisprudencia {i}"
        tribunal = m.get("court") or m.get("tribunal_principal") or "-"
        juris = m.get("jurisdiction") or "-"
        fecha = _fmt_date(m.get("decision_date") or m.get("fecha_sentencia"))
        score = _fmt_score(m.get("score"))
        bullets = "\n".join(item.get("bullets", []))
        resumen = item.get("resumen", "")

        bloque = (
            f"**{i}. {titulo}**\n"
            f"- Tribunal: {tribunal} | Jurisdicción: {juris} | Fecha: {fecha}\n"
            f"- **Puntaje**: {score}\n"
            f"{bullets}\n"
            + (f"_Conclusión:_ {resumen}" if resumen else "")
        )
        bloques.append(bloque)

    texto = ("**" + (intro or "Resultados más relevantes") + "**\n\n") + "\n\n".join(bloques)
    st.session_state.messages.append({"role": "assistant", "content": texto})
    st.chat_message("assistant").markdown(texto)

# ====== Segundo pase de justificación breve y neutral por fallo ======
def llm_extra_why(user_query: str, descriptor: str, extracto: str, ya_dicho: str = "") -> str:
    system = (
        "Eres un asistente jurídico. Redacta una justificación breve (2–3 frases), neutral y profesional, "
        "respondiendo por qué este fallo es adecuado para el caso planteado. "
        "Evita repetir literalmente argumentos ya dichos. No inventes."
    )
    user = (
        f"Caso del abogado:\n{user_query}\n\n"
        f"Fallo:\n{descriptor}\n\n"
        f"Extracto (contexto real):\n{extracto[:1200]}\n\n"
        f"Lo ya dicho (para no repetir):\n{ya_dicho}\n\n"
        "Devuelve solo el párrafo (2–3 frases)."
    )
    out = main_llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])
    return (out.content if hasattr(out, "content") else str(out)).strip()


# =============================
# Chat sobre jurisprudencias activas
# =============================
def run_chat(user_message: str):
    docs = st.session_state.picked_docs
    if not docs:
        return run_research(user_message)

    # contexto mínimo para el LLM
    ctx_blocks = []
    for idx, d in enumerate(docs, start=1):
        m = d["metadata"] or {}
        titulo = m.get("caratula") or m.get("title") or f"Jurisprudencia {idx}"
        tribunal = m.get("court") or m.get("tribunal_principal") or "-"
        juris = m.get("jurisdiction") or "-"
        fecha = m.get("decision_date") or m.get("fecha_sentencia") or "-"
        header = f"[{idx}] {titulo} — {tribunal} — {juris} — {fecha}"
        extracto = (d.get("page_content") or "")[:2000]
        ctx_blocks.append(f"{header}\n{extracto}")

    context = "\n\n---\n\n".join(ctx_blocks)

    system = (
        "Eres un asistente jurídico en MODO CONVERSACIÓN. "
        "Responde solo con la jurisprudencia listada a continuación. "
        "Si algo no figura, indícalo claramente.\n\n"
        f"Jurisprudencia disponible:\n{context}"
    )
    out = main_llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user_message}
    ])
    answer = out.content if hasattr(out, "content") else str(out)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)

# =============================
# Mostrar historial previo
# =============================
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])


# =============================
# Entrada de usuario
# =============================
user_input = st.chat_input("Escribí tu mensaje (p. ej., 'buscá...', 'compará...', 'resumí...', 'redactá...')")
if user_input:
    # mostrar el mensaje del usuario inmediatamente
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # clasificar intención por LLM
    intent = classify_intent(user_input)
    st.session_state.last_mode = intent

    # ejecutar modo correspondiente
    if intent == "research":
        # (A) Si estamos en medio del bucle de aclaración, tratamos el input como respuestas
        if st.session_state.clar_pending:
            answers = parse_answer_like(user_input)
            from clarification_loop import clarification_loop  # import local si no lo pusiste arriba
            res = clarification_loop(
                dudas_priorizadas=st.session_state.clar_dudas,
                current_features=st.session_state.clar_features,
                sufficiency_threshold=0.72,
                provided_answers=answers,
                reanalyze_fn=reanalizar_seccion1,
            )
            st.session_state.clar_features = res.features_updated
            if res.sufficiency_reached:
                st.session_state.clar_pending = False
                refined = maybe_refine_query(user_input, res.features_updated)
                run_research(refined)
            else:
                # seguimos preguntando (máximo 3–5 preguntas)
                qs = res.questions or []
                if not qs:
                    from clarification_loop import generate_clarifying_questions
                    qs = generate_clarifying_questions(st.session_state.clar_dudas, res.features_updated)
                st.session_state.clar_questions = qs
                st.chat_message("assistant").markdown("Necesito algunos datos para afinar la búsqueda. Respondé así: `clave=valor; clave=valor`.")
                for q in qs:
                    st.chat_message("assistant").markdown(q)
                st.stop()
        else:
            # (B) Primera pasada: extraemos features y disparamos el bucle
            feats, dudas = extract_features_basic(user_input)
            from clarification_loop import clarification_loop
            res = clarification_loop(dudas, feats, sufficiency_threshold=0.72, reanalyze_fn=reanalizar_seccion1)
            st.session_state.clar_features = res.features_updated
            st.session_state.clar_dudas = dudas
            if res.sufficiency_reached:
                refined = maybe_refine_query(user_input, res.features_updated)
                run_research(refined)
            else:
                st.session_state.clar_pending = True
                st.session_state.clar_questions = res.questions
                st.chat_message("assistant").markdown("Para afinar la búsqueda, ¿podés aclarar esto?")
                for q in res.questions:
                    st.chat_message("assistant").markdown(q)
                st.chat_message("assistant").markdown("Respondé en una línea con el formato: `materia=...; fuero=...; jurisdiccion=...; anio=...`.")
                st.stop()


# =============================
# Siempre visible: jurisprudencias activas (si hay)
# =============================
#show_active_docs()


# =============================
# Nota:
# - Este código consulta Postgres (pgvector) directamente como fuente única.
# - Asegurate de haber creado la extensión vector: CREATE EXTENSION IF NOT EXISTS vector;
# - El modelo de embeddings debe coincidir con el usado al indexar.
# =============================
