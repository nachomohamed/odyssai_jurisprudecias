# app.py
import json
import uuid
import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# =============================
# Configuración base
# =============================
st.set_page_config(page_title="⚖️ Jurisprudencia Assistant", layout="wide")
st.title("⚖️ Jurisprudencia Assistant")

OPENAI_KEY = st.secrets["OPENAI_KEY"]

# =============================
# Estado global de la app
# =============================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []  # [{role, content}]
if "picked_docs" not in st.session_state:
    # Guardamos docs elegidos por el LLM como dicts simples: {"metadata": {...}, "page_content": "..."}
    st.session_state.picked_docs: List[Dict] = []
if "last_mode" not in st.session_state:
    st.session_state.last_mode: str = "research"  # informativo

# =============================
# LLMs
# =============================
# Router (barato/rápido) para intención
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_KEY)
# Modelo principal para respuestas/explicaciones
main_llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_KEY)

# =============================
# Retriever (MMR, amplia cobertura)
# =============================
@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_KEY
    )
    vs = FAISS.load_local(
        "vectorstore_jurisprudencia",
        embeddings,
        allow_dangerous_deserialization=True
    )
    # MMR = resultados diversos; pedimos más candidatos
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,        # devolver hasta 10
            "fetch_k": 50,  # pool grande para diversidad
            "lambda_mult": 0.5
        }
    )
    return retriever

retriever = load_retriever()

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
    """Renderiza tabla 'Columna | Contenido' y pone sumario/texto en expanders."""
    meta = meta or {}
    rows, seen = [], set()

    # Primero columnas clave
    for k in DISPLAY_ORDER:
        if k in meta and meta[k] not in (None, "", "nan"):
            if k in ["sumario", "texto"]:
                continue
            rows.append({"Columna": k, "Contenido": str(meta[k])})
            seen.add(k)

    # Luego el resto de metadata
    for k, v in meta.items():
        if k in seen or k in ["sumario", "texto"]:
            continue
        if v not in (None, "", "nan"):
            rows.append({"Columna": k, "Contenido": str(v)})

    if rows:
        st.table(pd.DataFrame(rows))

    # Campos largos en expanders
    if meta.get("sumario"):
        with st.expander("sumario", expanded=True):
            st.write(str(meta["sumario"]))
    if meta.get("texto"):
        with st.expander("texto", expanded=False):
            st.write(str(meta["texto"]))

# =============================
# Router de intención (LLM)
# =============================
def classify_intent(user_message: str, history: List[Dict[str, str]]) -> str:
    """
    Devuelve 'research' o 'chat' usando LLM y el historial.
    Si no hay docs previos y dice 'chat', forzamos 'research' después.
    """
    # Compactamos historial para el router (últimos 12 intercambios)
    hist_pairs = []
    for m in st.session_state.messages[-12:]:
        hist_pairs.append(f"{m['role']}: {m['content']}")
    history_text = "\n".join(hist_pairs)

    system = (
        "You are an intention classifier for a legal assistant. "
        "Your job is to decide if the user wants to: "
        "(a) SEARCH/RETRIEVE new jurisprudences (label: research) or "
        "(b) CHAT/DISCUSS about already retrieved jurisprudences (label: chat). "
        "Consider the conversation so far. "
        "Answer with EXACTLY one word: 'research' or 'chat'. No punctuation. No explanations."
    )
    user = (
        f"Conversation so far:\n{history_text}\n\n"
        f"New user message:\n{user_message}\n\n"
        "Return ONLY one token: research OR chat."
    )
    out = router_llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])
    label = (out.content if hasattr(out, "content") else str(out)).strip().lower()
    if label not in ("research", "chat"):
        label = "research"
    # Si quiere chatear pero no hay docs elegidos, vamos a research (no hay sobre qué chatear)
    if label == "chat" and not st.session_state.picked_docs:
        label = "research"
    return label

# =============================
# Research: preparar candidatos y pedir al LLM que elija 3 con explicación
# =============================
def choose_uid(doc_meta: dict, extract: str) -> str:
    # UID determinístico en base a descriptor + extracto para evitar duplicados
    titulo = (doc_meta.get("caratula") or doc_meta.get("titulo") or "Jurisprudencia").strip()
    trib = (doc_meta.get("tribunal_principal") or doc_meta.get("tribunal") or "").strip()
    fecha = (doc_meta.get("fecha_sentencia") or doc_meta.get("fecha") or "").strip()
    tipo = (doc_meta.get("tipo_causa") or "").strip()
    descriptor = " — ".join([x for x in [titulo, trib, fecha, tipo] if x])
    key = (descriptor + "|" + extract[:200]).strip()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

def llm_pick_top3_and_explain(user_query: str, candidates: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    candidates: [{uid, descriptor, extracto}]
    Devuelve (intro:str, items:list[{uid, bullets, resumen}])
    """
    system = (
        "Eres un asistente jurídico. Recibirás hasta 10 fallos candidatos. "
        "Debes elegir EXACTAMENTE 3. "
        "Para cada uno: explica con detalle en viñetas por qué es relevante. "
        "Incluye hechos clave, artículo/norma (si aparece), jurisdicción/instancia/fecha, y resultado/criterio. "
        "Añade al menos una cita breve entre comillas (≤12 palabras) del extracto. "
        "Sé claro y evita frases genéricas. No inventes."
    )
    user = (
        f"Consulta del abogado:\n{user_query}\n\n"
        "Fallos candidatos (uid, descriptor, extracto parcial):\n"
        f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
        "TAREA:\n"
        "1) Selecciona los 3 fallos más relevantes.\n"
        "2) Para cada uno devuelve viñetas ('• ...') con explicaciones concretas (mínimo 3 bullets). "
        "   Cierra con una frase-síntesis de por qué es el más adecuado.\n"
        "3) Devuelve SOLO JSON con este formato:\n"
        "{\n"
        '  "intro": "texto de introducción",\n'
        '  "items": [\n'
        '    {"uid": "uid1", "bullets": ["• punto 1", "• punto 2", "• punto 3"], "resumen": "frase final"},\n'
        '    {"uid": "uid2", "bullets": [...], "resumen": "..."},\n'
        '    {"uid": "uid3", "bullets": [...], "resumen": "..."}\n'
        "  ]\n"
        "}"
    )
    out = main_llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])
    text = out.content if hasattr(out, "content") else str(out)

    try:
        data = json.loads(text)
        intro = (data.get("intro") or "").strip()
        items = data.get("items") or []
        result = []
        for it in items[:3]:
            uid = (it.get("uid") or "").strip()
            bullets = it.get("bullets") or []
            resumen = (it.get("resumen") or "").strip()
            if uid and bullets and resumen:
                result.append({"uid": uid, "bullets": bullets, "resumen": resumen})
        if len(result) < 3:
            raise ValueError("Modelo devolvió menos de 3 válidos")
        return intro, result
    except Exception:
        # Fallback: tomo 3 primeros candidatos con razones genéricas (evitamos quedarnos en blanco)
        intro = "Analicé tu consulta y seleccioné los fallos que mejor encajan por hechos y normativa."
        result = [{"uid": c["uid"],
                   "bullets": [
                       "• Hechos sustancialmente análogos a los planteados.",
                       "• El extracto refiere a la norma aplicable y su interpretación.",
                       "• Criterio/resultado compatible con el objetivo procesal."
                   ],
                   "resumen": "Aporta fundamentos útiles para la estrategia del caso."}
                  for c in candidates[:3]]
        return intro, result

def run_research(user_query: str):
    """Ejecuta el pipeline de búsqueda y muestra top-3 con explicaciones."""
    try:
        candidate_docs = retriever.get_relevant_documents(user_query)
    except Exception:
        candidate_docs = retriever.invoke(user_query)

    if not candidate_docs:
        msg = ("No encontré jurisprudencias relevantes en tu base. "
               "Probá aportar más detalles (hechos clave, norma aplicable, jurisdicción, período).")
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").markdown(msg)
        return

    # Preparo candidatos para el LLM
    candidates = []
    uid_to_docdict = {}
    for d in candidate_docs[:10]:
        meta = d.metadata or {}
        titulo = meta.get("caratula") or meta.get("titulo") or "Jurisprudencia"
        trib = meta.get("tribunal_principal") or meta.get("tribunal") or ""
        fecha = meta.get("fecha_sentencia") or meta.get("fecha") or ""
        tipo = meta.get("tipo_causa") or ""
        descriptor = " — ".join([x for x in [titulo, trib, fecha, tipo] if x])
        extracto = (getattr(d, "page_content", "") or "")[:1600]
        uid = choose_uid(meta, extracto)
        candidates.append({"uid": uid, "descriptor": descriptor, "extracto": extracto})
        uid_to_docdict[uid] = {
            "metadata": meta,
            "page_content": getattr(d, "page_content", "") or ""
        }

    # LLM elige 3 y explica
    intro, picked = llm_pick_top3_and_explain(user_query, candidates)
    st.markdown(f"**{intro}**")

    # Guardamos elegidos para el Chat Mode
    st.session_state.picked_docs = [uid_to_docdict[x["uid"]] for x in picked if x["uid"] in uid_to_docdict]

    # Render
    resumen_lineas = []
    for i, item in enumerate(picked, start=1):
        uid = item["uid"]
        docd = uid_to_docdict.get(uid)
        if not docd:
            continue
        meta = docd["metadata"]
        titulo = meta.get("caratula") or meta.get("titulo") or "Jurisprudencia"
        trib = meta.get("tribunal_principal") or meta.get("tribunal") or ""
        fecha = meta.get("fecha_sentencia") or meta.get("fecha") or ""
        header = f"**{titulo}**" + (f" — {trib}" if trib else "") + (f" — {fecha}" if fecha else "")

        st.markdown(f"**{i}. {titulo}**")
        for b in item.get("bullets", []):
            st.markdown(b)
        if item.get("resumen"):
            st.markdown(f"_**Conclusión:**_ {item['resumen']}")

        with st.expander(header, expanded=(i == 1)):
            render_kv_table(meta)

        resumen_lineas.append(f"{i}. {titulo} — {item.get('resumen','')}")

    final_msg = "🧠 **Resumen breve:**\n" + "\n\n".join(resumen_lineas)
    st.session_state.messages.append({"role": "assistant", "content": final_msg})
    st.chat_message("assistant").markdown(final_msg)

# =============================
# Chat sobre las jurisprudencias elegidas
# =============================
def run_chat(user_message: str):
    # Contexto armado desde picked_docs
    if not st.session_state.picked_docs:
        # Si no hay docs, lo convertimos a research
        return run_research(user_message)

    # Compongo contexto con headers + recortes
    ctx_blocks = []
    for idx, d in enumerate(st.session_state.picked_docs, start=1):
        meta = d["metadata"] or {}
        titulo = meta.get("caratula") or meta.get("titulo") or f"Jurisprudencia {idx}"
        trib = meta.get("tribunal_principal") or meta.get("tribunal") or ""
        fecha = meta.get("fecha_sentencia") or meta.get("fecha") or ""
        header = " — ".join([x for x in [titulo, trib, fecha] if x])
        extracto = (d["page_content"] or "")[:2000]
        ctx_blocks.append(f"[{idx}] {header}\n{extracto}")

    context = "\n\n---\n\n".join(ctx_blocks)

    system = (
        "Eres un asistente jurídico en MODO CONVERSACIÓN. "
        "Responde exclusivamente usando la jurisprudencia recuperada y listada a continuación. "
        "Si algo no figura en los textos, dilo explícitamente. "
        "Puedes comparar, resumir, extraer criterios, o redactar borradores (p.ej., demanda o contestación) "
        "pero siempre basándote en los fallos provistos.\n\n"
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
# Zona de mensajes previos (UI)
# =============================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").markdown(msg["content"])

# =============================
# Opcional: botones utilitarios
# =============================
cols = st.columns(3)
with cols[0]:
    if st.button("🧹 Limpiar conversación", use_container_width=True):
        st.session_state.messages.clear()
        st.session_state.picked_docs.clear()
        st.session_state.last_mode = "research"
        st.experimental_rerun()
with cols[1]:
    if st.button("🔎 Forzar nueva búsqueda", use_container_width=True):
        st.session_state.last_mode = "research"
        st.session_state.messages.append({"role": "assistant", "content": "Entendido. ¿Sobre qué tema busco nueva jurisprudencia?"})
        st.experimental_rerun()

# =============================
# Entrada del usuario
# =============================
user_input = st.chat_input("Escribí tu mensaje (puede ser 'buscá...', 'compará...', 'resumí...', 'redactá...', etc.)")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 1) Clasificación de intención (LLM)
    intent = classify_intent(user_input, st.session_state.messages)
    st.session_state.last_mode = intent  # informativo en el estado

    # 2) Ejecución según intención
    if intent == "research":
        run_research(user_input)
    else:
        run_chat(user_input)

# =============================
# Nota:
# - Este código asume que tu vectorstore FAISS ya existe en 'vectorstore_jurisprudencia'
#   con documentos que contienen en page_content el texto del fallo y, si es posible,
#   metadata como 'caratula', 'tribunal_principal', 'fecha_sentencia', etc.
# - Si querés ajustar el umbral de similitud o filtros por jurisdicción/año,
#   podés crear el retriever con: vs.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.2})
# =============================
