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
# Modelos
# =============================
# Router barato para intención
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_KEY)
# Modelo principal
main_llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_KEY)

# =============================
# Retriever (MMR)
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
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 50, "lambda_mult": 0.5}
    )

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
    """Panel con las jurisprudencias activas para poder conversar siempre sobre ellas."""
    docs = st.session_state.picked_docs
    if not docs:
        return
    with st.expander(f"📚 Jurisprudencias activas ({len(docs)})", expanded=False):
        for i, d in enumerate(docs, start=1):
            meta = d["metadata"] or {}
            titulo = meta.get("caratula") or meta.get("titulo") or f"Jurisprudencia {i}"
            trib = meta.get("tribunal_principal") or meta.get("tribunal") or ""
            fecha = meta.get("fecha_sentencia") or meta.get("fecha") or ""
            header = f"**{titulo}**" + (f" — {trib}" if trib else "") + (f" — {fecha}" if fecha else "")
            st.markdown(f"{i}. {header}")
            with st.expander(f"Ver detalles: {titulo}", expanded=False):
                render_kv_table(meta)

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
# Research helpers
# =============================
def choose_uid(doc_meta: dict, extract: str) -> str:
    titulo = (doc_meta.get("caratula") or doc_meta.get("titulo") or "Jurisprudencia").strip()
    trib = (doc_meta.get("tribunal_principal") or doc_meta.get("tribunal") or "").strip()
    fecha = (doc_meta.get("fecha_sentencia") or doc_meta.get("fecha") or "").strip()
    tipo = (doc_meta.get("tipo_causa") or "").strip()
    descriptor = " ".join([x for x in [titulo, trib, fecha, tipo] if x])
    key = (descriptor + " | " + extract[:200]).strip()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

def _distinct_strings(lst: List[str]) -> bool:
    norm = [(" ".join((s or "").lower().split())) for s in lst]
    return len(set(norm)) == len(norm)

def llm_pick_top3_and_explain(user_query: str, candidates: List[Dict], retry_hint: str = "") -> Tuple[str, List[Dict]]:
    """
    candidates: [{uid, descriptor, extracto}]
    Devuelve (intro:str, items:list[{uid, bullets, resumen}])
    """
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
        "2) Devuelve SOLO JSON exacto:\n"
        "{\n"
        '  "intro": "texto",\n'
        '  "items": [\n'
        '    {"uid": "uid1", "bullets": ["• punto 1", "• punto 2", "• punto 3"], "resumen": "frase final"},\n'
        '    {"uid": "uid2", "bullets": ["• ..."], "resumen": "..."},\n'
        '    {"uid": "uid3", "bullets": ["• ..."], "resumen": "..."}\n'
        "  ]\n"
        "}\n"
        "Evita bullets genéricos tipo 'coincidencias fácticas y normativas'. "
        "Cita nombres de partes, artículos o fragmentos reales del extracto cuando ayuden.\n"
        f"{retry_hint}"
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
            raise ValueError("Menos de 3 ítems válidos")
        return intro, result
    except Exception:
        return "", []

# ====== NUEVO: segundo pase de justificación breve y neutral por fallo ======
def llm_extra_why(user_query: str, descriptor: str, extracto: str, ya_dicho: str = "") -> str:
    """
    Pide una justificación adicional breve (2–3 frases) y neutral sobre
    por qué este fallo es adecuado para el caso, evitando repetir lo ya dicho.
    """
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

def run_research(user_query: str):
    """Pipeline de búsqueda y render con segundo pase de justificación adicional."""
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

    # Armar candidatos
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
        uid_to_docdict[uid] = {"metadata": meta, "page_content": getattr(d, "page_content", "") or ""}

    # LLM elige 3 y explica (con posible reintento si son muy similares)
    intro, picked = llm_pick_top3_and_explain(user_query, candidates)
    if not picked or not _distinct_strings([p.get("resumen","") for p in picked]):
        intro2, picked2 = llm_pick_top3_and_explain(
            user_query, candidates,
            retry_hint="ATENCIÓN: en el intento anterior las explicaciones fueron muy parecidas. "
                       "Ahora asegura diferencias concretas entre los casos (hechos, norma/artículo, "
                       "resultado, jurisdicción/fecha) usando fragmentos distintos del extracto."
        )
        if picked2:
            intro = intro2 or intro or "Analicé tu consulta y seleccioné los fallos más pertinentes."
            picked = picked2

    st.markdown(f"**{intro or 'Analicé tu consulta y seleccioné los fallos más pertinentes.'}**")

    # Guardar en estado para chat posterior
    st.session_state.picked_docs = [uid_to_docdict[x["uid"]] for x in picked if x["uid"] in uid_to_docdict]

    # Render resultados + detalles + segundo pase
    resumen_lines = []
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

        # 1) explicación principal
        st.markdown(f"**{i}. {titulo}**")
        for b in item.get("bullets", []):
            st.markdown(b)
        if item.get("resumen"):
            st.markdown(f"_**Conclusión:**_ {item['resumen']}")

        # 2) NUEVO: segundo pase de justificación adicional (2–3 frases)
        descriptor = " — ".join([x for x in [titulo, trib, fecha] if x])
        extracto = (docd["page_content"] or "")[:1600]
        ya_dicho = " | ".join(item.get("bullets", [])) + " | " + item.get("resumen", "")
        extra = llm_extra_why(user_query, descriptor, extracto, ya_dicho)
        if extra:
            st.markdown(f"_**Justificación adicional:**_ {extra}")

        # Detalles de la fila del CSV (tabla Columna | Contenido)
        with st.expander(header, expanded=(i == 1)):
            render_kv_table(meta)

        # Resumen breve: conclusión + extra
        resumen_lines.append(f"{i}. {titulo} — {item.get('resumen','')}".strip())
        if extra:
            resumen_lines.append(f"   ➤ {extra}")

    final_msg = "🧠 **Resumen breve (con justificación adicional):**\n" + "\n\n".join(resumen_lines)
    st.session_state.messages.append({"role": "assistant", "content": final_msg})
    st.chat_message("assistant").markdown(final_msg)

# =============================
# Chat sobre jurisprudencias activas
# =============================
def run_chat(user_message: str):
    docs = st.session_state.picked_docs
    if not docs:
        # si no hay, degradar a research
        return run_research(user_message)

    # mostrar siempre qué jurisprudencias están activas
    show_active_docs()

    # contexto
    ctx_blocks = []
    for idx, d in enumerate(docs, start=1):
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
        "Puedes comparar, resumir, o redactar borradores basándote en los fallos provistos.\n\n"
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
        run_research(user_input)
    else:
        run_chat(user_input)

# =============================
# Siempre visible: jurisprudencias activas (si hay)
# =============================
show_active_docs()


# =============================
# Nota:
# - Este código asume que tu vectorstore FAISS ya existe en 'vectorstore_jurisprudencia'
#   con documentos que contienen en page_content el texto del fallo y, si es posible,
#   metadata como 'caratula', 'tribunal_principal', 'fecha_sentencia', etc.
# - Si querés ajustar el umbral de similitud o filtros por jurisdicción/año,
#   podés crear el retriever con: vs.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.2})
# =============================
