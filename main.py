# app.py
import json
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# =============================
# Configuración base
# =============================
st.set_page_config(page_title="⚖️ Jurisprudencia Assistant", layout="wide")
st.title("⚖️ Jurisprudencia Assistant")

# API Key desde Streamlit Secrets
openai_api_key = st.secrets["OPENAI_KEY"]

# =============================
# Carga del retriever (cacheado)
# =============================
@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # podés subir a -3-large si querés
        openai_api_key=openai_api_key
    )
    vs = FAISS.load_local(
        "vectorstore_jurisprudencia",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vs.as_retriever(search_kwargs={"k": 3})

retriever = load_retriever()

# =============================
# LLM
# =============================
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=openai_api_key)

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
    rows = []
    seen = set()

    # Primero columnas más útiles
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
# Helper LLM: intro + razones en JSON
# =============================
def llm_intro_and_reasons(llm, user_query, docs):
    """
    Devuelve (intro:str, items:list[str]) generados por el LLM en JSON:
    {
      "intro": "...",
      "items": ["razón 1", "razón 2", "razón 3"]
    }
    """
    # Descriptores breves por doc (título/tribunal/fecha + extracto)
    descriptors = []
    for d in docs:
        m = d.metadata or {}
        titulo = m.get("caratula") or m.get("titulo") or "Jurisprudencia"
        trib = m.get("tribunal_principal") or m.get("tribunal") or ""
        fecha = m.get("fecha_sentencia") or m.get("fecha") or ""
        descriptor = f"{titulo}" + (f" — {trib}" if trib else "") + (f" — {fecha}" if fecha else "")
        body = d.page_content[:1200] if getattr(d, "page_content", None) else ""
        descriptors.append({"descriptor": descriptor, "extracto": body})

    system = (
        "Eres un asistente jurídico. Vas a presentar resultados de búsqueda de jurisprudencia para un abogado. "
        "Varía el estilo de redacción (no uses siempre las mismas frases). No inventes. Sé profesional y concreto."
    )
    user = (
        "Consulta del abogado:\n"
        f"{user_query}\n\n"
        "Fallos recuperados (descriptor + extracto parcial):\n"
        f"{json.dumps(descriptors, ensure_ascii=False)}\n\n"
        "TAREA:\n"
        "1) Escribe una INTRODUCCIÓN breve (1–2 frases) explicando que analizaste el caso y hallaste resultados. "
        "   Cambia el estilo (evita fórmulas repetidas).\n"
        "2) Escribe una explicación para CADA fallo (2–4 frases) justificando su pertinencia (hechos, tipo de acción, "
        "   normas aplicadas, resultado, jurisdicción/instancia/fecha si aportan).\n"
        "3) Devuelve SOLO JSON válido EXACTAMENTE con este formato:\n"
        '{\n  "intro": "texto",\n  "items": ["razón 1", "razón 2", "razón 3"]\n}\n'
        "   El tamaño de 'items' debe coincidir con la cantidad de fallos (máx 3)."
    )

    out = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])
    text = out.content if hasattr(out, "content") else str(out)

    try:
        data = json.loads(text)
        intro = (data.get("intro") or "").strip()
        items = data.get("items") or []
        items = items[:len(docs)]
        if not intro or not items:
            raise ValueError("JSON sin intro/items")
        return intro, items
    except Exception:
        # Fallback por si el modelo no devuelve JSON parseable
        intro_fb = "Analicé tu consulta y seleccioné los fallos que mejor se ajustan por similitud fáctica y encuadre normativo."
        items_fb = []
        for _ in docs:
            items_fb.append(
                "Resulta pertinente por la cercanía de los hechos, la norma aplicada y el criterio decidido en el extracto provisto."
            )
        return intro_fb, items_fb

# =============================
# Memoria simple de chat (frontend)
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# =============================
# Interfaz
# =============================
user_input = st.chat_input("Planteá tu caso (hechos, norma, jurisdicción, año, etc.)...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # Recupero top-3
    try:
        docs = retriever.get_relevant_documents(user_input)
    except Exception:
        docs = retriever.invoke(user_input)

    if not docs:
        answer = (
            "No encontré jurisprudencias relevantes en tu base. Probá aportar más detalles "
            "(hechos clave, norma aplicable, jurisdicción, período)."
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
    else:
        # LLM redacta intro + razones (varía el wording)
        intro, reasons = llm_intro_and_reasons(llm, user_input, docs[:3])
        st.markdown(f"**{intro}**")

        resumen_lineas = []
        for i, (doc, razon) in enumerate(zip(docs[:3], reasons), start=1):
            meta = doc.metadata or {}
            titulo = meta.get("caratula") or meta.get("titulo") or "Jurisprudencia"
            trib = meta.get("tribunal_principal") or meta.get("tribunal") or ""
            fecha = meta.get("fecha_sentencia") or meta.get("fecha") or ""
            header = f"**{titulo}**" + (f" — {trib}" if trib else "") + (f" — {fecha}" if fecha else "")

            st.markdown(f"**{i}.** {razon}")
            with st.expander(header, expanded=(i == 1)):
                render_kv_table(meta)

            resumen_lineas.append(f"{i}. {razon}\n{titulo}")

        final_msg = "**Resumen breve:**\n" + "\n\n".join(resumen_lineas)
        st.session_state.messages.append({"role": "assistant", "content": final_msg})
        st.chat_message("assistant").markdown(final_msg)

# =============================
# Nota:
# - Este código asume que tu vectorstore FAISS ya existe en 'vectorstore_jurisprudencia'
#   con documentos que contienen en page_content el texto del fallo y, si es posible,
#   metadata como 'caratula', 'tribunal_principal', 'fecha_sentencia', etc.
# - Si querés ajustar el umbral de similitud o filtros por jurisdicción/año,
#   podés crear el retriever con: vs.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.2})
# =============================
