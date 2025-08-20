# app.py
import json
import uuid
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
# Carga del retriever (MMR, k=10)
# =============================
@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
    vs = FAISS.load_local(
        "vectorstore_jurisprudencia",
        embeddings,
        allow_dangerous_deserialization=True
    )
    # Usamos MMR para diversidad, y pedimos más candidatos
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,        # devolver hasta 10 candidatos
            "fetch_k": 50,  # pool grande para diversidad
            "lambda_mult": 0.5
        }
    )
    return retriever

retriever = load_retriever()
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=openai_api_key)

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

    # Primero columnas clave en orden deseado
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
# LLM: elegir 3 de 10 y justificar (con UID)
# =============================
def llm_pick_top3_and_explain(user_query: str, candidates: list[dict]):
    """
    candidates: lista de dicts con:
      - uid: str
      - descriptor: str  (carátula — tribunal — fecha — tipo si hay)
      - extracto: str   (recorte del contenido)
    Devuelve (intro:str, items:list[{uid, razon}])
    """
    system = (
        "Eres un asistente jurídico. Se te dan hasta 10 fallos candidatos para un caso. "
        "Debes elegir EXACTAMENTE 3 y justificarlos con claridad. "
        "Criterio: coincidencia fáctica, norma/artículo aplicado, resultado, jurisdicción/instancia/fecha. "
        "No inventes. Sé específico y profesional. Evita duplicados."
    )
    # Para no pasar metadata completa, mandamos descriptor + extracto
    user = (
        "Consulta del abogado:\n"
        f"{user_query}\n\n"
        "Fallos candidatos (lista de objetos con uid, descriptor y extracto parcial):\n"
        f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
        "TAREA:\n"
        "1) Selecciona los 3 fallos más relevantes (NO más de 3, NO menos de 3).\n"
        "2) Para cada uno, explica por qué lo elegiste en 3–5 frases concretas. "
        "   Menciona hechos, norma/artículo (si aparece), resultado/criterio, y jurisdicción/instancia/fecha si aportan. "
        "   Puedes incluir 1–2 citas cortas del extracto entre comillas (≤12 palabras) si ayudan.\n"
        "3) Devuelve SOLO JSON válido con este formato EXACTO:\n"
        "{\n"
        '  "intro": "texto",\n'
        '  "items": [\n'
        '    {"uid": "UID_DEL_FALLO_1", "razon": "explicación detallada"},\n'
        '    {"uid": "UID_DEL_FALLO_2", "razon": "explicación detallada"},\n'
        '    {"uid": "UID_DEL_FALLO_3", "razon": "explicación detallada"}\n'
        "  ]\n"
        "}\n"
        "IMPORTANTE: usa los uid EXACTOS provistos. No inventes uids."
    )

    out = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])
    text = out.content if hasattr(out, "content") else str(out)

    # Parse robusto
    try:
        data = json.loads(text)
        intro = (data.get("intro") or "").strip()
        items = data.get("items") or []
        # Normalización mínima
        result = []
        for it in items[:3]:
            uid = (it.get("uid") or "").strip()
            razon = (it.get("razon") or "").strip()
            if uid and razon:
                result.append({"uid": uid, "razon": razon})
        # Si el modelo devolvió menos de 3 válidos, completamos con fallback
        while len(result) < 3 and candidates:
            # Agrego candidatos no usados aún
            used = {x["uid"] for x in result}
            for c in candidates:
                if c["uid"] not in used:
                    result.append({"uid": c["uid"], "razon": "Pertinente por coincidencias fácticas y normativas."})
                    break
        # Cortamos a 3
        result = result[:3]
        if not intro:
            intro = "Analicé tu consulta y seleccioné los fallos más relevantes entre las opciones recuperadas."
        return intro, result
    except Exception:
        # Fallback completo si el JSON no parsea
        intro = "Analicé tu consulta y seleccioné los fallos más relevantes entre las opciones recuperadas."
        # Tomo los 3 primeros candidatos como plan B
        result = [{"uid": c["uid"], "razon": "Pertinente por coincidencias fácticas y normativas."}
                  for c in candidates[:3]]
        return intro, result

# =============================
# Memoria visible
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# =============================
# Interfaz principal
# =============================
user_input = st.chat_input("Planteá tu caso (hechos, norma, jurisdicción, año, etc.)...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # 1) Recupero hasta 10 candidatos del retriever
    try:
        candidate_docs = retriever.get_relevant_documents(user_input)
    except Exception:
        candidate_docs = retriever.invoke(user_input)

    if not candidate_docs:
        answer = ("No encontré jurisprudencias relevantes en tu base. Probá aportar más detalles "
                  "(hechos clave, norma aplicable, jurisdicción, período).")
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
    else:
        # 2) Preparo lista para el LLM con uid estable y descriptor legible
        candidates = []
        uid_to_doc = {}
        for d in candidate_docs[:10]:
            m = d.metadata or {}
            titulo = m.get("caratula") or m.get("titulo") or "Jurisprudencia"
            trib = m.get("tribunal_principal") or m.get("tribunal") or ""
            fecha = m.get("fecha_sentencia") or m.get("fecha") or ""
            tipo = m.get("tipo_causa") or ""
            descriptor = " — ".join([x for x in [titulo, trib, fecha, tipo] if x])
            extracto = (d.page_content or "")[:1600]
            # UID determinístico por si se repite en otra corrida (hash de campos clave + recorte)
            uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, (descriptor + extracto[:200]).strip()))
            candidates.append({"uid": uid, "descriptor": descriptor, "extracto": extracto})
            uid_to_doc[uid] = d

        # 3) Le pido al LLM que elija 3 y justifique
        intro, picked = llm_pick_top3_and_explain(user_input, candidates)
        st.markdown(f"**{intro}**")

        resumen_lineas = []
        for i, item in enumerate(picked, start=1):
            uid = item["uid"]
            razon = item["razon"]
            d = uid_to_doc.get(uid)
            if not d:
                # Si por alguna razón el LLM devolvió un uid inexistente, me salteo
                continue

            meta = d.metadata or {}
            titulo = meta.get("caratula") or meta.get("titulo") or "Jurisprudencia"
            trib = meta.get("tribunal_principal") or meta.get("tribunal") or ""
            fecha = meta.get("fecha_sentencia") or meta.get("fecha") or ""
            header = f"**{titulo}**" + (f" — {trib}" if trib else "") + (f" — {fecha}" if fecha else "")

            st.markdown(f"**{i}.** {razon}")
            with st.expander(header, expanded=(i == 1)):
                render_kv_table(meta)

            resumen_lineas.append(f"{i}. {titulo} — {razon}")

        final_msg = "🧠 **Resumen breve:**\n" + "\n\n".join(resumen_lineas)
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
