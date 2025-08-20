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
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 50,
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

# =============================
# LLM: elegir 3 y explicar con bullets
# =============================
def llm_pick_top3_and_explain(user_query: str, candidates: list[dict]):
    """
    candidates: [{uid, descriptor, extracto}]
    Devuelve (intro:str, items:list[{uid, bullets, resumen}])
    """
    system = (
        "Eres un asistente jurídico. Recibirás hasta 10 fallos candidatos. "
        "Debes elegir EXACTAMENTE 3. "
        "Para cada uno: explica con detalle en viñetas por qué es relevante. "
        "Incluye hechos clave, artículo/norma (ej. art. 242 LCT si aparece), "
        "jurisdicción/instancia/fecha, y resultado/criterio. "
        "Añade al menos una cita breve entre comillas (≤12 palabras) del extracto. "
        "Sé claro y evita frases genéricas."
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

    out = llm.invoke([
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
        if not result:
            raise ValueError("Modelo devolvió vacío")
        return intro, result
    except Exception:
        intro = "Tras revisar tu consulta, seleccioné los 3 fallos más cercanos por hechos y normativa."
        result = [{"uid": c["uid"],
                   "bullets": [
                       "• Hechos análogos a los planteados.",
                       "• Norma o artículo citado en el extracto.",
                       "• Resultado y criterio similar al interés del abogado."
                   ],
                   "resumen": "Relevante para sustentar la demanda en curso."}
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

    # Recupero 10 candidatos
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
        # Preparo lista para LLM con uid único
        candidates = []
        uid_to_doc = {}
        for d in candidate_docs[:10]:
            m = d.metadata or {}
            titulo = m.get("caratula") or m.get("titulo") or "Jurisprudencia"
            trib = m.get("tribunal_principal") or m.get("tribunal") or ""
            fecha = m.get("fecha_sentencia") or ""
            tipo = m.get("tipo_causa") or ""
            descriptor = " — ".join([x for x in [titulo, trib, fecha, tipo] if x])
            extracto = (d.page_content or "")[:1600]
            uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, (descriptor + extracto[:200]).strip()))
            candidates.append({"uid": uid, "descriptor": descriptor, "extracto": extracto})
            uid_to_doc[uid] = d

        # El LLM elige 3 y explica
        intro, picked = llm_pick_top3_and_explain(user_input, candidates)
        st.markdown(f"**{intro}**")

        resumen_lineas = []
        for i, item in enumerate(picked, start=1):
            uid = item["uid"]
            d = uid_to_doc.get(uid)
            if not d:
                continue
            meta = d.metadata or {}
            titulo = meta.get("caratula") or meta.get("titulo") or "Jurisprudencia"
            trib = meta.get("tribunal_principal") or meta.get("tribunal") or ""
            fecha = meta.get("fecha_sentencia") or ""
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
# Nota:
# - Este código asume que tu vectorstore FAISS ya existe en 'vectorstore_jurisprudencia'
#   con documentos que contienen en page_content el texto del fallo y, si es posible,
#   metadata como 'caratula', 'tribunal_principal', 'fecha_sentencia', etc.
# - Si querés ajustar el umbral de similitud o filtros por jurisdicción/año,
#   podés crear el retriever con: vs.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.2})
# =============================
