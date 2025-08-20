# app.py
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="⚖️ Jurisprudencia Assistant", layout="wide")
st.title("⚖️ Jurisprudencia Assistant")

openai_api_key = st.secrets["OPENAI_KEY"]

@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vs = FAISS.load_local(
        "vectorstore_jurisprudencia",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vs.as_retriever(search_kwargs={"k": 3})

retriever = load_retriever()
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=openai_api_key)

rationale_system = (
    "Eres un asistente jurídico. Explica en 2–4 frases por qué este fallo es útil para el caso del abogado, "
    "citando criterios de afinidad: hechos relevantes, tipo de acción, normas aplicadas, resultado, "
    "jurisdicción/instancia/fecha si aportan. Sé concreto y no inventes."
)

def explain_match(user_query, doc):
    msgs = [
        {"role": "system", "content": rationale_system},
        {"role": "user", "content": f"Consulta del abogado:\n{user_query}\n\nFallo recuperado:\n{doc.page_content[:5000]}"},
    ]
    out = llm.invoke(msgs)
    return out.content if hasattr(out, "content") else str(out)

# Orden sugerido de columnas para mostrar primero
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

    # Armo filas en orden preferido primero
    rows = []
    seen = set()
    for k in DISPLAY_ORDER:
        if k in meta and meta[k] not in (None, "", "nan"):
            val = str(meta[k])
            if k in ["sumario", "texto"]:
                # Los largos, en expander aparte
                continue
            rows.append({"Columna": k, "Contenido": val})
            seen.add(k)

    # Luego resto de campos que vengan en metadata
    for k, v in meta.items():
        if k in seen or k in ["sumario", "texto"]:
            continue
        if v not in (None, "", "nan"):
            rows.append({"Columna": k, "Contenido": str(v)})

    if rows:
        st.table(pd.DataFrame(rows))

    # Campos largos en expanders separados
    if meta.get("sumario"):
        with st.expander("sumario", expanded=True):
            st.write(str(meta["sumario"]))
    if meta.get("texto"):
        with st.expander("texto", expanded=False):
            st.write(str(meta["texto"]))

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

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
        answer = ("Estuve buscando y analizando tu caso, pero no encontré jurisprudencias "
                  "relevantes en tu base. Afiná la consulta con hechos/norma/jurisdicción/año.")
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
    else:
        st.markdown("**Estuve buscando y analizando tu caso y encontré estas 3:**")

        resumen_lineas = []
        for i, doc in enumerate(docs[:3], start=1):
            razon = explain_match(user_input, doc)
            titulo = doc.metadata.get("caratula") or doc.metadata.get("titulo") or "Jurisprudencia"

            if i == 1:
                st.markdown(f"**1 - Elegí esta jurisprudencia por:** {razon}")
            elif i == 2:
                st.markdown(f"**2 - En segundo lugar, creo que esta te puede servir porque:** {razon}")
            else:
                st.markdown(f"**3 - Por último, sumé esta porque:** {razon}")

            with st.expander(f"**{titulo}**", expanded=(i == 1)):
                render_kv_table(doc.metadata)  # <- **Columna | Contenido**

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
