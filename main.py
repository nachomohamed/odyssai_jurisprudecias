import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory

# =============================
# Configuración base
# =============================
st.set_page_config(page_title="⚖️ Jurisprudencia Assistant", layout="wide")
st.title("⚖️ Jurisprudencia Assistant")

# API Key desde Streamlit Secrets
openai_api_key = st.secrets["OPENAI_KEY"]

# =============================
# Carga del vectorstore (cacheado)
# =============================
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    try:
        vs = FAISS.load_local(
            "vectorstore_jurisprudencia",
            embeddings,
            allow_dangerous_deserialization=True
        )
        # Ajustamos a top-3 resultados
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        return retriever
    except Exception as e:
        st.error(f"❌ Error cargando el vectorstore: {e}")
        st.stop()

retriever = load_vectorstore()

# =============================
# LLM y prompts
# =============================
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=openai_api_key)

# Prompt para responder con contexto (si lo necesitás luego)
system_prompt = (
    "Sos un asistente especializado en jurisprudencia. Usá el contexto legal recuperado "
    "para responder las preguntas. Si no sabés, decilo claramente. Sé conciso y formal.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
# (Opcional) Cadena para QA estilo stuff si querés usarla en otro flujo
qa_chain = create_stuff_documents_chain(llm, prompt)

# =============================
# Memoria simple de chat (frontend)
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# =============================
# Helpers
# =============================
def summarize_doc(doc):
    """Arma un header prolijo con metadata si existe y devuelve (header, cuerpo)."""
    meta = getattr(doc, "metadata", {}) or {}
    titulo = meta.get("caratula") or meta.get("titulo") or meta.get("title") or "Jurisprudencia"
    tribunal = meta.get("tribunal_principal") or meta.get("tribunal") or ""
    fecha = meta.get("fecha_sentencia") or meta.get("fecha") or ""
    extra = " — ".join([x for x in [tribunal, fecha] if x])
    header = f"**{titulo}**" + (f" ({extra})" if extra else "")
    cuerpo = getattr(doc, "page_content", str(doc))
    return header, cuerpo

rationale_system = (
    "Eres un asistente jurídico. Explica en 2–4 frases por qué este fallo es útil para el caso del abogado, "
    "citando criterios de afinidad: hechos relevantes, tipo de acción, normas aplicadas, resultado, "
    "jurisdicción/instancia/fecha si aportan. Sé concreto y no inventes."
)

def explain_match(user_query, doc):
    """Pide al LLM una explicación breve de por qué el doc es útil para la consulta."""
    msgs = [
        {"role": "system", "content": rationale_system},
        {"role": "user", "content": f"Consulta del abogado:\n{user_query}\n\nFallo recuperado:\n{doc.page_content[:5000]}"},
    ]
    out = llm.invoke(msgs)
    return out.content if hasattr(out, "content") else str(out)

# =============================
# Interfaz de chat principal
# =============================
user_input = st.chat_input("Planteá tu caso (hechos, norma, jurisdicción, año, lo más específico posible)...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # 1) Recupero top-3 documentos
    try:
        docs = retriever.get_relevant_documents(user_input)
    except Exception:
        # Fallback para versiones que soportan invoke
        try:
            docs = retriever.invoke(user_input)
        except Exception as e:
            docs = []
            st.warning(f"No se pudo consultar el retriever: {e}")

    if not docs:
        answer = (
            "Estuve buscando y analizando tu caso, pero no encontré jurisprudencias relevantes en tu base. "
            "Probá reformular con más detalles (hechos clave, norma aplicable, jurisdicción, año)."
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
    else:
        st.markdown("**Estuve buscando y analizando tu caso y encontré estas 3:**")

        bloques = []
        for i, doc in enumerate(docs[:3], start=1):
            # 2) Explicación breve de por qué lo elegí
            razon = explain_match(user_input, doc)
            header, cuerpo = summarize_doc(doc)

            if i == 1:
                st.markdown(f"**1 - Elegí esta jurisprudencia por:** {razon}")
            elif i == 2:
                st.markdown(f"**2 - En segundo lugar, creo que esta te puede servir porque:** {razon}")
            else:
                st.markdown(f"**3 - Por último, sumé esta porque:** {razon}")

            with st.expander(header, expanded=(i == 1)):
                st.write(cuerpo)

            bloques.append((razon, header))

        # 3) Mensaje compacto al hilo del chat
        compact = "\n\n".join([f"{idx}. {raz}\n{hdr}" for idx, (raz, hdr) in enumerate(bloques, start=1)])
        final_msg = f"**Resumen breve:**\n{compact}"
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
