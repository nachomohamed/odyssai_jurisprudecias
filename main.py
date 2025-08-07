import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Configuración
st.set_page_config(page_title="Jurisprudencia Assistant", layout="wide")
st.title("⚖️ Jurisprudencia Assistant")

# Obtener API key desde Streamlit Secrets
openai_api_key = st.secrets["OPENAI_KEY"]

# Vectorstore cacheado
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    try:
        return FAISS.load_local(
            "vectorstore_jurisprudencia",
            embeddings,
            allow_dangerous_deserialization=True
        ).as_retriever()
    except Exception as e:
        st.error(f"❌ Error cargando el vectorstore: {e}")
        st.stop()

retriever = load_vectorstore()

# Prompt y LLM
system_prompt = (
    "Sos un asistente especializado en jurisprudencia. Usá el contexto legal recuperado "
    "para responder las preguntas. Si no sabés, decilo claramente. Sé conciso y formal.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=openai_api_key)
qa_chain = create_stuff_documents_chain(llm, prompt)

# RAG Chain (contexto + pregunta)
rag_chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["input"]),
    "input": lambda x: x["input"]
}) | qa_chain

# Memoria de conversación
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Interfaz de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if user_input := st.chat_input("Escribí tu consulta legal..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    response = conversational_rag.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "juris-chat"}}
    )

    # Corregido: la respuesta ya es un string
    answer = response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)
