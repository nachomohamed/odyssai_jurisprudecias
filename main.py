import streamlit as st
import os
import openai

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableMap

# --------------------- CONFIGURACIÓN DE PÁGINA ---------------------
st.set_page_config(page_title="Jurisprudencia Assistant", layout="wide")
st.title("⚖️ Jurisprudencia Assistant")

# --------------------- CLAVE DE OPENAI DESDE SECRETS ---------------------
openai_api_key = st.secrets["OPENAI_KEY"]  # ← asegúrate de que se llame así en tu secrets.toml
os.environ["OPENAI_API_KEY"] = openai_api_key

# Validar clave (esto lanza error si es inválida)
try:
    openai.api_key = openai_api_key
    openai.Model.list()
except openai.AuthenticationError:
    st.error("❌ API Key inválida. Revisá tu configuración en Secrets.")
    st.stop()

# --------------------- CARGAR VECTORSTORE ---------------------
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(
        "vectorstore_jurisprudencia",
        embeddings,
        allow_dangerous_deserialization=True
    ).as_retriever()

retriever = load_vectorstore()

# --------------------- PROMPT Y LLM ---------------------
system_prompt = (
    "You are an assistant specialized in jurisprudence. Use the retrieved legal context "
    "to answer user questions. If unsure, say you don't know. Keep answers concise and formal.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=openai_api_key)
qa_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = (
    RunnableMap({
        "context": RunnableLambda(lambda x: retriever.invoke(x["input"])),
        "input": lambda x: x["input"]
    }) | qa_chain
)

# --------------------- HISTORIAL DE CONVERSACIÓN ---------------------
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# --------------------- INTERFAZ DE CHAT ---------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Entrada del usuario
if user_input := st.chat_input("Escribí tu consulta legal..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    response = conversational_rag.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "juris-chat"}}
    )

    answer = response["answer"]
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)