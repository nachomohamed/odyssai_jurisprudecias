import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableMap


# Configuración de la página
st.set_page_config(page_title="Jurisprudencia Assistant", layout="wide")
st.title("⚖️ Jurisprudencia Assistant")

# Sidebar: clave de OpenAI
openai_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
if not openai_api_key:
    st.sidebar.warning("Por favor, ingresá tu OpenAI API Key.")
    st.stop()

# --- Cargar vectorstore desde disco ---
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local("vectorstore_jurisprudencia", embeddings).as_retriever()

retriever = load_vectorstore()

# --- Prompt y LLM ---
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
    RunnableMap({"context": retriever, "input": lambda x: x["input"]})
    | qa_chain
)
# --- Memoria de conversación ---
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

# --- Interfaz de chat ---
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

    answer = response["answer"]
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)