import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

model = "gpt-4o-mini"

# Page config
st.set_page_config(page_title="LangChain Chatbot", layout="wide")

# API Key input (only once)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.sidebar.warning("Please provide your OpenAI API Key.")
    st.stop()

st.set_page_config(page_title="Juriprudencia Chatbot", layout="wide")
st.title("⚖️ Jurisprudencia Assistant")

# --- Initialize Vector Store (only once per session) ---

from langchain_community.document_loaders import UnstructuredFileLoader
import glob

@st.cache_resource
def init_vectorstore():
    # Buscar todos los archivos .doc en la carpeta data
    file_paths = glob.glob("data/*.doc")
    docs = []
    for file_path in file_paths:
        loader = UnstructuredFileLoader(file_path)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key)
    )   
    return vectorstore.as_retriever()

retriever = init_vectorstore()

# --- Setup LangChain ---
system_prompt = (
    "You are an assistant specialized in jurisprudence. Use the retrieved legal context "
    "to answer user questions. If unsure, say you don't know. Keep answers concise and formal.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

llm = ChatOpenAI(model=model, temperature=0.3, openai_api_key=openai_api_key)
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# --- Memory with RunnableWithMessageHistory ---
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if query := st.chat_input("Ask about a jurisprudencia case..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    response = conversational_rag.invoke(
        {"input": query},
        config={"configurable": {"session_id": "juris-001"}}
    )

    answer = response["answer"]
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)