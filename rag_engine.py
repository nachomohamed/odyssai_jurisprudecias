import os
import chromadb
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel as LCBaseModel # For compatibility if needed, usually standard pydantic works with new LC

# =====================================
# CONFIG
# =====================================
CHROMA_DIR = "./chroma_juris"
DEFAULT_COLLECTION = "jurisprudencia"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OPENAI_MODEL = "gpt-4o-mini"
USE_RERANK = True

# =====================================
# API KEY SETUP
# =====================================
try:
    import streamlit as st
    # Intentar cargar desde secrets si no está en variables de entorno
    if "OPENAI_API_KEY" not in os.environ:
        # El usuario indicó que se llama "OPENAI_KEY" en sus secrets
        if "OPENAI_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]
        elif "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass # Si falla (ej: no estamos en streamlit), confiamos en que ya esté en env vars

# =====================================
# MODELS (Pydantic)
# =====================================
class SearchFilters(BaseModel):
    tribunal: Optional[str] = Field(None, description="Tribunal específico, ej: 'Corte Suprema', 'Cámara Civil'")
    sala: Optional[str] = Field(None, description="Sala específica, ej: 'Sala I', 'Sala 2'")
    tipo_causa: Optional[str] = Field(None, description="Tipo de causa, ej: 'Despido', 'Accidente'")
    anio: Optional[int] = Field(None, description="Año numérico de interés, ej: 2023")

class QueryAnalysis(BaseModel):
    intent: str = Field(..., description="Intención del usuario: 'SEARCH' (buscar fallos) o 'CHAT' (conversación general).")
    filters: SearchFilters = Field(default_factory=SearchFilters, description="Filtros extraídos si la intención es SEARCH.")
    search_query: str = Field(..., description="Texto optimizado para la búsqueda vectorial.")

# =====================================
# CHROMA SETUP
# =====================================
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        col = client.get_collection(name=DEFAULT_COLLECTION)
    except Exception:
        cols = client.list_collections()
        if not cols:
            raise RuntimeError(f"❌ No hay colecciones en {CHROMA_DIR}")
        col = client.get_collection(name=cols[0].name)
    return col

# =====================================
# LANGCHAIN COMPONENTS
# =====================================
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# --- Intent Analysis Chain ---
# Usamos with_structured_output para forzar el JSON schema
analyzer_llm = llm.with_structured_output(QueryAnalysis)

system_analysis = """
Sos un asistente inteligente que clasifica la intención del usuario en un sistema legal.

1. Determinar la INTENCIÓN (intent):
   - "SEARCH": Si el usuario pide buscar fallos, sentencias, jurisprudencia, casos parecidos, o da criterios específicos.
   - "CHAT": Si el usuario saluda, hace preguntas teóricas generales, pide redactar un escrito SIN buscar casos específicos, o conversa sobre lo que ya se mostró.

2. Si es "SEARCH", extraer filtros (tribunal, sala, tipo_causa, anio) y definir la "search_query".
"""

analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", system_analysis),
    ("user", "{input}")
])

analysis_chain = analysis_prompt | analyzer_llm

# --- RAG Response Chain ---
rag_system_prompt = """
Sos un asistente legal experto. Tu tarea es presentar los fallos encontrados al usuario.

Para cada fallo recuperado:
1. Menciona brevemente sus datos clave (Carátula, Tribunal, Fecha).
2. Explica POR QUÉ es relevante para la consulta del usuario. ¿Qué punto jurídico toca que coincide con lo buscado?

No inventes información. Basate solo en los fragmentos provistos.
"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt),
    ("user", "Consulta: {query}\n\nFallos recuperados:\n{context}")
])

rag_chain = rag_prompt | llm | StrOutputParser()

# --- Chat Response Chain ---
chat_system_prompt = """
Sos un asistente virtual para abogados. 
Tu tono es profesional, conciso y técnico.

Tus objetivos:
1. Ayudar en la redacción, brainstorming o dudas teóricas.
2. NO desviarte del tema legal. Si te preguntan de cocina o deportes, cortésmente volvé al derecho.
3. Si el usuario pide buscar fallos, sugerile que sea específico con los términos.
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", chat_system_prompt),
    ("placeholder", "{history}") # LangChain maneja la lista de mensajes aquí
])

chat_chain = chat_prompt | llm | StrOutputParser()


# =====================================
# PUBLIC API
# =====================================
def analyze_query(user_query: str) -> Dict[str, Any]:
    """
    Usa LangChain para analizar la query.
    Retorna dict compatible con la app: {"intent":..., "filters":..., "search_query":...}
    """
    try:
        result: QueryAnalysis = analysis_chain.invoke({"input": user_query})
        # Convertimos a dict simple para facilitar uso en app
        return {
            "intent": result.intent,
            "filters": result.filters.dict(),
            "search_query": result.search_query
        }
    except Exception as e:
        print(f"Error en analyze_query: {e}")
        return {"intent": "CHAT", "filters": {}, "search_query": user_query}

def generate_rag_response(user_query: str, retrieved_items: List[Dict]) -> str:
    """
    Genera respuesta RAG usando LangChain.
    """
    context_str = ""
    for i, item in enumerate(retrieved_items, 1):
        meta = item["metadata"]
        context_str += f"--- FALLO {i} ---\n"
        context_str += f"Carátula: {meta.get('caratula', '?')}\n"
        context_str += f"Tribunal: {meta.get('tribunal_principal', '?')}\n"
        context_str += f"Fecha: {meta.get('fecha_sentencia', '?')}\n"
        context_str += f"Texto: {item['texto'][:800]}...\n\n"
    
    return rag_chain.invoke({"query": user_query, "context": context_str})

def generate_chat_response(history: List[Dict]) -> str:
    """
    Genera respuesta Chat usando LangChain.
    history: lista de dicts {"role":..., "content":...}
    """
    # Convertir formato de historial a formato LangChain si fuera necesario, 
    # pero ChatPromptTemplate suele aceptar lista de (role, content) o mensajes.
    # Aquí pasamos la lista de mensajes tal cual, asumiendo que el prompt template lo maneja
    # o lo convertimos a tuplas (role, content).
    
    lc_history = []
    for msg in history:
        role = msg["role"]
        if role == "user":
            lc_history.append(("user", msg["content"]))
        elif role == "assistant":
            lc_history.append(("assistant", msg["content"]))
        # system ya está en el prompt template
            
    return chat_chain.invoke({"history": lc_history})


# =====================================
# SEARCH ENGINE (Custom Logic Preserved)
# =====================================
def build_query_filters(filters):
    conditions = []
    if filters.get("tribunal"):
        conditions.append({"tribunal_principal": filters["tribunal"]})
    if filters.get("sala"):
        conditions.append({"tribunal_sala": filters["sala"]})
    if filters.get("tipo_causa"):
        conditions.append({"tipo_causa": filters["tipo_causa"]})
    if filters.get("anio"):
        conditions.append({"fecha_sentencia": {"$contains": str(filters["anio"])}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}

def rerank(query, docs):
    try:
        from sentence_transformers import CrossEncoder
        import numpy as np
        
        model = CrossEncoder(CROSS_ENCODER_NAME)
        pairs = [[query, d] for d in docs]
        scores = model.predict(pairs)
        
        sorted_indices = np.argsort(-scores)
        return sorted_indices, scores[sorted_indices]
    except Exception:
        return list(range(len(docs))), [1.0] * len(docs)

def search(col, query, filters=None, k=3):
    where = build_query_filters(filters) if filters else None
    fetch_k = k * 5 if USE_RERANK else k
    
    results = col.query(
        query_texts=[query],
        n_results=fetch_k,
        where=where
    )
    
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    
    items = [{"id": i, "texto": d, "metadata": m} for i, d, m in zip(ids, docs, metadatas)]
    
    if not items:
        return []

    if USE_RERANK:
        idxs, scores = rerank(query, [x["texto"] for x in items])
        ranked_items = []
        for pos, idx in enumerate(idxs[:k]):
            item = items[idx].copy()
            item["score"] = float(scores[pos])
            ranked_items.append(item)
        return ranked_items
    
    return items[:k]
