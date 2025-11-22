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
# METADATA LISTS (Dynamic Loading)
# =====================================
import json

METADATA_JSON_PATH = 'metadata_values.json'

def load_metadata_lists():
    try:
        if os.path.exists(METADATA_JSON_PATH):
            with open(METADATA_JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('tribunales', []), data.get('salas', [])
    except Exception as e:
        print(f"Error cargando metadatos desde JSON: {e}")
    return [], []

TRIBUNALES, SALAS = load_metadata_lists()

# Fallback si están vacíos (para evitar errores en Pydantic si no hay JSON)
if not TRIBUNALES:
    TRIBUNALES = ["Corte Suprema de Justicia", "Cámara Civil y Comercial"] # Mínimo default
if not SALAS:
    SALAS = ["Sala I", "Sala II"] # Mínimo default

# =====================================
# MODELS (Pydantic)
# =====================================
class SearchFilters(BaseModel):
    tribunal: Optional[str] = Field(None, description=f"Tribunal específico. Opciones válidas: {TRIBUNALES}")
    sala: Optional[str] = Field(None, description=f"Sala específica. Opciones válidas: {SALAS}")
    tipo_causa: Optional[str] = Field(None, description="Tipo de causa, ej: 'Despido', 'Accidente', 'Amparo'. (Usar términos generales)")
    anio_min: Optional[int] = Field(None, description="Año de inicio del rango (o año exacto). Ej: 2010")
    anio_max: Optional[int] = Field(None, description="Año de fin del rango. Ej: 2020")

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

system_analysis = f"""
Sos un asistente inteligente que clasifica la intención del usuario en un sistema legal.

1. Determinar la INTENCIÓN (intent):
   - "SEARCH": Si el usuario pide buscar fallos, sentencias, jurisprudencia, casos parecidos, o da criterios específicos.
     **IMPORTANTE:** Si el usuario dice "hace una búsqueda más general", "busca de nuevo", "probá con otros términos", TAMBIÉN es "SEARCH" (con filtros más amplios o sin filtros).
   - "CHAT": SOLO si el usuario saluda, hace preguntas teóricas generales SIN intención de buscar casos, o pide redactar un escrito genérico.

2. Si es "SEARCH", extraer filtros y definir la "search_query".
   - TRIBUNALES VÁLIDOS: {TRIBUNALES}
   - SALAS VÁLIDAS: {SALAS}
   - Si el usuario menciona un tribunal o sala parecido, MAPEARLO a uno de la lista. Si no coincide, dejar None.
   - Fechas: "entre 2010 y 2020" -> anio_min=2010, anio_max=2020.
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
Sos un asistente virtual para abogados con acceso a una base de datos de jurisprudencia local.
Tu tono es profesional, conciso y técnico.

Tus objetivos:
1. Ayudar en la redacción, brainstorming o dudas teóricas.
2. Si el usuario te pide buscar fallos y NO encontraste nada antes, sugerile términos de búsqueda alternativos.
3. **NUNCA digas que no tienes acceso a bases de datos.** Tienes una base interna. Si no encontraste nada, di "No encontré resultados en mi base de datos interna para esa consulta".
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
    print(f"--- [DEBUG] Analizando query: '{user_query}' ---")
    try:
        result: QueryAnalysis = analysis_chain.invoke({"input": user_query})
        print(f"--- [DEBUG] Resultado análisis: Intent={result.intent}, Filters={result.filters} ---")
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
    # REMOVED: tipo_causa is too strict for metadata filtering. 
    # We rely on vector search for the topic.
    # if filters.get("tipo_causa"):
    #    conditions.append({"tipo_causa": filters["tipo_causa"]})
    
    # REMOVED: anio filter causing crash with $contains
    
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
    print(f"--- [DEBUG] Inicio de búsqueda. Query: '{query}' ---")
    print(f"--- [DEBUG] Filtros recibidos: {filters} ---")
    
    # 1. Construir filtros para Chroma (excluyendo año)
    where = build_query_filters(filters) if filters else None
    print(f"--- [DEBUG] Filtros Chroma (where): {where} ---")
    
    # 2. Traer más resultados para filtrar después
    # Si hay filtro de año, traemos más para compensar los que descartaremos
    has_year_filter = filters and (filters.get("anio_min") or filters.get("anio_max"))
    fetch_k = k * 10 if has_year_filter else (k * 5 if USE_RERANK else k)
    print(f"--- [DEBUG] Fetch K: {fetch_k} (Rerank: {USE_RERANK}, YearFilter: {has_year_filter}) ---")
    
    print("--- [DEBUG] Ejecutando col.query en Chroma... ---")
    results = col.query(
        query_texts=[query],
        n_results=fetch_k,
        where=where
    )
    print("--- [DEBUG] Chroma retornó resultados. Procesando... ---")
    
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    
    items = [{"id": i, "texto": d, "metadata": m} for i, d, m in zip(ids, docs, metadatas)]
    print(f"--- [DEBUG] Items recuperados de Chroma: {len(items)} ---")
    
    # 3. Filtrado Manual por Año (Post-processing)
    if has_year_filter:
        min_y = filters.get("anio_min")
        max_y = filters.get("anio_max")
        print(f"--- [DEBUG] Aplicando filtro de año: Min={min_y}, Max={max_y} ---")
        
        filtered_items = []
        import re
        
        for item in items:
            # Intentar extraer año de la fecha (asumiendo formato DD/MM/YYYY o YYYY)
            fecha_str = str(item["metadata"].get("fecha_sentencia", ""))
            
            # Buscar 4 dígitos seguidos que parezcan un año (19xx o 20xx)
            match = re.search(r'(19|20)\d{2}', fecha_str)
            if match:
                year = int(match.group(0))
                
                # Aplicar rango
                if min_y and year < min_y:
                    continue
                if max_y and year > max_y:
                    continue
                
                filtered_items.append(item)
            else:
                # Si no encontramos fecha, decidimos si incluirlo o no. 
                # Por defecto, si el usuario pide fecha específica, mejor excluir los que no tienen fecha.
                pass
        
        print(f"--- [DEBUG] Items después de filtro de año: {len(filtered_items)} ---")
        items = filtered_items
    
    if not items:
        print("--- [DEBUG] No quedaron items después del filtrado. ---")
        return []

    # 4. Reranking
    if USE_RERANK:
        print("--- [DEBUG] Iniciando Reranking... ---")
        idxs, scores = rerank(query, [x["texto"] for x in items])
        print("--- [DEBUG] Reranking finalizado. ---")
        ranked_items = []
        for pos, idx in enumerate(idxs[:k]):
            item = items[idx].copy()
            item["score"] = float(scores[pos])
            ranked_items.append(item)
        return ranked_items
    
    return items[:k]
