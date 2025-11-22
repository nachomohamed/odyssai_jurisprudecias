import streamlit as st
import rag_engine
import os
import gdown

# =====================================
# DEPLOYMENT CONFIG
# =====================================
# ID del archivo en Google Drive (proporcionado por el usuario)
FILE_ID = '10J4VsnOel0Njd_mkUsJZ9kMqo8JQ3O1r'
# Ruta donde ChromaDB espera encontrar la base de datos
DB_PATH = 'chroma_juris/chroma.sqlite3'

@st.cache_resource
def download_db_if_missing():
    """Descarga la base de datos desde Google Drive si no existe localmente."""
    if not os.path.exists(DB_PATH):
        with st.spinner("Descargando base de datos de jurisprudencia (3.3GB)... esto puede tardar unos minutos."):
            try:
                # Asegurar que el directorio exista
                os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
                
                url = f'https://drive.google.com/uc?id={FILE_ID}'
                # quiet=False para ver progreso en logs si es necesario
                gdown.download(url, DB_PATH, quiet=False)
                st.success("Base de datos descargada correctamente.")
            except Exception as e:
                st.error(f"Error descargando la base de datos: {e}")
                # Intentar limpiar si falló
                if os.path.exists(DB_PATH):
                    os.remove(DB_PATH)
    else:
        print("Base de datos encontrada localmente.")

# Ejecutar descarga antes de cualquier otra cosa
download_db_if_missing()

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Asistente Jurídico IA",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Asistente Jurídico & Buscador de Jurisprudencia")

# =====================================
# SESSION STATE
# =====================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection" not in st.session_state:
    with st.spinner("Cargando base de datos de jurisprudencia..."):
        try:
            st.session_state.collection = rag_engine.load_collection()
            st.success("Base de datos cargada correctamente.")
        except Exception as e:
            st.error(f"Error cargando la base de datos: {e}")

# =====================================
# CHAT INTERFACE
# =====================================
# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input del usuario
if prompt := st.chat_input("Escribí tu consulta o pedido..."):
    # 1. Guardar y mostrar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Procesar con el motor RAG
    with st.chat_message("assistant"):
        with st.spinner("Analizando consulta..."):
            # Analizar intención
            analysis = rag_engine.analyze_query(prompt)
            intent = analysis.get("intent", "CHAT")
            filters = analysis.get("filters", {})
            search_q = analysis.get("search_query", prompt)
            
            response_text = ""

            if intent == "SEARCH":
                st.caption(f"🔍 **Modo Búsqueda detectado** | Filtros: {filters}")
                
                # Buscar
                results = rag_engine.search(
                    st.session_state.collection, 
                    query=search_q, 
                    filters=filters,
                    k=3
                )
                
                if results:
                    # Generar explicación
                    explanation = rag_engine.generate_rag_response(prompt, results)
                    response_text = explanation
                    
                    # Mostrar tarjetas de resultados (opcional, visualmente lindo)
                    st.markdown("### 📄 Fallos Encontrados")
                    for i, res in enumerate(results, 1):
                        meta = res["metadata"]
                        with st.expander(f"#{i} {meta.get('caratula', 'Sin Carátula')}"):
                            st.markdown(f"**Tribunal:** {meta.get('tribunal_principal', '-')}")
                            st.markdown(f"**Fecha:** {meta.get('fecha_sentencia', '-')}")
                            st.markdown(f"**Score:** {res.get('score', 0):.2f}")
                            st.text(res["texto"][:500] + "...")
                else:
                    response_text = "No encontré jurisprudencia que coincida con esos criterios específicos. ¿Querés probar con términos más generales?"
            
            else:
                # Modo CHAT
                # Preparamos historial para OpenAI (solo texto)
                chat_history = [
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state.messages
                ]
                response_text = rag_engine.generate_chat_response(chat_history)

            # Mostrar respuesta final
            st.markdown(response_text)
            
            # Guardar en historial
            st.session_state.messages.append({"role": "assistant", "content": response_text})
