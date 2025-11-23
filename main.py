import streamlit as st
import os
import utils

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Asistente Jurídico IA",
    page_icon="⚖️",
    layout="centered"
)

# =====================================
# INITIALIZATION
# =====================================
# Ejecutar descarga y extracción de metadatos antes de cargar el motor RAG
utils.initialize_app()

import rag_engine

st.title("⚖️ Asistente Jurídico & Buscador de Jurisprudencia")

# =====================================
# SESSION STATE MANAGEMENT
# =====================================
import uuid

if "chats" not in st.session_state:
    # Estructura: { "chat_id": { "title": "...", "messages": [] } }
    default_id = str(uuid.uuid4())
    st.session_state.chats = {
        default_id: {"title": "Nueva Conversación", "messages": []}
    }
    st.session_state.current_chat_id = default_id

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]

if "collection" not in st.session_state:
    with st.spinner("Cargando base de datos de jurisprudencia..."):
        try:
            st.session_state.collection = rag_engine.load_collection()
            st.success("Base de datos cargada correctamente.")
        except Exception as e:
            st.error(f"Error cargando la base de datos: {e}")

# =====================================
# SIDEBAR: CHAT MANAGEMENT
# =====================================
with st.sidebar:
    st.title("🗂️ Historial")
    
    # Botón Nueva Conversación
    if st.button("➕ Nueva Conversación", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "Nueva Conversación", "messages": []}
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.divider()

    # Lista de Conversaciones
    # Ordenar por creación (aunque dict no garantiza orden en versiones viejas, en 3.7+ sí)
    # Lo ideal sería guardar timestamp, pero simplificamos iterando keys.
    chat_ids = list(st.session_state.chats.keys())
    
    # Usamos radio button para seleccionar (es lo más limpio en Streamlit nativo)
    # Mapeamos ID -> Título para mostrar
    options = chat_ids
    format_func = lambda x: st.session_state.chats[x]["title"]
    
    selected_id = st.radio(
        "Tus Chats:",
        options=options,
        format_func=format_func,
        index=options.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in options else 0,
        label_visibility="collapsed"
    )
    
    # Actualizar selección si cambió
    if selected_id != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_id
        st.rerun()

    st.divider()
    
    # Botón Eliminar
    if st.button("🗑️ Eliminar Conversación Actual", type="primary", use_container_width=True):
        if len(st.session_state.chats) > 1:
            del st.session_state.chats[st.session_state.current_chat_id]
            # Seleccionar otro
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
            st.rerun()
        else:
            st.warning("No puedes eliminar la única conversación activa.")

# =====================================
# MAIN CHAT INTERFACE
# =====================================
current_chat = st.session_state.chats[st.session_state.current_chat_id]

# Mostrar historial del chat actual
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input del usuario
if prompt := st.chat_input("Escribí tu consulta o pedido..."):
    # 1. Guardar y mostrar mensaje usuario
    current_chat["messages"].append({"role": "user", "content": prompt})
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
                    for m in current_chat["messages"]
                ]
                response_text = rag_engine.generate_chat_response(chat_history)

            # Mostrar respuesta final
            st.markdown(response_text)
            
            # Guardar en historial
            current_chat["messages"].append({"role": "assistant", "content": response_text})

    # Actualizar título si es el primer mensaje (y recargar para mostrarlo en sidebar)
    if len(current_chat["messages"]) == 2: # 1 user + 1 assistant
        # Usar primeras 5 palabras como título
        title = " ".join(prompt.split()[:5]) + "..."
        current_chat["title"] = title
        st.rerun()
