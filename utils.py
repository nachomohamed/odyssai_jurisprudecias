import os
import json
import sqlite3
import streamlit as st
import gdown

# =====================================
# CONFIG
# =====================================
FILE_ID = '1V4BVNytXGNBmjprpmC0xPmFeQnH3v0hl'
DB_PATH = 'chroma_juris/chroma.sqlite3'
METADATA_JSON_PATH = 'metadata_values.json'

def download_db_if_missing():
    """Descarga la base de datos desde Google Drive si no existe localmente."""
    if not os.path.exists(DB_PATH):
        with st.spinner("Descargando base de datos de jurisprudencia (3.3GB)... esto puede tardar unos minutos."):
            try:
                # Asegurar que el directorio exista
                os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
                
                url = f'https://drive.google.com/uc?id={FILE_ID}'
                gdown.download(url, DB_PATH, quiet=False)
                st.success("Base de datos descargada correctamente.")
            except Exception as e:
                st.error(f"Error descargando la base de datos: {e}")
                if os.path.exists(DB_PATH):
                    os.remove(DB_PATH)
    else:
        print("Base de datos encontrada localmente.")

def extract_metadata_to_json():
    """Extrae valores únicos de metadatos de SQLite y los guarda en un JSON."""
    if not os.path.exists(DB_PATH):
        print("No se puede extraer metadatos: No existe la DB.")
        return

    print("Extrayendo metadatos de la base de datos...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        metadata = {}
        fields_map = {
            'tribunal_principal': 'tribunales',
            'tribunal_sala': 'salas'
        }
        
        for db_field, json_key in fields_map.items():
            query = f"SELECT DISTINCT string_value FROM embedding_metadata WHERE key = '{db_field}' ORDER BY string_value"
            cursor.execute(query)
            rows = cursor.fetchall()
            values = [r[0] for r in rows if r[0]]
            metadata[json_key] = values
            
        conn.close()
        
        with open(METADATA_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
            
        print(f"Metadatos guardados en {METADATA_JSON_PATH}")
        
    except Exception as e:
        print(f"Error extrayendo metadatos: {e}")

@st.cache_resource
def initialize_app():
    """Ejecuta tareas de inicialización una sola vez."""
    download_db_if_missing()
    extract_metadata_to_json()
