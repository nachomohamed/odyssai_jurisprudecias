import os
import uuid
import math
import pandas as pd
from typing import Dict, Iterable, Tuple, List, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# ---------------- Configuración ----------------
CSV_PATH = None  # si lo dejás en None, te lo pide por input
PERSIST_DIR = "./chroma_juris"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
MAX_CHARS = 1200
OVERLAP_CHARS = 150

# Ingesta streaming
READ_CHUNK_ROWS = 1000
ADD_BATCH = 1024

EXPECTED_COLS = [
    "tribunal_principal","tribunal_sala","caratula","tipo_causa",
    "nro_expediente","nro_sentencia","fecha_sentencia","registro",
    "sumario","texto"
]

# --------------- Utilidades --------------------

def safe_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()

def safe_int_year(val) -> Optional[int]:
    s = safe_str(val)
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None

def sanitize_metadata(meta: Dict) -> Dict:
    """
    Chroma exige metadatos tipo JSON-serializable escalar (str/int/float/bool).
    Reemplazamos None con "" o eliminamos claves vacías si preferís.
    """
    clean = {}
    for k, v in meta.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            # estrategia: convertir a "" (o podrías omitir la clave)
            clean[k] = "" if k not in ("anio",) else -1
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = safe_str(v)
    return clean

def soft_split_stream(text: str, max_chars: int, overlap: int) -> Iterable[Tuple[int, int, str]]:
    """
    Generador que emite (start, end, chunk_text) para no acumular en memoria.
    Corta preferentemente en fin de oración.
    """
    if not isinstance(text, str):
        return
    text = text.strip()
    if not text:
        return
    n = len(text)
    start = 0
    # límites de seguridad
    max_iters = 1 + math.ceil(n / max(1, (max_chars - overlap)))
    iters = 0

    while start < n and iters < max_iters + 5:
        iters += 1
        end = min(start + max_chars, n)
        window = text[start:end]
        # tratar de cortar en un punto agradable
        last_dot = window.rfind(". ")
        if last_dot > max_chars * 0.5:
            end = start + last_dot + 1

        chunk = text[start:end].strip()
        if chunk:
            yield start, end, chunk

        # avanzar con overlap controlado
        if end <= start:
            # fallback de seguridad para evitar loops
            start = min(n, start + max(1, max_chars))
        else:
            start = max(0, end - overlap)

def build_metadata(row: pd.Series, chunk_id: int) -> Dict:
    meta = {
        "tribunal_principal": safe_str(row.get("tribunal_principal", "")),
        "tribunal_sala": safe_str(row.get("tribunal_sala", "")),
        "caratula": safe_str(row.get("caratula", "")),
        "tipo_causa": safe_str(row.get("tipo_causa", "")),
        "materia": safe_str(row.get("materia", "")) if "materia" in row else "",
        "nro_expediente": safe_str(row.get("nro_expediente", "")),
        "nro_sentencia": safe_str(row.get("nro_sentencia", "")),
        "fecha_sentencia": safe_str(row.get("fecha_sentencia", "")),
        "registro": safe_str(row.get("registro", "")),
        "anio": safe_int_year(row.get("fecha_sentencia", "")) if safe_int_year(row.get("fecha_sentencia", "")) is not None else -1,
        "sumario": safe_str(row.get("sumario", "")),
        "chunk_id": int(chunk_id),
    }
    return sanitize_metadata(meta)

def ensure_expected_columns(df_cols: List[str]):
    missing = [c for c in EXPECTED_COLS if c not in df_cols]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

# --------------- Proceso principal ----------------

def main():
    global CSV_PATH
    if CSV_PATH is None:
        CSV_PATH = input("Ruta al CSV de fallos: ").strip()
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No existe: {CSV_PATH}")

    os.makedirs(PERSIST_DIR, exist_ok=True)

    # Embeddings locales (sin costo)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(EMB_MODEL)

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        col = client.get_collection("jurisprudencia")
    except:
        col = client.create_collection("jurisprudencia", embedding_function=emb_fn)

    # lectura en streaming
    total_rows = 0
    buffer_ids, buffer_docs, buffer_meta = [], [], []

    for df in pd.read_csv(CSV_PATH, chunksize=READ_CHUNK_ROWS):
        ensure_expected_columns(df.columns.tolist())

        for _, row in df.iterrows():
            total_rows += 1

            base_text = " ".join([
                safe_str(row.get("sumario", "")),
                safe_str(row.get("texto", "")),
            ]).strip()

            if not base_text:
                continue

            chunk_id = 0
            for start, end, chunk_text in soft_split_stream(base_text, MAX_CHARS, OVERLAP_CHARS):
                meta = build_metadata(row, chunk_id)
                # ID único por fila+chunk
                uid = f"{meta.get('nro_sentencia','')}-{meta.get('nro_expediente','')}-{chunk_id}-{uuid.uuid4().hex[:8]}"
                buffer_ids.append(uid)
                buffer_docs.append(chunk_text)
                buffer_meta.append(meta)
                chunk_id += 1

                if len(buffer_ids) >= ADD_BATCH:
                    col.add(ids=buffer_ids, documents=buffer_docs, metadatas=buffer_meta)
                    buffer_ids, buffer_docs, buffer_meta = [], [], []

        # flush por bloque del CSV (opcional pero sano)
        if buffer_ids:
            col.add(ids=buffer_ids, documents=buffer_docs, metadatas=buffer_meta)
            buffer_ids, buffer_docs, buffer_meta = [], [], []

        print(f"Ingeridas {total_rows} filas...")

    # flush final
    if buffer_ids:
        col.add(ids=buffer_ids, documents=buffer_docs, metadatas=buffer_meta)

    print("✅ Índice construido en:", PERSIST_DIR)

if __name__ == "__main__":
    # Sugerencia: silenciar telemetría de Chroma si molesta
    os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
    main()
