# -*- coding: utf-8 -*-
"""
Ingesta CSV -> limpieza/normalización -> extracción de sumarios -> detección de secciones
-> chunking -> embeddings (OpenAI 1536) -> indexación en Postgres/pgvector.

Requiere:
  pip install pandas sqlalchemy psycopg2-binary openai unidecode python-slugify

Tablas esperadas (ya creadas):
  - juris.JURISPRUDENCES
  - juris.JURIS_CHUNKS (EMBEDDING vector(1536))

Autor: Juriprudencias
"""

import os
import csv
import re
import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from unidecode import unidecode

# ========= CONFIG =========
CSV_PATH = "fallos_combinados.csv"  # ← ruta a tu CSV
PG_DSN   = "postgresql+psycopg2://postgres:1a2b.3c4d@localhost:5433/jurisprudencias"

# OpenAI API key:
# Opción A (recomendada): leer de variable de entorno OPENAI_API_KEY
OPENAI_API_KEY = "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"
# Opción B (rápida local): descomentar y pegar tu clave (NO subir a GitHub con esto habilitado)
# OPENAI_API_KEY = "TU_API_KEY_AQUI"

EMBED_MODEL = "text-embedding-3-small"  # 1536 dimensiones
CHARS_PER_CHUNK = 1200
CHARS_OVERLAP   = 120
# =========================

# OpenAI SDK (nuevo)
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Falta instalar la librería 'openai' (pip install openai)") from e

# ============ Utilitarios generales ============

def read_csv_robusto(path: str) -> pd.DataFrame:
    """Lectura robusta de CSV (BOM, comillas internas, filas largas)."""
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        sep=",",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        on_bad_lines="skip",
        dtype=str
    )

def normalize_ws(x: str) -> str:
    """Normaliza espacios consecutivos y recorta extremos."""
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()

def parse_date(s: str) -> Optional[datetime]:
    """Convierte fecha en formatos comunes (DD/MM/YYYY, YYYY-MM-DD, etc.)."""
    if pd.isna(s) or not str(s).strip():
        return None
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except:
            pass
    return None

# ============ Detección de sumarios embebidos ============

_SUMARIO_PAT = re.compile(
    r"(?:^|\s)(SUMARIO\s+[^:\n]*:\s*)(.+?)(?=(?:\s+SUMARIO\s+|$))",
    flags=re.IGNORECASE | re.DOTALL
)

def extract_sumarios_from_text(texto: str) -> List[str]:
    """
    Devuelve lista de bloques 'SUMARIO ...' encontrados en `texto`.
    Soporta múltiples SUMARIO … SUMARIO … consecutivos.
    """
    if not texto:
        return []
    out = []
    for m in _SUMARIO_PAT.finditer(texto):
        out.append(normalize_ws(m.group(2)))
    # De-duplicar conservando orden
    seen = set()
    uniq = []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq

def choose_ratio_prefer_sumario(explicit_sumario: str, texto: str) -> str:
    """
    Prioriza `sumario` explícito; si no hay, toma el primer SUMARIO embebido.
    Si tampoco hay, usa primer párrafo sustantivo del texto.
    """
    if explicit_sumario and explicit_sumario.strip():
        return normalize_ws(explicit_sumario)
    emb = extract_sumarios_from_text(texto or "")
    if emb:
        return emb[0]
    paras = [p.strip() for p in re.split(r"\n{2,}|\r{2,}", texto or "") if p.strip()]
    if paras:
        return normalize_ws(paras[0] if len(paras[0]) >= 120 or len(paras) == 1 else paras[1])
    return ""

# ============ Detección de secciones (heurística ampliada) ============

def detect_sections(full_text: str) -> Dict[str, Tuple[int, int]]:
    """
    Heurística ampliada:
      - HECHOS: HECHOS / VISTOS / RESULTA(NDO)
      - CONSIDERANDOS: (Y) CONSIDERANDO(S)
      - PARTE_RESOLUTIVA: RESUELVO / RESUELVE / SE RESUELVE / POR ELLO / POR TANTO
      - (opcional) VOTO
    Devuelve dict {seccion: (start, end)} con offsets sobre el texto original.
    """
    if not full_text:
        return {}
    t = full_text
    low = unidecode(t.lower())

    markers = {
        "HECHOS": [
            r"\bhechos\b", r"\bvistos\b", r"\bresulta\b", r"\bresultando\b"
        ],
        "CONSIDERANDOS": [
            r"\by\s+considerando\b", r"\bconsiderando\b", r"\bconsiderandos\b"
        ],
        "PARTE_RESOLUTIVA": [
            r"\bresuelvo\b", r"\bresuelve\b", r"\bse\s+resuelve\b",
            r"\bpor\s+ello\b", r"\bpor\s+tanto\b"
        ],
        "VOTO": [r"\bvoto\b", r"\bmi\s+voto\b"]
    }

    found = {}
    for sec, pats in markers.items():
        for pat in pats:
            m = re.search(pat, low)
            if m:
                found[sec] = m.start()
                break

    if not found:
        return {}
    ordered = sorted(found.items(), key=lambda kv: kv[1])

    spans: Dict[str, Tuple[int, int]] = {}
    for i, (sec, start) in enumerate(ordered):
        end = len(t) if i == len(ordered) - 1 else ordered[i+1][1]
        spans[sec] = (start, end)
    return spans

# ============ Normalización de metadatos ============

_FUERO_RULES = [
    (r"camara\s+del\s+trabajo|laboral|cpl|cpt", "LABORAL"),
    (r"civil|comercial|locaciones|documentos", "CIVIL-COMERCIAL"),
    (r"penal|colegio\s+de\s+jueces|tribunal\s+de\s+impugnacion", "PENAL"),
    (r"contencioso\s+administrativo|tributario", "CONTENCIOSO-ADM"),
]

def norm_fuero(tribunal_principal: str, tipo_causa: str) -> str:
    base = unidecode(f"{tribunal_principal} {tipo_causa}".lower())
    for pat, fuero in _FUERO_RULES:
        if re.search(pat, base):
            return fuero
    return "GENERAL"

def norm_jurisdic(tribunal_principal: str) -> str:
    b = unidecode(tribunal_principal.lower())
    if "tucuman" in b:
        return "TUCUMAN"
    return "DESCONOCIDA"

def norm_materia(fuero: str, tipo_causa: str, sumario: str) -> str:
    base = unidecode(f"{fuero} {tipo_causa} {sumario}".lower())
    keys = [
        ("VIOLENCIA DE GENERO", "VIOLENCIA DE GENERO"),
        ("DAÑOS Y PERJUICIOS", "DAÑOS Y PERJUICIOS"),
        ("CONTRATO|PLAN DE AHORRO|AHORRO PREVIO", "CONTRATOS / CONSUMO"),
        ("ANATOCISMO|INTERESES", "INTERESES / BANCO"),
        ("ACCIDENTE IN ITINERE|LRT|ART", "RIESGOS DEL TRABAJO"),
        ("CASAS PARTICULARES|SERVICIO DOMESTICO", "CASAS PARTICULARES"),
        ("RESPONSABILIDAD SOLIDARIA|LCT 31", "RESP. SOLIDARIA"),
        ("TRANSFERENCIA DE ESTABLECIMIENTO", "TRANSFERENCIA ESTABLEC."),
        ("DESPIDO|INJURIA|LCT", "DESPIDO"),
        ("PRESCRIPCION|RECURSO DE CASACION|QUEJA", "PROCESAL"),
        ("ABUSO SEXUAL|INTEGRIDAD SEXUAL|ART\. 119", "PENAL - INTEGRIDAD SEX."),
        ("HOMICIDIO|ROBO|ARMAS|189 BIS", "PENAL - VIDA/PATRIMONIO"),
        ("TRIBUTARIO|ALICUOTA|DGR", "TRIBUTARIO"),
        ("NULIDAD|CAUTELAR|NO INNOVAR", "PROCESAL/CAUTELAR"),
    ]
    for pat, lab in keys:
        if re.search(pat.lower(), base):
            return lab
    # fallback por fuero
    return fuero if fuero != "GENERAL" else "GENERAL"

# ============ Estandarizar DF desde tu CSV ============

def load_standardize(csv_path: str) -> pd.DataFrame:
    """
    Espera columnas del CSV:
    tribunal_principal, tribunal_sala, caratula, tipo_causa, nro_expediente,
    nro_sentencia, fecha_sentencia, registro, sumario, texto
    """
    raw = read_csv_robusto(csv_path)
    for col in raw.columns:
        if raw[col].dtype == object:
            raw[col] = raw[col].apply(normalize_ws)

    std = pd.DataFrame({
        "TITLE":          raw.get("caratula", ""),
        "DOCKET_NUMBER":  raw.get("nro_expediente", ""),
        "DECISION_NUMBER":raw.get("nro_sentencia", ""),
        "DECISION_DATE":  raw.get("fecha_sentencia", "").apply(parse_date),
        "REGISTER_CODE":  raw.get("registro", ""),
        "COURT":          raw.get("tribunal_principal", ""),
        "CHAMBER":        raw.get("tribunal_sala", ""),
        "CASE_TYPE":      raw.get("tipo_causa", ""),
        "CARATULA":       raw.get("caratula", ""),
        "SUMMARY":        raw.get("sumario", ""),
        "FULL_TEXT":      raw.get("texto", "")
    })

    std["JURISDICTION"] = std["COURT"].apply(norm_jurisdic)
    std["FUERO"]        = std.apply(lambda r: norm_fuero(r["COURT"], r["CASE_TYPE"]), axis=1)
    std["MATTER"]       = std.apply(lambda r: norm_materia(r["FUERO"], r["CASE_TYPE"], r["SUMMARY"]), axis=1)
    std["RATIO"]        = std.apply(lambda r: choose_ratio_prefer_sumario(r["SUMMARY"], r["FULL_TEXT"]), axis=1)
    std["EXTRA_SUMARIOS"] = std["FULL_TEXT"].apply(extract_sumarios_from_text)
    return std

# ============ Chunking ============

def generic_chunk(text: str, max_chars: int = CHARS_PER_CHUNK, overlap: int = CHARS_OVERLAP) -> List[str]:
    """Chunking por caracteres con solapamiento simple."""
    t = normalize_ws(text or "")
    if not t:
        return []
    chunks, start, n = [], 0, len(t)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(t[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def build_pairs_for_chunking(row) -> List[Tuple[str, str]]:
    """
    Construye pares (SECTION, CONTENT_BASE) por documento:
      1) SUMARIO_n (si hay SUMARIOs extra)
      2) Secciones por heurística (HECHOS, CONSIDERANDOS, PARTE_RESOLUTIVA, VOTO)
      3) Si no hay secciones, un bloque GENERICO
    """
    pairs: List[Tuple[str, str]] = []
    # 1) SUMARIOS extra
    for i, s in enumerate(row.get("EXTRA_SUMARIOS", []) or []):
        pairs.append((f"SUMARIO_{i+1}", s))

    # 2) Secciones
    full_text = row.get("FULL_TEXT", "") or ""
    spans = detect_sections(full_text)
    if spans:
        for sec, (a, b) in spans.items():
            section_text = full_text[a:b].strip()
            if section_text:
                pairs.append((sec, section_text))
    else:
        if full_text.strip():
            pairs.append(("GENERICO", full_text))
    return pairs

def explode_to_chunks(row) -> List[Tuple[str, int, str]]:
    """
    A partir de pares (SECTION, CONTENT_BASE), aplica chunking y devuelve
    lista de (section, chunk_index, content_chunk).
    """
    out: List[Tuple[str, int, str]] = []
    for section, base in build_pairs_for_chunking(row):
        parts = generic_chunk(base)
        for i, ch in enumerate(parts):
            out.append((section, i, ch))
    return out

# ============ Embeddings (OpenAI) ============

def openai_embed_batch(texts: List[str], client: OpenAI, model: str) -> List[List[float]]:
    """Embeddings en batch con OpenAI."""
    if not texts:
        return []
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

# ============ Inserción en Postgres ============

def insert_all(pg_dsn: str, docs_rows: List[Dict]):
    """
    Inserta:
      - Un registro en JURISPRUDENCES por documento
      - N registros en JURIS_CHUNKS por cada chunk (con EMBEDDING vector)
    """
    from sqlalchemy import text, bindparam
    import sqlalchemy as sa

    engine = create_engine(pg_dsn)

    ins_doc = text("""
        INSERT INTO juris.JURISPRUDENCES
          (TITLE, DOCKET_NUMBER, DECISION_NUMBER, DECISION_DATE, REGISTER_CODE,
           COURT, CHAMBER, JURISDICTION, FUERO, MATTER,
           CASE_TYPE, CARATULA, SUMMARY, RATIO, FULL_TEXT, SOURCE_URL)
        VALUES
          (:TITLE, :DOCKET_NUMBER, :DECISION_NUMBER, :DECISION_DATE, :REGISTER_CODE,
           :COURT, :CHAMBER, :JURISDICTION, :FUERO, :MATTER,
           :CASE_TYPE, :CARATULA, :SUMMARY, :RATIO, :FULL_TEXT, :SOURCE_URL)
        RETURNING ID;
    """)

    # ✅ ÚNICA definición válida de ins_chunk (no la sobrescribas después)
    ins_chunk = text("""
        INSERT INTO juris.JURIS_CHUNKS
          (DOC_ID, SECTION, CHUNK_INDEX, CONTENT, EMBEDDING)
        VALUES
          (:doc_id, :section, :chunk_index, :content, CAST(:emb AS vector));
    """).bindparams(
        bindparam("doc_id", type_=sa.BigInteger),
        bindparam("section", type_=sa.Text),
        bindparam("chunk_index", type_=sa.Integer),
        bindparam("content", type_=sa.Text),
        bindparam("emb", type_=sa.String),
    )

    total_docs = len(docs_rows)
    print(f"🗄️ Insertando {total_docs} documentos en Postgres/pgvector…")

    with engine.begin() as conn:
        for idx, item in enumerate(docs_rows, start=1):
            r = item["row"]
            # Insert doc
            doc_id = conn.execute(ins_doc, {
                "TITLE":           r.get("TITLE"),
                "DOCKET_NUMBER":   r.get("DOCKET_NUMBER"),
                "DECISION_NUMBER": r.get("DECISION_NUMBER"),
                "DECISION_DATE":   r.get("DECISION_DATE"),  # asegurate que sea date/ISO
                "REGISTER_CODE":   r.get("REGISTER_CODE"),
                "COURT":           r.get("COURT"),
                "CHAMBER":         r.get("CHAMBER"),
                "JURISDICTION":    r.get("JURISDICTION"),
                "FUERO":           r.get("FUERO"),
                "MATTER":          r.get("MATTER"),
                "CASE_TYPE":       r.get("CASE_TYPE"),
                "CARATULA":        r.get("CARATULA"),
                "SUMMARY":         r.get("SUMMARY"),
                "RATIO":           r.get("RATIO"),
                "FULL_TEXT":       r.get("FULL_TEXT"),
                "SOURCE_URL":      None,
            }).scalar()

            # Prep batch de chunks
            params = []
            for ch in item["chunks"]:
                emb = "[" + ",".join(f"{x:.6f}" for x in ch["embedding"]) + "]"
                params.append({
                    "doc_id":      doc_id,
                    "section":     ch["section"],
                    "chunk_index": ch["index"],
                    "content":     ch["content"],
                    "emb":         emb,
                })

            if params:
                conn.execute(ins_chunk, params)  # executemany

            if idx % 10 == 0 or idx == total_docs:
                print(f"✅ Docs insertados: {idx}/{total_docs}")

    print("🏁 Inserción terminada.")

# ============ Pipeline principal ============

def build_docs_with_chunks(std_df: pd.DataFrame, client: OpenAI) -> List[Dict]:
    """
    Para cada fila (doc):
      - Explota a chunks
      - Embeddear en batch
      - Devuelve estructura lista para insertar
    """
    docs_with_chunks: List[Dict] = []
    total = len(std_df)
    print(f"🔄 Procesando {total} documentos...")

    for idx, row in std_df.iterrows():
        triplets = explode_to_chunks(row)  # [(section, idx, content), ...]
        texts = [c for (_, _, c) in triplets]
        vecs = openai_embed_batch(texts, client, EMBED_MODEL) if texts else []

        chunks = []
        for (section, i, content), vec in zip(triplets, vecs):
            chunks.append({
                "section": section,
                "index":   i,
                "content": content,
                "embedding": vec
            })

        docs_with_chunks.append({"row": row, "chunks": chunks})

        # feedback cada 10 docs
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"✅ Procesados {idx+1}/{total} documentos")

    print("🏁 Finalizado build_docs_with_chunks")
    return docs_with_chunks

def main():
    # Valida API key
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Falta OPENAI_API_KEY. "
            "Cargala como variable de entorno o pega la clave en el bloque CONFIG (opción B)."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    print("📥 Leyendo CSV…")
    std = load_standardize(CSV_PATH)
    print(f"   → Registros leídos: {len(std)}")

    if len(std) == 0:
        print("No hay filas para procesar. Fin.")
        return

    print("🧩 Construyendo chunks + generando embeddings (OpenAI 1536)…")
    docs = build_docs_with_chunks(std, client)
    total_chunks = sum(len(d["chunks"]) for d in docs)
    print(f"   → Chunks generados: {total_chunks}")

    print("🗄️ Insertando en Postgres/pgvector…")
    insert_all(PG_DSN, docs)

    print("✅ Listo. ¡Índice cargado!")

if __name__ == "__main__":
    main()
