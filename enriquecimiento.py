# -*- coding: utf-8 -*-
"""
Enriquece fallos desde un CSV generando keywords y tags vía LLM (OpenAI).
Salida: mismo CSV + columnas 'keywords' y 'tags_json'.

Requisitos:
  pip install pandas openai tenacity python-dotenv
"""

import os
import json
import time
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any, List, Optional

# ========= CONFIGURACIÓN RÁPIDA (EDITABLE) =========
OPENAI_API_KEY = ""
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")            # Sugerido: 'gpt-4o-mini' (costo/latencia)
BASE_URL       = os.getenv("OPENAI_BASE_URL", None)                  # Opcional (p.ej. LM Studio con compatibilidad)

INPUT_CSV  = "fallos_combinados.csv"          # <-- Tu CSV de entrada
OUTPUT_CSV = "fallos_combinados_enriquecido.csv"
BATCH_SIZE = 200                              # Procesamiento por lotes para no cargar todo en RAM
MAX_TOKENS = 800                              # Tope de salida por llamada (ajustá a tu gusto)

# Prompt maestro: ajustalo a tu criterio/estilo.
PROMPT_EXTRACCION = """
Eres un asistente jurídico que etiqueta jurisprudencia laboral argentina.

TAREA:
A partir de los campos `sumario` y `texto`, devolvé un JSON **válido** con:
- keywords: lista de 5 a 12 palabras o n-gramas cortos, en minúsculas, sin tildes, normalizadas, sin duplicados (ej: "abandono de trabajo", "injuria laboral", "despido con causa", "carga de la prueba").
- temas: 3 a 8 etiquetas temáticas de mayor nivel (ej: "despido", "abandono de trabajo", "relacion laboral no registrada", "prueba").
- figuras_juridicas: 0 a 6 si corresponde (ej: "abandono de trabajo", "pérdida de confianza", "injuria", "mora", "intercambio telegráfico").
- partes_relevantes: si se detectan en el contenido, lista con posibles nombres/razones sociales relevantes.
- nota: breve justificación (1 a 2 frases) del por qué de las keywords.

REGLAS:
- No inventes información; si no hay evidencia, deja el campo como lista vacía.
- No incluyas citas extensas del texto; sintetizá.
- El JSON debe ser autocontenido y estricto, sin texto adicional.
"""

# ========= CLIENTE OPENAI =========
from openai import OpenAI
client_kwargs = {}
if BASE_URL:
    client_kwargs["base_url"] = BASE_URL
client = OpenAI(api_key=OPENAI_API_KEY, **client_kwargs)

# ========= FUNCIÓN LLM =========
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def llamar_llm(sumario: str, cuerpo: str) -> Dict[str, Any]:
    """Llama al LLM y devuelve un dict con las claves esperadas."""
    content_user = (
        "Extrae etiquetas para jurisprudencia laboral argentina.\n\n"
        f"SUMARIO:\n{sumario or ''}\n\n"
        f"TEXTO:\n{cuerpo or ''}\n"
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": PROMPT_EXTRACCION.strip()},
            {"role": "user", "content": content_user},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=MAX_TOKENS,
    )

    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Reintento forzado con una instrucción dura de JSON
        resp2 = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Respondé EXCLUSIVAMENTE un JSON válido. Sin texto adicional."},
                {"role": "user", "content": content_user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=MAX_TOKENS,
        )
        data = json.loads(resp2.choices[0].message.content)

    # Normalización mínima
    data.setdefault("keywords", [])
    data.setdefault("temas", [])
    data.setdefault("figuras_juridicas", [])
    data.setdefault("partes_relevantes", [])
    data.setdefault("nota", "")

    # Limpiar keywords (lower + dedup + sin comillas)
    def _norm_list(vals: List[str]) -> List[str]:
        out, seen = [], set()
        for v in vals or []:
            k = (v or "").strip().strip('"').strip("'").lower()
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    data["keywords"] = _norm_list(data["keywords"])
    data["temas"] = _norm_list(data["temas"])
    data["figuras_juridicas"] = _norm_list(data["figuras_juridicas"])
    data["partes_relevantes"] = _norm_list(data["partes_relevantes"])

    return data

# ========= PROCESAMIENTO POR LOTES =========
def procesar_csv(input_csv: str, output_csv: str, batch_size: int = 200) -> None:
    print("⚙️ Iniciando procesamiento del CSV...")
    # Cargamos por chunks para memoria estable
    reader = pd.read_csv(input_csv, chunksize=batch_size)
    processed_chunks = []

    total_rows = 0
    for i, chunk in enumerate(reader, start=1):
        # Asegurar columnas esperadas
        for col in ["sumario", "texto"]:
            if col not in chunk.columns:
                raise ValueError(f"Falta la columna requerida: '{col}'")

        # Nuevas columnas
        chunk["keywords"] = ""
        chunk["tags_json"] = ""

        for idx, row in chunk.iterrows():
            sumario = str(row.get("sumario", "") or "")
            texto   = str(row.get("texto", "") or "")

            try:
                tags = llamar_llm(sumario, texto)
                # Guardamos keywords en un solo string separado por ';'
                kws = ";".join(tags.get("keywords", []))
                chunk.at[idx, "keywords"] = kws
                chunk.at[idx, "tags_json"] = json.dumps(tags, ensure_ascii=False)
            except Exception as e:
                # En caso de error, dejamos vacío para revisión posterior
                chunk.at[idx, "keywords"] = ""
                chunk.at[idx, "tags_json"] = json.dumps(
                    {"error": str(e), "keywords": [], "temas": [], "figuras_juridicas": [], "partes_relevantes": [], "nota": ""},
                    ensure_ascii=False
                )

        processed_chunks.append(chunk)
        total_rows += len(chunk)
        print(f"Lote {i} procesado. Filas acumuladas: {total_rows}")

        # Pequeño sleep defensivo para evitar rate limits duros
        time.sleep(0.5)

    # Concatenar y guardar
    df_out = pd.concat(processed_chunks, ignore_index=True)
    # Ordenar columnas: originales + nuevas al final
    cols = list(df_out.columns)
    # Asegurar que 'keywords' y 'tags_json' queden al final
    for c in ["keywords", "tags_json"]:
        if c in cols:
            cols.remove(c)
            cols.append(c)
    df_out = df_out[cols]
    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✅ Listo. Guardado en: {output_csv}")

if __name__ == "__main__":
    if not OPENAI_API_KEY or OPENAI_API_KEY == "PON_AQUI_TU_API_KEY":
        raise RuntimeError("Configura OPENAI_API_KEY (variable de entorno o arriba del script).")
    procesar_csv(INPUT_CSV, OUTPUT_CSV, BATCH_SIZE)
