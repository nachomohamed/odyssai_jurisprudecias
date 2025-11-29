# chat_rag.py
import os
import sys
import chromadb
from chromadb.utils import embedding_functions

# === Config básica ===
CHROMA_DIR = "./chroma_juris"
DEFAULT_COLLECTION = "juris"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Debe matchear el usado al indexar

# (Opcional) Cross-Encoder para reranking
USE_RERANK = True
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# (Opcional) LLM vía OPENAI_API_KEY
USE_OPENAI = True
OPENAI_MODEL = "gpt-4o-mini"  # barato/rápido para MVP

# === Utilidades ===
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Si no sabés el nombre exacto de la colección, intentamos DEFAULT y si no existe tomamos la primera.
    try:
        col = client.get_collection(name=DEFAULT_COLLECTION)
    except Exception:
        cols = client.list_collections()
        if not cols:
            print("❌ No hay colecciones en", CHROMA_DIR)
            sys.exit(1)
        col = client.get_collection(name=cols[0].name)
        print(f"ℹ️ Usando colección detectada: {cols[0].name}")

    return col

def build_query_filters(tribunal=None, sala=None, tipo_causa=None, anio=None):
    """
    Arma un dict de 'where' para Chroma usando metadata.
    Usá solo lo que tengas; si no pasás nada, vuelve None.
    """
    where = {}
    if tribunal:
        where["tribunal_principal"] = tribunal
    if sala:
        where["tribunal_sala"] = sala
    if tipo_causa:
        where["tipo_causa"] = tipo_causa
    if anio:
        # si en tu CSV 'fecha_sentencia' es 'YYYY-MM-DD', podemos filtrar por prefijo con $contains o by range si guardaste 'anio'
        where["fecha_sentencia"] = {"$contains": str(anio)}
    return where or None

def rerank(query, docs):
    """
    Reordenamiento opcional con CrossEncoder.
    docs: lista de strings
    """
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return docs  # si no está instalado, seguimos sin rerank

    model = CrossEncoder(CROSS_ENCODER_NAME)
    pairs = [[query, d] for d in docs]
    scores = model.predict(pairs)
    # Ordenamos por score desc
    ranked = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return ranked

def llm_answer(history, retrieved):
    """
    Genera respuesta con o sin OpenAI (fallback: respuesta basada en snippets).
    """
    # Construimos prompt sencillo
    citations = []
    context_blocks = []
    for i, r in enumerate(retrieved, 1):
        meta = r.get("metadata", {})
        tag = f"[{i}] {meta.get('tribunal_principal','?')} | {meta.get('caratula','?')} | {meta.get('fecha_sentencia','?')}"
        snippet = r.get("texto","")[:1000]  # recorte para el prompt
        context_blocks.append(f"{tag}\n{snippet}")
        citations.append(tag)

    system = (
        "Sos un asistente legal que contesta con precisión y cita jurisprudencias recuperadas entre corchetes.\n"
        "Si no hay base suficiente en los fallos recuperados, pedí más detalles o proponé nuevas búsquedas."
    )
    user = history[-1]["user"]

    context = "\n\n".join(context_blocks) if context_blocks else "No se recuperó jurisprudencia relevante."
    prompt = f"""{system}

Contexto recuperado:
{context}

Pregunta del usuario:
{user}

Instrucciones:
- Responder de forma breve y clara.
- Citar casos usando [n] según el listado de arriba.
- Si el usuario pide un escrito, dar un borrador inicial con estructura.

Respuesta:
"""
    if USE_OPENAI and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role":"system","content":"Eres un asistente experto en derecho argentino."},
                    {"role":"user","content":prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"(LLM no disponible: {e})\n\nContexto:\n" + context
    else:
        # Fallback sin LLM: devolvemos un sumario muy básico
        base = "No tengo LLM activo; te dejo los fragmentos más relevantes:\n\n"
        for i, r in enumerate(retrieved, 1):
            meta = r.get("metadata", {})
            base += f"[{i}] {meta.get('tribunal_principal','?')} | {meta.get('caratula','?')} | {meta.get('fecha_sentencia','?')}\n"
            base += r.get("texto","")[:500] + "\n\n"
        return base.strip()

def search(col, query, k=5, tribunal=None, sala=None, tipo_causa=None, anio=None):
    where = build_query_filters(tribunal, sala, tipo_causa, anio)

    results = col.query(
        query_texts=[query],
        n_results=k*3 if USE_RERANK else k,
        where=where
    )

    # Normalizamos salida
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    items = []
    for doc, meta, _id in zip(docs, metadatas, ids):
        items.append({
            "id": _id,
            "texto": doc,
            "metadata": meta
        })

    # Reranking opcional
    if USE_RERANK and items:
        ranked_texts = rerank(query, [x["texto"] for x in items])
        # mapear al top-k final manteniendo metadata acorde al texto
        ranked_items = []
        used = set()
        for txt in ranked_texts:
            for it in items:
                if it["texto"] == txt and id(it) not in used:
                    ranked_items.append(it)
                    used.add(id(it))
                    break
        items = ranked_items[:k]
    else:
        items = items[:k]

    return items

def main():
    # Embeddings para que Chroma los calcule si hiciera falta en runtime
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    col = load_collection()
    # Vinculamos emb fn por si la colección no la trae embebida
    try:
        col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(
            name=col.name,
            embedding_function=emb_fn
        )
    except Exception:
        pass

    print("🟢 Chat RAG listo. Escribí tu consulta legal.")
    print("   Podés usar filtros con líneas tipo: tribunal:CAMARA CIVIL; sala:SALA II; anio:2022")
    print("   Para salir: /exit\n")

    history = []

    while True:
        user_in = input("Tú> ").strip()
        if not user_in:
            continue
        if user_in.lower() == "/exit":
            break

        # parse filtros en la misma línea (formato clave:valor; clave:valor)
        tribunal = sala = tipo_causa = anio = None
        if ";" in user_in or "tribunal:" in user_in or "sala:" in user_in or "anio:" in user_in or "tipo_causa:" in user_in:
            # separamos filtros al final si vienen tipo "consulta ... tribunal:XXX; sala:YYY"
            parts = user_in.split()
            kv = [p for p in parts if ":" in p]
            qparts = [p for p in parts if ":" not in p]
            for pair in kv:
                key, val = pair.split(":", 1)
                val = val.strip(";")
                key = key.lower()
                if key == "tribunal":
                    tribunal = val
                elif key == "sala":
                    sala = val
                elif key == "tipo_causa":
                    tipo_causa = val
                elif key == "anio":
                    anio = val
            query = " ".join(qparts).strip()
            if not query:
                query = " ".join([p for p in parts if ":" not in p])
        else:
            query = user_in

        history.append({"user": query})

        hits = search(
            col,
            query=query,
            k=5,
            tribunal=tribunal,
            sala=sala,
            tipo_causa=tipo_causa,
            anio=anio
        )

        answer = llm_answer(history, hits)
        print("\nAssistant>\n" + answer + "\n")

        # Mostrar una tablita mínima de los hits para navegar
        if hits:
            print("Resultados:")
            for i, h in enumerate(hits, 1):
                m = h["metadata"] or {}
                print(f" [{i}] id={h['id']} | {m.get('tribunal_principal','?')} | {m.get('caratula','?')} | {m.get('fecha_sentencia','?')}")
            print("")