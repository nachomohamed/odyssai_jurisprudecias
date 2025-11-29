import chromadb
import pandas as pd

CHROMA_DIR = "./chroma_juris"
DEFAULT_COLLECTION = "jurisprudencia"

def get_unique_metadata():
    print(f"Conectando a Chroma en {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        col = client.get_collection(name=DEFAULT_COLLECTION)
    except:
        cols = client.list_collections()
        if not cols:
            print("No collections found.")
            return
    print(f"Using collection: {cols[0].name}")

    # Dummy embedding function to avoid loading onnxruntime/sentence-transformers
    class DummyEmbeddingFunction:
        def __call__(self, input):
            return [[0.0]*384 for _ in input]

    print("Fetching all metadata...")
    try:
        col = client.get_collection(name=DEFAULT_COLLECTION, embedding_function=DummyEmbeddingFunction())
    except:
        # Fallback if name is different
        col = client.get_collection(name=cols[0].name, embedding_function=DummyEmbeddingFunction())
    metadatas = results['metadatas']
    
    print(f"Found {len(metadatas)} documents.")
    
    df = pd.DataFrame(metadatas)
    
    print("\n--- UNIQUE VALUES ---")
    for field in ['tribunal_principal', 'tribunal_sala', 'tipo_causa']:
        if field in df.columns:
            unique_vals = df[field].dropna().unique()
            print(f"\nField: {field} ({len(unique_vals)} unique):")
            # Print top 50 to avoid spamming if there are too many
            print(list(unique_vals)[:50])
        else:
            print(f"\nField {field} not found in metadata.")

if __name__ == "__main__":
    get_unique_metadata()
