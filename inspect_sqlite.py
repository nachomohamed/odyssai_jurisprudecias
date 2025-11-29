import sqlite3
import json

DB_PATH = "chroma_juris/chroma.sqlite3"

def inspect_sqlite():
    print(f"Connecting to SQLite at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # List tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", tables)
    
    # Try to find metadata table. Usually 'embeddings' or 'embedding_metadata'
    # In newer Chroma versions, metadata might be in 'embeddings' table in a 'metadata' column (JSON or string)
    # or in a separate table.
    
    # Let's check columns of 'embeddings' table if it exists
    table_names = [t[0] for t in tables]
    if 'embeddings' in table_names:
        cursor.execute("PRAGMA table_info(embeddings);")
        columns = cursor.fetchall()
        print("\nColumns in 'embeddings':", [c[1] for c in columns])
        
        # Try to fetch some metadata
        # It might be in a column named 'metadata' or similar
        # Or maybe there is a 'embedding_metadata' table?
        
    if 'embedding_metadata' in table_names:
         print("\nFound 'embedding_metadata' table. Checking columns...")
         cursor.execute("PRAGMA table_info(embedding_metadata);")
         columns = cursor.fetchall()
         print([c[1] for c in columns])
         
         # Fetch unique values
         # This depends on how metadata is stored. 
         # If it's key-value rows: key, value, id
         # If it's JSON string: we need to parse it.
         
         cursor.execute("SELECT * FROM embedding_metadata LIMIT 5")
         rows = cursor.fetchall()
         print("\nSample rows from embedding_metadata:", rows)

    conn.close()

if __name__ == "__main__":
    inspect_sqlite()
