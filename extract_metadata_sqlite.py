import sqlite3

DB_PATH = "chroma_juris/chroma.sqlite3"

def get_unique_values():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    fields = ['tribunal_principal', 'tribunal_sala', 'tipo_causa', 'keywords', 'tags_json']
    
    for field in fields:
        print(f"\n--- {field} ---")
        query = f"SELECT DISTINCT string_value FROM embedding_metadata WHERE key = '{field}' ORDER BY string_value"
        cursor.execute(query)
        rows = cursor.fetchall()
        values = [r[0] for r in rows if r[0]] # Filter None or empty
        
        print(f"Count: {len(values)}")
        if field == 'tipo_causa':
            print("First 50:", values[:50])
        else:
            print(values)

    conn.close()

if __name__ == "__main__":
    get_unique_values()
