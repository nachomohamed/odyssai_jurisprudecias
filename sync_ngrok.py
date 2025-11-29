import requests
import json
import time
import subprocess
import os

NGROK_API = "http://localhost:4040/api/tunnels"
CONFIG_FILE = "server_config.json"

def get_ngrok_url():
    print("⏳ Esperando a que Ngrok inicie...")
    for _ in range(10):  # Intentar por 20 segundos
        try:
            response = requests.get(NGROK_API)
            data = response.json()
            tunnels = data.get('tunnels', [])
            if tunnels:
                public_url = tunnels[0]['public_url']
                # Asegurar https
                if public_url.startswith("http://"):
                    public_url = public_url.replace("http://", "https://")
                return public_url
        except Exception:
            pass
        time.sleep(2)
    return None

def update_config_and_push(url):
    host = url.replace("https://", "").replace("http://", "")
    port = "443"
    
    config = {
        "CHROMA_HOST": host,
        "CHROMA_PORT": port
    }
    
    # Guardar JSON
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✅ Configuración guardada: {config}")
    
    # Git commands
    try:
        print("🚀 Subiendo nueva URL a GitHub...")
        subprocess.run(["git", "add", CONFIG_FILE], check=True)
        subprocess.run(["git", "commit", "-m", f"Update Ngrok URL: {host}"], check=True)
        subprocess.run(["git", "push", "origin", "develop"], check=True)
        print("🎉 URL actualizada en GitHub correctamente.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error en Git (puede que no haya cambios): {e}")

if __name__ == "__main__":
    url = get_ngrok_url()
    if url:
        print(f"🔗 URL de Ngrok detectada: {url}")
        update_config_and_push(url)
    else:
        print("❌ No se pudo detectar la URL de Ngrok. Asegúrate de que esté corriendo.")
