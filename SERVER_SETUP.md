# 🚀 Cómo levantar tu Servidor ChromaDB (Opción 2)

Para que la app de Streamlit (que corre en la nube) pueda acceder a tu base de datos (que está en tu PC o servidor), necesitas ejecutar Chroma en modo **Servidor**.

## Paso 1: Ejecutar Chroma en tu PC/Servidor

Abre una terminal en la carpeta de tu proyecto (donde está la carpeta `chroma_juris`) y ejecuta:

```bash
chroma run --path chroma_juris --port 8000
```

Esto levantará el servidor en `localhost:8000`.

## Paso 2: Exponer tu servidor a Internet (Si usas tu PC local)

Como Streamlit Cloud no puede ver tu `localhost`, necesitas crear un túnel. La forma más fácil es usar **Ngrok**.

1.  Descarga e instala [Ngrok](https://ngrok.com/).
2.  Ejecuta en otra terminal:
    ```bash
    ngrok http 8000
    ```
3.  Ngrok te dará una URL pública, algo como: `https://a1b2-c3d4.ngrok-free.app`. **Esa es tu `CHROMA_HOST`**.

## Paso 3: Configurar Streamlit Cloud

Ve a tu panel de Streamlit Cloud -> App Settings -> **Secrets** y agrega:

```toml
CHROMA_HOST = "tu-url-de-ngrok.ngrok-free.app"
CHROMA_PORT = "80" 
# Nota: Si usas ngrok https, el puerto es 443 o 80, no 8000. 
# La librería client de chroma a veces es quisquillosa con https/http.
# Si usas VPS directo con IP pública, usa el puerto 8000.
```

### ⚠️ Importante sobre Ngrok y ChromaClient
El cliente de Python de Chroma (`HttpClient`) espera conectarse a un host y puerto.
Si usas Ngrok, la URL es `https://...`.
A veces es más fácil configurar:
`CHROMA_HOST = "a1b2-c3d4.ngrok-free.app"` (sin https://)
`CHROMA_PORT = "443"` (si es https)

## Paso 4: Generar `metadata_values.json`

Como ya no extraemos metadatos en vivo (porque no tenemos el archivo local), asegúrate de generar el archivo `metadata_values.json` en tu máquina local y **subirlo a GitHub**.

Ejecuta localmente una vez:
```bash
python utils.py
```
(O descomenta temporalmente la función en `utils.py` para correrla localmente).

Una vez que tengas `metadata_values.json`, haz commit y push.
