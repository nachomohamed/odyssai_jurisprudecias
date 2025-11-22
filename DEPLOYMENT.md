# Guía de Despliegue (Deployment Guide)

Debido a que la base de datos vectorial (`chroma.sqlite3`) pesa más de 3GB, **no se puede subir a GitHub**. GitHub tiene un límite estricto de 100MB por archivo.

Para desplegar esta aplicación en Streamlit Cloud (o cualquier otro servidor), sigue estos pasos:

## 1. Ignorar Archivos Locales
El archivo `.gitignore` ya ha sido configurado para ignorar:
- `chroma_juris/`
- `*.sqlite3`
- `*.csv` (si son muy grandes)

Esto asegura que `git push` funcione rápido y no falle por archivos gigantes.

## 2. Subir la Base de Datos a la Nube
**YA COMPLETADO.**
Tu archivo ya está en Google Drive con el ID: `10J4VsnOel0Njd_mkUsJZ9kMqo8JQ3O1r`

## 3. Configurar la Descarga Automática en Streamlit
**YA COMPLETADO.**
He agregado el código necesario en `main.py` para que descargue automáticamente la base de datos usando `gdown`.

```python
# Este código ya está en tu main.py
FILE_ID = '10J4VsnOel0Njd_mkUsJZ9kMqo8JQ3O1r'
DB_PATH = 'chroma_juris/chroma.sqlite3'
# ... lógica de descarga ...
```

## 4. Requirements.txt
**YA COMPLETADO.**
He agregado `gdown` a tu archivo `requirements.txt`.

## 5. Advertencia sobre Recursos (RAM)
Streamlit Cloud tiene un límite de disco generoso (aprox. 50GB), por lo que **el espacio en disco NO debería ser un problema**.

Sin embargo, la **Memoria RAM** es limitada (aprox. 1GB - 3GB).
- Si al cargar la base de datos la aplicación se reinicia o muestra un error de "Memory Limit Exceeded", significará que ChromaDB necesita más RAM de la disponible.

### Solución Alternativa (si falla por RAM)
Si la aplicación falla por memoria, la solución es **no alojar la base de datos en Streamlit**, sino usar un servicio externo:
1.  **Chroma Client/Server**: Alojar Chroma en un servidor propio (AWS, Railway, etc.) y conectarse vía API.
2.  **Pinecone / Weaviate**: Usar una base de datos vectorial en la nube (SaaS) en lugar de un archivo local.
