# ⚖️ Odyssai Jurisprudencias - AI Legal Assistant

*[Leer versión en español abajo](#-odyssai-jurisprudencias---asistente-jurídico-ia)*

![Main Interface](/path/to/main_interface_screenshot.png)
*Main assistant view showing chat history and configuration panel.*

## 📋 Project Overview

**Odyssai Jurisprudencias** is an advanced **RAG (Retrieval-Augmented Generation)** system designed to revolutionize legal research and jurisprudence analysis in Argentina. Unlike traditional keyword-based search engines, this system leverages **Generative AI** and **Semantic Search** to understand the *intent* behind a lawyer's query, retrieving relevant case law even without exact keyword matches, and generating clear, grounded explanations.

The goal is to drastically reduce legal research time, allowing professionals to find precise precedents and obtain contextualized summaries in seconds.

## 🚀 Key Features

*   **🧠 Semantic & Hybrid Search**: Combines the power of vector embeddings with traditional metadata filters (Court, Chamber, Date) for unmatched precision.
*   **🔍 Neural Reranking**: Uses a Cross-Encoder model to reorder retrieved results, ensuring the most relevant cases always appear first.
*   **🤖 AI Data Enrichment**: An ETL pipeline that uses GPT-4o to analyze raw case texts and automatically extract keywords, legal figures, and key topics before indexing.
*   **💬 Conversational Assistant**: A natural chat interface that allows "talking" to the database, asking for clarifications, summaries, or drafting legal documents based on found cases.
*   **📂 Context Management**: Maintains conversation history to allow follow-up questions and search refinement.

## 🛠️ Technical Architecture

The system is divided into two main stages: the **Data Pipeline (ETL)** and the **Runtime Engine**.

### 1. Data Engineering (ETL Pipeline)

Before users can search, data undergoes a rigorous engineering process:

1.  **Ingestion & Cleaning**: Raw CSV files with case law are processed (`build_index.py`).
2.  **AI Enrichment**: Each case is analyzed by **GPT-4o-mini** (`enriquecimiento.py`) to generate structured metadata not present in the source:
    *   *Keywords*: "unjustified dismissal", "commuting accident".
    *   *Legal Figures*: Detects implicit legal concepts.
    *   *Tags*: Automatic thematic categorization.
3.  **Smart Chunking**: Implementation of a "soft split" algorithm that divides extensive texts into manageable chunks, respecting sentence and paragraph boundaries to preserve semantic context.
4.  **Vectorization**: Enriched chunks are converted into dense vectors using the `sentence-transformers/all-MiniLM-L6-v2` model and stored in **ChromaDB**.

### 2. Runtime (RAG Engine)

When a user makes a query, the system executes the following flow (`rag_engine.py`):

1.  **Query Analysis (LangChain + OpenAI)**:
    *   An agent classifies the user's intent: Do they want to chat (`CHAT`) or search for case law (`SEARCH`)?
    *   If searching, it extracts structured filters (e.g., "Civil Chamber", "last 5 years") and optimizes the search query.
2.  **Hybrid Retrieval**:
    *   Executes a vector search in ChromaDB to find semantic similarity.
    *   Simultaneously applies metadata filters (Court, Chamber, Date Range) to narrow the search space.
3.  **Neural Reranking**:
    *   The top-k raw results pass through a **Cross-Encoder** model (`ms-marco-MiniLM-L-6-v2`).
    *   This model "reads" the query and the document pair-by-pair to assign a relevance score much more precise than simple cosine similarity.
4.  **Contextual Generation**:
    *   The most relevant cases are injected into the **GPT-4o-mini** context.
    *   The model generates a natural response explaining why these cases are relevant to the user's specific scenario.

## 💻 Tech Stack

### Core & Backend
*   **Python 3.10+**: Base language.
*   **LangChain**: Orchestration framework for LLMs and Chains.
*   **FastAPI** (Backend API): To expose the engine as a service (optional).
*   **Pydantic**: Data validation and input/output schemas.

### AI & Data
*   **OpenAI API (GPT-4o-mini)**: Reasoning and generation engine.
*   **ChromaDB**: Open-source vector database.
*   **Sentence-Transformers**: Embedding models (`all-MiniLM-L6-v2`) and reranking (`cross-encoder`).
*   **Pandas**: Structured data manipulation and analysis.

### Frontend
*   **Streamlit**: Interactive and fast UI for prototyping and production.

## 📂 Project Structure

```bash
odyssai_jurisprudencias/
├── main.py                 # Streamlit Application Entry Point (Frontend)
├── rag_engine.py           # Core Logic: Search, RAG, LangChain, and Reranking
├── build_index.py          # Ingestion and Vector DB Creation Script
├── enriquecimiento.py      # ETL Script for AI Enrichment
├── extract_metadata.py     # Utility to inspect the DB
├── requirements.txt        # Project Dependencies
└── chroma_juris/           # Vector DB Persistence Directory
```

## 🔧 Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/odyssai-jurisprudencias.git
    cd odyssai-jurisprudencias
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables**:
    Create a `.env` or `secrets.toml` (for Streamlit) with your OpenAI API Key:
    ```toml
    OPENAI_API_KEY = "sk-..."
    ```

5.  **Run the application**:
    ```bash
    streamlit run main.py
    ```

---

# ⚖️ Odyssai Jurisprudencias - Asistente Jurídico IA

*[Read English version above](#-odyssai-jurisprudencias---ai-legal-assistant)*

![Main Interface](/path/to/main_interface_screenshot.png)
*Vista principal del asistente mostrando el historial de chat y el panel de configuración.*

## 📋 Descripción del Proyecto

**Odyssai Jurisprudencias** es un sistema avanzado de **RAG (Retrieval-Augmented Generation)** diseñado para revolucionar la búsqueda y análisis de jurisprudencia argentina. A diferencia de los buscadores tradicionales por palabras clave, este sistema utiliza **Inteligencia Artificial Generativa** y **Búsqueda Semántica** para entender la *intención* detrás de la consulta del abogado, recuperando fallos relevantes incluso si no coinciden las palabras exactas, y generando explicaciones claras y fundamentadas.

El objetivo es reducir drásticamente el tiempo de investigación legal, permitiendo a los profesionales encontrar antecedentes precisos y obtener resúmenes contextualizados en segundos.

## 🚀 Key Features

*   **🧠 Búsqueda Semántica & Híbrida**: Combina la potencia de los embeddings vectoriales con filtros de metadatos tradicionales (Tribunal, Sala, Fecha) para una precisión inigualable.
*   **🔍 Reranking Neuronal**: Utiliza un modelo Cross-Encoder para reordenar los resultados recuperados, asegurando que los fallos más relevantes aparezcan siempre primero.
*   **🤖 Enriquecimiento de Datos con IA**: Pipeline de ETL que utiliza GPT-4o para analizar fallos crudos y extraer automáticamente keywords, figuras jurídicas y temas clave antes de la indexación.
*   **💬 Asistente Conversacional**: Interfaz de chat natural que permite "dialogar" con la base de datos, pedir aclaraciones, resúmenes o redacción de escritos basados en los fallos encontrados.
*   **📂 Gestión de Contexto**: Mantiene el historial de la conversación para permitir preguntas de seguimiento y refinamiento de búsquedas.

## 🛠️ Arquitectura Técnica

El sistema se divide en dos grandes etapas: el **Pipeline de Datos (ETL)** y el **Motor de Ejecución (Runtime)**.

### 1. Data Engineering (ETL Pipeline)

Antes de que el usuario pueda buscar, los datos pasan por un proceso riguroso de ingeniería:

1.  **Ingesta & Limpieza**: Se procesan archivos CSV con fallos crudos (`build_index.py`).
2.  **AI Enrichment**: Cada fallo es analizado por **GPT-4o-mini** (`enriquecimiento.py`) para generar metadatos estructurados que no existían en la fuente original:
    *   *Keywords*: "despido injustificado", "accidente in itinere".
    *   *Figuras Jurídicas*: Detecta conceptos legales implícitos.
    *   *Tags*: Categorización temática automática.
3.  **Smart Chunking**: Implementación de un algoritmo de "soft split" que divide los textos extensos en fragmentos manejables (chunks) respetando los límites de oraciones y párrafos para no perder contexto semántico.
4.  **Vectorización**: Los chunks enriquecidos se convierten en vectores densos utilizando el modelo `sentence-transformers/all-MiniLM-L6-v2` y se almacenan en **ChromaDB**.

### 2. Runtime (RAG Engine)

Cuando el usuario realiza una consulta, el sistema ejecuta el siguiente flujo (`rag_engine.py`):

1.  **Query Analysis (LangChain + OpenAI)**:
    *   Un agente clasifica la intención del usuario: ¿Quiere charlar (`CHAT`) o buscar jurisprudencia (`SEARCH`)?
    *   Si es búsqueda, extrae filtros estructurados (ej: "Cámara Civil", "últimos 5 años") y optimiza la query de búsqueda.
2.  **Hybrid Retrieval**:
    *   Ejecuta una búsqueda vectorial en ChromaDB para encontrar similitud semántica.
    *   Aplica simultáneamente filtros de metadatos (Tribunal, Sala, Rango de Fechas) para acotar el espacio de búsqueda.
3.  **Neural Reranking**:
    *   Los top-k resultados crudos pasan por un modelo **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`).
    *   Este modelo "lee" la query y el documento par-a-par para asignar un puntaje de relevancia mucho más preciso que la similitud de coseno simple.
4.  **Contextual Generation**:
    *   Los fallos más relevantes se inyectan en el contexto de **GPT-4o-mini**.
    *   El modelo genera una respuesta natural explicando por qué esos fallos son relevantes para el caso planteado por el usuario.

## 💻 Tech Stack

### Core & Backend
*   **Python 3.10+**: Lenguaje base.
*   **LangChain**: Framework de orquestación para LLMs y Chains.
*   **FastAPI** (Backend API): Para exponer el motor como servicio (opcional).
*   **Pydantic**: Validación de datos y esquemas de entrada/salida.

### AI & Data
*   **OpenAI API (GPT-4o-mini)**: Motor de razonamiento y generación.
*   **ChromaDB**: Base de datos vectorial open-source.
*   **Sentence-Transformers**: Embedding models (`all-MiniLM-L6-v2`) and reranking (`cross-encoder`).
*   **Pandas**: Structured data manipulation and analysis.

### Frontend
*   **Streamlit**: Interfaz de usuario interactiva y rápida para prototipado y producción.

## 📂 Estructura del Proyecto

```bash
odyssai_jurisprudencias/
├── main.py                 # Punto de entrada de la aplicación Streamlit (Frontend)
├── rag_engine.py           # Núcleo lógico: Búsqueda, RAG, LangChain y Reranking
├── build_index.py          # Script de ingestión y creación de la base vectorial
├── enriquecimiento.py      # Script ETL para enriquecer fallos con IA
├── extract_metadata.py     # Utilidad para inspeccionar la DB
├── requirements.txt        # Dependencias del proyecto
└── chroma_juris/           # Directorio de persistencia de la base vectorial
```

## 🔧 Instalación y Uso

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/tu-usuario/odyssai-jurisprudencias.git
    cd odyssai-jurisprudencias
    ```

2.  **Crear entorno virtual**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar variables de entorno**:
    Crear un archivo `.env` o `secrets.toml` (para Streamlit) con tu API Key de OpenAI:
    ```toml
    OPENAI_API_KEY = "sk-..."
    ```

5.  **Ejecutar la aplicación**:
    ```bash
    streamlit run main.py
    ```

---
![Search Results](/path/to/search_results_screenshot.png)
*Ejemplo de resultados de búsqueda con explicación generada y tarjetas de fallos.*
