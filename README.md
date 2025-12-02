# 📘 Book Summarizer

## Descripción

**Book Summarizer** es una aplicación avanzada de resumen de textos construida con *Streamlit* y potenciada por **Gemma 3**, específicamente el modelo `croko22/gemma-booksum-lora-v1` fine-tuneado con el dataset BookSum.

Esta herramienta está diseñada para procesar textos largos (libros completos, capítulos, artículos extensos) y generar resúmenes coherentes, detallados y estructurados. A diferencia de los resúmenes genéricos, este proyecto permite enfoques personalizados (centrarse en personajes, conceptos, lecciones) y utiliza técnicas de procesamiento iterativo para mantener el contexto a lo largo de documentos extensos.

## Características Principales

![pipeline](docs/img/pipeline.png)

*   **Modelo Especializado**: Utiliza `croko22/gemma-booksum-lora-v1`, un modelo Gemma 3 optimizado para la tarea de resumen de libros.
*   **Procesamiento de Textos Largos**:
    *   **Método Iterativo**: Procesa el texto en fragmentos secuenciales, donde cada resumen se construye sobre el contexto del anterior, ideal para mantener la narrativa.
    *   **Método Map-Reduce**: (Opcional) Procesa fragmentos en paralelo y luego los combina.
*   **Enfoques de Resumen Personalizables**:
    *   General (Resumen estándar)
    *   Personajes y Relaciones
    *   Conceptos Clave
    *   Lecciones Prácticas
    *   Instrucciones Personalizadas
*   **Soporte de Archivos**: Carga directa de PDF, DOCX, EPUB y TXT.
*   **Historial y Gestión**: Base de datos SQLite integrada para guardar, buscar, etiquetar y exportar resúmenes.
*   **Aceleración GPU**: Optimizado para ejecutarse localmente con CUDA.

## Resultados de Evaluación

El modelo ha demostrado una capacidad excepcional para capturar tanto la estructura general como los detalles finos de obras complejas.

### Ejemplo: *Meditaciones* de Marco Aurelio
*Tiempo de procesamiento: ~10 chunks | Enfoque: General*

> **Updated Summary:**
> The book is divided into twelve books, with an introduction, appendix, notes, and glossary.
>
> The introduction details Marcus Aurelius's life, starting with his birth in 121 A.D. as M. Annius Verus. It describes his noble lineage, his adoption by his grandfather and later by Emperor Antoninus Pius, his education in Stoic philosophy, and his marriage to Faustina.
> ...
> Specific points covered include:
> *   **Gratitude for positive influences** in his life, such as never having cause to repent for actions towards Rusticus...
> *   **A reminder to prepare for encountering difficult people** (idle, unthankful, envious) and to remember their shared humanity...
> *   **An encouragement to simplify one's life**, focusing on reason and avoiding distractions from books and anxieties about the future.

*(Puedes ver ejemplos completos en la carpeta `evaluation_results/`)*

## Estructura del Proyecto

```
book_summarizer/
├── app.py                     # Aplicación principal Streamlit
├── book_summarizer/
│   ├── providers.py           # Lógica del modelo Gemma 3 y Gemini
│   ├── summarizer.py          # Algoritmos de resumen (Iterativo/Map-Reduce)
│   ├── file_processor.py      # Extractores de texto (PDF, EPUB, etc.)
│   └── database.py            # Gestión de historial SQLite
├── evaluation_results/        # Ejemplos de resúmenes generados
├── requirements.txt           # Dependencias
└── README.md                  # Documentación
```

## Instalación

Se recomienda utilizar un entorno virtual y una GPU con soporte CUDA para un rendimiento óptimo.

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/tuusuario/book_summarizer.git
    cd book_summarizer
    ```

2.  **Crear entorno virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # .\venv\Scripts\activate  # Windows
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: Asegúrate de tener instalados los drivers de NVIDIA si planeas usar la aceleración por GPU.*

## Uso

1.  **Iniciar la aplicación:**
    ```bash
    streamlit run app.py
    ```

2.  **Generar un resumen:**
    *   Sube tu archivo (PDF, EPUB, etc.) en la pestaña "Generar Resumen".
    *   Selecciona el modelo **Gemma (Local)** en la barra lateral.
    *   Elige el **Tipo de resumen** (ej. "Conceptos Clave").
    *   Haz clic en **Generar Resumen**.

3.  **Explorar resultados:**
    *   Visualiza el resumen final y el desglose por "chunks".
    *   Descarga el resultado en Markdown o TXT.
    *   Accede a resúmenes anteriores en la pestaña "Biblioteca".

## Configuración Avanzada

Si deseas utilizar el proveedor de respaldo en la nube (Gemini 3 Pro), necesitarás configurar tu API Key:

```bash
export GOOGLE_API_KEY="tu_api_key_aqui"
```
O ingrésala directamente en la interfaz de usuario.