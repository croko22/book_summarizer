import streamlit as st
import time
import json
from datetime import datetime
from book_summarizer import file_processor
from book_summarizer.providers import GemmaBookSumProvider
from book_summarizer.summarizer import generate_summary_incremental
from book_summarizer.database import SummaryDatabase

st.set_page_config(
    page_title="Resumen de Textos",
    page_icon="📚",
    layout="wide",
)

@st.cache_resource
def get_provider():
    return GemmaBookSumProvider()

@st.cache_resource
def get_database():
    return SummaryDatabase()


@st.dialog("Detalles del Resumen", width="large")
def show_summary_details(item):
    st.subheader(item.get('title', 'Resumen sin título'))
    st.write(f"**Fecha:** {item['timestamp']}")
    st.write(f"**Método:** {item['method']}")
    st.write(f"**Palabras:** {item['word_count']}")

    col1, col2 = st.columns(2)
    with col1:
        markdown_content = f"# Resumen de Texto\n\n**Fecha:** {item['timestamp']}\n\n**Método:** {item['method']}\n\n**Palabras:** {item['word_count']}\n\n## Resumen\n\n{item['summary']}"
        st.download_button(
            label="Descargar MD",
            icon=":material/markdown:",
            data=markdown_content,
            file_name=f"resumen_{item['id']}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col2:
        if st.button("Borrar del Historial", icon=":material/delete:", key=f"delete_{item['id']}", use_container_width=True):
            db = get_database()
            db.delete_summary(item['id'])
            st.rerun()


    st.markdown("---")
    
    with st.expander("📄 Ver Resumen Generado", expanded=True):
        st.markdown(item['summary'])
        
    if item.get('chunks_data'):
        try:
            chunks = json.loads(item['chunks_data'])
            with st.expander("🧩 Ver Desglose por Chunks", expanded=False):
                for chunk in chunks:
                    st.markdown(f"### Chunk {chunk['chunk_number']}")
                    st.markdown(f"**Preview:** `{chunk['text_preview']}`")
                    st.markdown(chunk['summary'])
                    st.markdown("---")
        except:
            pass
        
    with st.expander("📝 Ver Texto Original", expanded=False):
        st.markdown(item['original_text'])

def render_sidebar():
    st.sidebar.title("⚙️ Configuración")
    
    method = st.sidebar.radio(
        "Método de procesamiento:",
        ("Iterativo (Recomendado)", "Estándar"),
        help="Iterativo: Procesa por chunks usando el prompt específico del modelo. Estándar: Método tradicional."
    )
    
    # Botón para limpiar cache si hay problemas
    if st.sidebar.button("🔄 Reiniciar Modelo", help="Limpia el cache y recarga el modelo"):
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Estadísticas")
    if "text_stats" in st.session_state:
        stats = st.session_state.text_stats
        st.sidebar.markdown(f"""
        📝 **{stats.get("words", 0)}** palabras  
        📊 **{stats.get("chars", 0)}** caracteres  
        {f'⏱️ **{stats["processing_time"]:.1f}s** procesamiento' if "processing_time" in stats else ''}
        """, help="Estadísticas del último texto procesado")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Historial")
    
    # Búsqueda en historial
    search_query = st.sidebar.text_input("🔍 Buscar en historial:", placeholder="Escribe para buscar...")
    
    db = get_database()
    
    # Obtener resúmenes (búsqueda o recientes)
    if search_query:
        recent_summaries = db.search_summaries(search_query, limit=5)
    else:
        recent_summaries = db.get_recent_summaries(limit=5)
    
    if recent_summaries:
        for item in recent_summaries:
            # Usar título si existe, sino fecha
            label = item.get('title') or f"#{item['id']} | {item['timestamp'][8:16]}"
            if len(label) > 30:
                label = label[:27] + "..."
                
            if st.sidebar.button(label, key=f"btn_summary_{item['id']}", use_container_width=True, help=f"{item.get('title', '')}\n{item['timestamp']}"):
                show_summary_details(item)
    else:
        if search_query:
            st.sidebar.info("No se encontraron resultados")
        else:
            st.sidebar.info("No hay historial disponible")
    
    # Estadísticas globales
    stats = db.get_statistics()
    if stats['total_summaries'] > 0:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Estadísticas Globales")
        st.sidebar.metric("Total resúmenes", stats['total_summaries'])
        st.sidebar.metric("Palabras procesadas", f"{stats['total_words']:,}")
        st.sidebar.metric("Tiempo promedio", f"{stats['avg_processing_time']:.1f}s")
    
    return method
def get_text_input() -> str:
    st.header("1. Ingresa el Texto")
    input_method = st.radio("Método de entrada:", ("Pegar texto", "Subir archivo"))
    
    text = ""
    if input_method == "Pegar texto":
        text = st.text_area("Pega el texto a resumir aquí:", height=250, label_visibility="collapsed")
    else:
        uploaded_file = st.file_uploader(
            "Sube un archivo (.txt, .pdf, .docx, .epub)",
            type=["txt", "pdf", "docx", "epub"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            with st.spinner(f"Procesando archivo '{uploaded_file.name}'..."):
                file_extension = uploaded_file.name.split('.')[-1].lower()
                text_extractors = {
                    "txt": file_processor.get_text_from_txt,
                    "pdf": file_processor.get_text_from_pdf,
                    "docx": file_processor.get_text_from_docx,
                    "epub": file_processor.get_text_from_epub,
                }
                if file_extension in text_extractors:
                    text = text_extractors[file_extension](uploaded_file)
    return text

def main():
    st.title("📚 Resumen de Textos con IA")

    # Inicializar estados de sesión
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "text_stats" not in st.session_state:
        st.session_state.text_stats = {}

    method = render_sidebar()
    user_text = get_text_input()

    # Calcular estadísticas del texto
    if user_text:
        st.session_state.text_stats = {
            "words": len(user_text.split()),
            "chars": len(user_text),
            "lines": len(user_text.split('\n'))
        }

    st.header("2. Genera el Resumen")
    if st.button("Generar Resumen", type="primary"):
        if not user_text.strip():
            st.warning("Por favor, ingresa texto o sube un archivo.")
            return
        
        try:
            start_time = time.time()
            
            # Crear placeholder para la barra de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            provider = get_provider()
            
            if method == "Iterativo (Recomendado)":
                if hasattr(provider, 'summarize_iterative'):
                    # Calcular chunks para estimar progreso
                    chunk_size = 4000
                    estimated_chunks = max(1, len(user_text) // chunk_size + (1 if len(user_text) % chunk_size else 0))
                    
                    status_text.info(f"📊 Procesando texto en ~{estimated_chunks} chunks. Tiempo estimado: ~{estimated_chunks * 15}s")
                    progress_bar.progress(5)
                    
                    # Función callback para actualizar progreso
                    def update_progress(current, total):
                        progress_percent = int((current / total) * 85) + 10  # 10-95%
                        progress_bar.progress(progress_percent)
                        status_text.info(f"🔄 Procesando chunk {current}/{total}... ({progress_percent}%)")
                    
                    # Intentar usar progress_callback si está disponible
                    try:
                        result = provider.summarize_iterative(user_text, chunk_size=chunk_size, progress_callback=update_progress)
                    except TypeError:
                        # Versión antigua sin progress_callback - usar sin callback
                        st.warning("⚠️ Modelo en versión antigua. Presiona '🔄 Reiniciar Modelo' para actualizar y ver progreso en tiempo real.")
                        result = provider.summarize_iterative(user_text, chunk_size=chunk_size)
                    
                    if isinstance(result, dict):
                        summary = result['summary']
                        chunks = result.get('chunks', [])
                    else:
                        summary = result
                        chunks = []
                    
                    progress_bar.progress(95)
                    status_text.info("✨ Finalizando resumen...")
                else:
                    st.error("El método iterativo no está disponible. Presiona 🔄 Reiniciar Modelo en el sidebar.")
                    progress_bar.empty()
                    status_text.empty()
                    return
            else:
                status_text.info("🔄 Procesando con método estándar...")
                progress_bar.progress(20)
                result = generate_summary_incremental(provider, user_text)
                
                if isinstance(result, dict):
                    summary = result['summary']
                    chunks = result.get('chunks', [])
                else:
                    summary = result
                    chunks = []
                    
                progress_bar.progress(90)
            
            # Completar progreso
            progress_bar.progress(98)
            status_text.info("🏷️ Generando título...")
            
            # Generar título
            try:
                title = provider.generate_title(user_text)
            except Exception:
                title = f"Resumen {datetime.now().strftime('%H:%M')}"
            
            progress_bar.progress(100)
            status_text.success("✅ ¡Resumen completado!")
            time.sleep(0.5)  # Breve pausa para mostrar el 100%
            
            # Limpiar indicadores
            progress_bar.empty()
            status_text.empty()
            
            processing_time = time.time() - start_time
            st.session_state.text_stats["processing_time"] = processing_time
            st.session_state.summary = summary
            st.session_state.chunks = chunks
            
            # Guardar en base de datos SQLite
            db = get_database()
            summary_id = db.save_summary(
                original_text=user_text,
                summary=summary,
                word_count=st.session_state.text_stats.get('words', 0),
                char_count=st.session_state.text_stats.get('chars', 0),
                processing_time=processing_time,
                method=method,
                chunks_data=json.dumps(chunks) if chunks else None,
                title=title
            )
            
            # Limpiar resúmenes antiguos (mantener últimos 100)
            db.cleanup_old_summaries(keep_last=100)
                
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
            st.exception(e)

    if st.session_state.summary:
        st.header("✅ Resumen Generado")
        
        # Mostrar el resumen con formato markdown nativo
        st.markdown(st.session_state.summary)
        
        if st.session_state.get('chunks'):
            with st.expander("🧩 Ver Desglose por Chunks", expanded=False):
                for chunk in st.session_state.chunks:
                    st.markdown(f"### Chunk {chunk['chunk_number']}")
                    st.markdown(f"**Preview:** `{chunk['text_preview']}`")
                    st.markdown(chunk['summary'])
                    st.markdown("---")
        
        # Opciones de descarga y copia
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Botón para descargar como .txt
            st.download_button(
                label="Descargar TXT",
                icon=":material/description:",
                data=st.session_state.summary,
                file_name=f"resumen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Botón para descargar como .md
            markdown_content = f"# Resumen de Texto\n\n**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n**Estadísticas del texto original:**\n- Palabras: {st.session_state.text_stats.get('words', 0)}\n- Caracteres: {st.session_state.text_stats.get('chars', 0)}\n\n## Resumen\n\n{st.session_state.summary}"
            
            st.download_button(
                label="Descargar MD",
                icon=":material/markdown:",
                data=markdown_content,
                file_name=f"resumen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        with col3:
            # Botón para limpiar
            if st.button("Limpiar", icon=":material/clear_all:", help="Limpiar el resumen actual"):
                st.session_state.summary = ""
                st.session_state.chunks = []
                st.rerun()
        
        # Botón para exportar historial completo
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Exportar Historial Completo", icon=":material/download:", help="Descargar todos los resúmenes en CSV"):
                db = get_database()
                csv_path = f"historial_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                try:
                    db.export_to_csv(csv_path)
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="Descargar CSV",
                            icon=":material/csv:",
                            data=f.read(),
                            file_name=csv_path,
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error al exportar: {e}")
        
        # Mostrar estadísticas compactas
        if st.session_state.text_stats:
            summary_words = len(st.session_state.summary.split())
            processing_time = st.session_state.text_stats.get('processing_time', 0)

            stats_text = f"📝 <strong>{st.session_state.text_stats.get('words', 0)}</strong> palabras originales | 📄 <strong>{summary_words}</strong> palabras resumen | ⏱️ <strong>{processing_time:.1f}s</strong> procesamiento"
            st.markdown(f"<div style='text-align: center; color: #666; font-size: 0.9em; margin: 10px 0;'>{stats_text}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()