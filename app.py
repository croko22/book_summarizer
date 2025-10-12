import streamlit as st
from typing import Dict, Type, Optional
import book_summarizer.file_processor as file_processor
from book_summarizer.providers import (
    SummarizationProvider,
    OpenAIProvider,
    HuggingFaceInstructionProvider,
    HuggingFaceSummarizerProvider,
)
from book_summarizer.summarizer import generate_summary_incremental, generate_summary_map_reduce

st.set_page_config(
    page_title="Resumen de Textos con IA",
    page_icon="📚",
    layout="wide",
)

# Diccionario centralizado para la configuración de modelos
MODEL_CONFIG: Dict[str, Type[SummarizationProvider]] = {
    "GPT-3.5-Turbo (OpenAI)": OpenAIProvider,
    "Flan-T5-Base (Local/Refine)": HuggingFaceInstructionProvider,
    "DistilBART (Local/Map-Reduce)": HuggingFaceSummarizerProvider,
}

# --- Funciones de Caching de Modelos ---

@st.cache_resource
def get_provider(
    provider_class: Type[SummarizationProvider],
    model_name: Optional[str] = None,
    **kwargs,
) -> SummarizationProvider:
    """Carga y cachea una instancia de un proveedor de modelo.

    Si no se provee `model_name`, se instancia el proveedor usando
    su valor por defecto definido en su constructor.
    """
    if model_name:
        return provider_class(model_name=model_name, **kwargs)
    return provider_class(**kwargs)

# --- Lógica de la Interfaz ---

def render_sidebar():
    """Renderiza la barra lateral y devuelve las opciones seleccionadas."""
    st.sidebar.title("⚙️ Configuración")
    
    model_key = st.sidebar.selectbox(
        "Elige el modelo:",
        MODEL_CONFIG.keys()
    )
    
    strategy_key = st.sidebar.radio(
        "Elige la estrategia:",
        ("Incremental (Refine)", "Map-Reduce"),
        index=0 if "Flan-T5" in model_key or "GPT" in model_key else 1,
        help="Incremental es mejor para coherencia; Map-Reduce es más rápido."
    )
    
    api_key = None
    if "OpenAI" in model_key:
        api_key = st.sidebar.text_input("Ingresa tu API Key de OpenAI:", type="password")

    return model_key, strategy_key, api_key

def get_text_input() -> str:
    """Maneja la entrada de texto, ya sea por área de texto o carga de archivo."""
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
                if file_extension == "txt":
                    text = file_processor.get_text_from_txt(uploaded_file)
                elif file_extension == "pdf":
                    text = file_processor.get_text_from_pdf(uploaded_file)
                elif file_extension == "docx":
                    text = file_processor.get_text_from_docx(uploaded_file)
                elif file_extension == "epub":
                    text = file_processor.get_text_from_epub(uploaded_file)
    return text

def main():
    """Función principal que ejecuta la aplicación Streamlit."""
    st.title("📚 Resumen de Textos con IA")

    if "summary" not in st.session_state:
        st.session_state.summary = ""

    model_key, strategy_key, api_key = render_sidebar()
    user_text = get_text_input()

    st.header("2. Genera el Resumen")
    if st.button("Generar Resumen", type="primary"):
        if not user_text.strip():
            st.warning("Por favor, ingresa texto o sube un archivo.")
            return

        provider_class = MODEL_CONFIG[model_key]
        provider_kwargs = {}
        if "OpenAI" in model_key:
            if not api_key:
                st.error("Se requiere una API Key de OpenAI para este modelo.")
                return
            provider_kwargs["api_key"] = api_key
        
        try:
            with st.spinner("Cargando modelo y procesando resumen..."):
                provider = get_provider(provider_class, **provider_kwargs)
                
                if "Incremental" in strategy_key:
                    if not isinstance(provider, (OpenAIProvider, HuggingFaceInstructionProvider)):
                        st.error("La estrategia 'Incremental' solo funciona con modelos de instrucciones (GPT, Flan-T5).")
                        return
                    summary = generate_summary_incremental(provider, user_text)
                else:
                    summary = generate_summary_map_reduce(provider, user_text)
                
                st.session_state.summary = summary
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

    if st.session_state.summary:
        st.header("✅ Resumen Generado")
        st.success(st.session_state.summary)

if __name__ == "__main__":
    main()