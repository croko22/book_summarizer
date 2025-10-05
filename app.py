import streamlit as st
from summarizer import generate_summary

st.set_page_config(layout="wide")

st.title("RESUMEN DE LIBROS")
st.write("Esta aplicación utiliza LangChain y Hugging Face para resumir textos largos.")

user_input = st.text_area("Pega el texto o documento que deseas resumir:", height=300)

if st.button("Generar resumen"):
    if user_input:
        with st.spinner("Procesando el documento y generando el resumen..."):
            try:
                summary_output = generate_summary(user_input)
                st.subheader("Resumen generado:")
                st.success(summary_output)
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
    else:
        st.warning("Por favor, pega un texto en el área de entrada.")
