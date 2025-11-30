from langchain.text_splitter import RecursiveCharacterTextSplitter
from .providers import SummarizationProvider

def generate_summary_map_reduce(
    provider: SummarizationProvider,
    long_text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> str:
    """
    Genera un resumen usando la estrategia Map-Reduce.
    
    Primero resume cada fragmento de forma independiente (map) y luego
    combina los resúmenes de los fragmentos en un resumen final (reduce).
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(long_text)
    if not chunks:
        return ""

    chunk_summaries = [provider.summarize(chunk, max_length=150, min_length=30) for chunk in chunks]
    
    combined_summaries = "\n".join(chunk_summaries)
    if len(chunks) == 1:
        return combined_summaries

    final_summary = provider.summarize(
        f"Crea un resumen final coherente a partir de los siguientes resúmenes parciales:\n\n{combined_summaries}",
        max_length=500,
        min_length=150
    )
    return final_summary

def generate_summary_incremental(
    provider: SummarizationProvider,
    long_text: str,
    *,
    chunk_size: int = 4000,
    chunk_overlap: int = 200
) -> dict:
    """
    Genera un resumen usando una estrategia incremental optimizada.
    
    Si el provider tiene método iterativo (como GemmaBookSumProvider),
    usa esa implementación. Sino, usa la estrategia Refine tradicional.
    Devuelve un diccionario con 'summary' y 'chunks'.
    """
    if hasattr(provider, 'summarize_iterative'):
        return provider.summarize_iterative(long_text, chunk_size)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(long_text)
    if not chunks:
        return {"summary": "", "chunks": []}

    chunk_summaries = []

    initial_prompt = f'Resume el siguiente texto de forma concisa y clara:\n\n"{chunks[0]}"'
    running_summary = provider.summarize(initial_prompt, max_length=400, min_length=100)
    
    chunk_summaries.append({
        'chunk_number': 1,
        'text_preview': chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0],
        'summary': running_summary
    })
    
    for i, chunk in enumerate(chunks[1:]):
        refine_prompt = f"""
        Resumen existente:
        "{running_summary}"

        Nuevo fragmento de texto:
        "{chunk}"

        Usando el nuevo fragmento, refina el resumen existente para que sea más completo y coherente.
        """
        running_summary = provider.summarize(refine_prompt, max_length=500, min_length=200)
        
        chunk_summaries.append({
            'chunk_number': i + 2,
            'text_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk,
            'summary': running_summary
        })

    return {
        "summary": running_summary,
        "chunks": chunk_summaries
    }