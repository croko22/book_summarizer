from langchain.text_splitter import RecursiveCharacterTextSplitter
from .providers import SummarizationProvider
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_summary_map_reduce(
    provider: SummarizationProvider,
    long_text: str,
    *,
    chunk_size: int = 1000,

    chunk_overlap: int = 100,
    language: str = "es"
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

    # Parallelize map phase
    chunk_summaries = []
    # Use max_workers=5 to avoid hitting rate limits too hard with external APIs, 
    # though valid for local models too if memory permits.
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_chunk = {
            executor.submit(provider.summarize, chunk, max_length=150, min_length=30, language=language): i 
            for i, chunk in enumerate(chunks)
        }
        
        results = [None] * len(chunks)
        for future in as_completed(future_to_chunk):
            index = future_to_chunk[future]
            try:
                results[index] = future.result()
            except Exception as e:
                print(f"Error processing chunk {index}: {e}")
                results[index] = "" 
        
        chunk_summaries = [r for r in results if r]
    
    combined_summaries = "\n".join(chunk_summaries)
    if len(chunks) == 1:
        return combined_summaries

    final_summary = provider.summarize(
        f"Crea un resumen final coherente a partir de los siguientes resúmenes parciales:\n\n{combined_summaries}" if language == "es" else f"Create a coherent final summary from the following partial summaries:\n\n{combined_summaries}",
        max_length=500,
        min_length=150,
        language=language
    )
    return final_summary

def generate_summary_incremental(
    provider: SummarizationProvider,
    long_text: str,
    *,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    focus_instruction: str = None,
    language: str = "es"
) -> dict:
    """
    Genera un resumen usando una estrategia incremental optimizada.
    
    Si el provider tiene método iterativo (como GemmaBookSumProvider),
    usa esa implementación. Sino, usa la estrategia Refine tradicional.
    Devuelve un diccionario con 'summary' y 'chunks'.
    """
    if hasattr(provider, 'summarize_iterative'):
        return provider.summarize_iterative(long_text, chunk_size, focus_instruction=focus_instruction, language=language)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(long_text)
    if not chunks:
        return {"summary": "", "chunks": []}

    chunk_summaries = []

    base_instruction = "Resume el siguiente texto de forma concisa y clara" if language == "es" else "Summarize the following text concisely and clearly"
    if focus_instruction:
        if language == "es":
            base_instruction += f" siguiendo esta instrucción: {focus_instruction}"
        else:
            base_instruction += f" following this instruction: {focus_instruction}"
    
    initial_prompt = f'{base_instruction}:\n\n"{chunks[0]}"'
    running_summary = provider.summarize(initial_prompt, max_length=400, min_length=100, language=language)
    
    chunk_summaries.append({
        'chunk_number': 1,
        'text_preview': chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0],
        'summary': running_summary
    })
    
    for i, chunk in enumerate(chunks[1:]):
        if language == "es":
            refine_prompt = f"""
            Resumen existente:
            "{running_summary}"

            Nuevo fragmento de texto:
            "{chunk}"

            Usando el nuevo fragmento, refina el resumen existente para que sea más completo y coherente.
            {f'Instrucción de enfoque: {focus_instruction}' if focus_instruction else ''}

            PAUTAS:
            - NO repitas frases del resumen anterior.
            - Integra la nueva información fluidamente.
            """
        else:
            refine_prompt = f"""
            Existing summary:
            "{running_summary}"

            New text segment:
            "{chunk}"

            Using the new segment, refine the existing summary to be more complete and coherent.
            {f'Focus instruction: {focus_instruction}' if focus_instruction else ''}

            GUIDELINES:
            - Do NOT repeat phrases from the previous summary.
            - Integrate new information smoothly.
            """
        running_summary = provider.summarize(refine_prompt, max_length=500, min_length=200, language=language)
        
        chunk_summaries.append({
            'chunk_number': i + 2,
            'text_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk,
            'summary': running_summary
        })

    return {
        "summary": running_summary,
        "chunks": chunk_summaries
    }