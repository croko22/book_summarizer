from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import torch
from typing import List

def _select_device():

    return 0 if torch.cuda.is_available() else -1

_summarizer_pipeline = None

def _get_pipeline():
    """Lazily initialize and cache the transformers summarization pipeline.

    Importing this module will no longer trigger model download / GPU allocation.
    """
    global _summarizer_pipeline
    if _summarizer_pipeline is None:
        
        from transformers import pipeline as _pipeline

        _summarizer_pipeline = _pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=_select_device(),
        )
    return _summarizer_pipeline

def _summarize_text(text: str, max_length: int = 142, min_length: int = 20) -> str:
    """Run the transformers summarization pipeline on `text` with truncation.

    This is a thin wrapper that ensures truncation=True to avoid token indexing errors
    and returns the resulting summary string.
    """
    pipeline = _get_pipeline()
    result = pipeline(text, max_length=max_length, min_length=min_length, truncation=True)
    return result[0]["summary_text"].strip()

def _reduce_summaries(summaries: List[str], final_max_length: int = 180) -> str:
    """Combine per-chunk summaries and summarize them into a single final summary.

    If the combined text is still too long for the model, reduce it recursively.
    """
    combined = "\n\n".join(summaries)
    
    while len(combined) > 3000:
        combined = _summarize_text(combined, max_length=final_max_length)

    return _summarize_text(combined, max_length=final_max_length)

def generate_summary(long_text: str, *, chunk_size: int = 800, chunk_overlap: int = 100) -> str:
    """Generate a summary for a long text using a simple map-reduce approach.

    - Splits the input into character-based chunks (defaults tuned to avoid tokenizer limits).
    - Summarizes each chunk separately (map phase).
    - Combines chunk summaries and summarizes again to produce a single final summary (reduce phase).

    This avoids using the deprecated LangChain HuggingFacePipeline and `Chain.run` APIs,
    and protects against token indexing errors by truncating long inputs.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents([long_text])
    chunks = [d.page_content for d in docs]

    chunk_summaries: List[str] = []
    for idx, chunk in enumerate(chunks):
        
        per_chunk_max = max(32, min(150, int(len(chunk) / 10)))
        try:
            summary = _summarize_text(chunk, max_length=per_chunk_max)
        except Exception:
            
            summary = _summarize_text(chunk, max_length=64)
        chunk_summaries.append(summary)

    if not chunk_summaries:
        return ""

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    final = _reduce_summaries(chunk_summaries)
    return final