from abc import ABC, abstractmethod
from typing import Optional, Union, Generator, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from google import genai
import time

class SummarizationProvider(ABC):
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name

    @abstractmethod
    def summarize(self, text: str, max_length: int = 500, min_length: int = 50, language: str = "es", stream: bool = False):
        pass

    def generate_title(self, text: str) -> str:
        return " ".join(text.split()[:5]) + "..."
    
    def generate_tags(self, text: str) -> list[str]:
        return []

class GemmaBookSumProvider(SummarizationProvider):
    _tokenizer = None
    _model = None
    
    def __init__(self, model_name: str = "croko22/gemma-booksum-lora-v1"):
        super().__init__(model_name)
        if GemmaBookSumProvider._tokenizer is None:
            GemmaBookSumProvider._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            GemmaBookSumProvider._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            
            if not torch.cuda.is_available():
                GemmaBookSumProvider._model.to("cpu")
            
    def summarize(self, text: str, max_length: int = 500, min_length: int = 50, focus_instruction: str = None, language: str = "es", stream: bool = False):
        base_instruction = "Resume el siguiente texto" if language == "es" else "Summarize the following text"
            
        if focus_instruction:
            connector = "siguiendo esta instrucción:" if language == "es" else "following this instruction:"
            base_instruction += f" {connector} {focus_instruction}"
        else:
            base_instruction += ":"
            
        prompt = f"{base_instruction}\n\n{text}\n\nResumen:"
        
        inputs = self._tokenizer(text=prompt, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if stream:
            return self._stream_generation(inputs, max_length, min_length)
            
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=min_length,
                temperature=0.6,
                do_sample=True,
            )
        
        generated_text = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text.replace(prompt, "").strip()
    
    def summarize_iterative(self, text: str, chunk_size: int = 4000, max_new_tokens: int = 2048, progress_callback=None, focus_instruction: str = None, language: str = "es", stream: bool = False) -> Union[Dict[str, Any], Generator]:
        chunks = self._split_text(text, chunk_size)
        if not chunks: return ""
        
        if len(chunks) == 1:
            result = self.summarize(text, max_length=max_new_tokens, focus_instruction=focus_instruction, stream=stream)
            return result if stream else result
        
        chunk_summaries = []
        accumulated_summary = ""
        context_summary = "" # Resumen breve para dar contexto al siguiente chunk
        
        # Generator for streaming
        def stream_generator():
            nonlocal accumulated_summary, context_summary
            
            for i, chunk in enumerate(chunks):
                if progress_callback: progress_callback(i + 1, len(chunks))
                
                # Incremental Append Strategy
                # Generamos el resumen SÓLO de este chunk, usando el anterior como contexto
                
                if i == 0:
                    prompt = self._get_initial_prompt(chunk, language)
                else:
                    prompt = self._get_incremental_prompt(chunk, context_summary, language)

                inputs = self._tokenizer(text=prompt, return_tensors="pt", padding=True)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Streaming generation for this chunk
                streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = dict(
                    inputs,
                    streamer=streamer,
                    max_new_tokens=600, # Summaries per chunk shouldn't be too long
                    min_new_tokens=100,
                    temperature=0.4, # Low temp to avoid hallucinations
                    do_sample=True,
                    repetition_penalty=1.2
                )
                
                thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
                thread.start()
                
                chunk_text = ""
                yield f"\n\n#### Parte {i+1}\n\n"
                
                for new_text in streamer:
                    chunk_text += new_text
                    yield new_text
                
                # Limpiar resultado
                chunk_text = chunk_text.replace(prompt, "").strip()
                
                # Actualizar acumulados
                chunk_summaries.append({
                    'chunk_number': i + 1,
                    'text_preview': chunk[:100] + "...",
                    'summary': chunk_text
                })
                accumulated_summary += f"\n\n#### Parte {i+1}\n\n{chunk_text}"
                # Mantener un contexto breve (últimos 1000 cars) para el siguiente paso
                context_summary = (context_summary + " " + chunk_text)[-1000:]

        if stream:
            return stream_generator()

        # Non-streaming execution
        for val in stream_generator():
            pass # Consume generator
            
        return {
            "summary": self._format_final_output(accumulated_summary, len(chunks), len(text), language),
            "chunks": chunk_summaries
        }

    def _get_initial_prompt(self, chunk: str, language: str) -> str:
        if language == "es":
            return f"""Resume el siguiente texto de manera detallada y objetiva.
ESTILO: Académico, formal, directo.
PROHIBIDO: Emojis, saludos, "Espero que sirva".

Texto:
{chunk}

Resumen:"""
        else:
            return f"""Summarize the following text in a detailed and objective way.
STYLE: Academic, formal, direct.
FORBIDDEN: Emojis, greetings, "Hope this helps".

Text:
{chunk}

Summary:"""

    def _get_incremental_prompt(self, chunk: str, context: str, language: str) -> str:
        if language == "es":
            return f"""Contexto anterior: "{context}..."

Resume la SIGUIENTE parte del texto, continuando la narrativa.
ESTILO: Académico, formal, directo. Sin repeticiones.
PROHIBIDO: Emojis, saludos.

Nueva Parte:
{chunk}

Resumen de la Nueva Parte:"""
        else:
            return f"""Previous context: "{context}..."

Summarize the FOLLOWING part of the text, continuing the narrative.
STYLE: Academic, formal, direct. No repetitions.
FORBIDDEN: Emojis, greetings.

New Part:
{chunk}

Summary of New Part:"""


    def _stream_generation(self, inputs, max_new_tokens=500, min_new_tokens=50, temperature=0.6):
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.3
        )
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def _get_style_guidelines(self, language: str) -> str:
        # Not used directly in the new prompts but kept for reference if needed
        return ""

    def _format_final_output(self, summary: str, chunks_count: int, length: int, language: str) -> str:
        header = "📚 Reporte de Resumen" if language == "es" else "📚 Summary Report"
        info = "Información de Procesamiento" if language == "es" else "Processing Info"
        return f"""# {header}
## 📊 {info}
- Chunks: {chunks_count}
- Input Length: {length} chars
---
## 🎯 Resumen Completo
{summary}"""

    def _split_text(self, text: str, chunk_size: int) -> list[str]:
        if len(text) <= chunk_size: return [text]
        chunks = []
        current_chunk = ""
        for para in text.split('\n\n'):
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += ("\n\n" + para) if current_chunk else para
            else:
                if current_chunk: chunks.append(current_chunk)
                current_chunk = para
                
                # Split huge paragraphs
                if len(para) > chunk_size:
                    while len(current_chunk) > chunk_size:
                        chunks.append(current_chunk[:chunk_size])
                        current_chunk = current_chunk[chunk_size:]
                        
        if current_chunk: chunks.append(current_chunk)
        return chunks

    def generate_title(self, text: str) -> str:
        prompt = f"Genera un título muy corto (máximo 5 palabras) para:\n\n{text[:1000]}\n\nTítulo:"
        inputs = self._tokenizer(text=prompt, return_tensors="pt", padding=True)
        if torch.cuda.is_available(): inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7)
            
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].replace(prompt, "").strip().split('\n')[0].strip('"')

    def generate_tags(self, text: str) -> list[str]:
        prompt = f"Genera 3-5 etiquetas separadas por comas para:\n\n{text[:2000]}\n\nEtiquetas:"
        inputs = self._tokenizer(text=prompt, return_tensors="pt", padding=True)
        if torch.cuda.is_available(): inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.3)
            
        text = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].replace(prompt, "").strip()
        return [t.strip().strip('.') for t in text.replace('\n', ',').replace('-', '').split(',') if t.strip()][:5]


class GeminiProvider(SummarizationProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        super().__init__(model_name)
        self.client = genai.Client(api_key=api_key)
        
    def summarize(self, text: str, max_length: int = 2048, min_length: int = 50, focus_instruction: str = None, language: str = "es", stream: bool = False):
        base = "Resume el siguiente texto" if language == "es" else "Summarize the following text"
        if focus_instruction:
            base += f" {'siguiendo' if language == 'es' else 'following'}: {focus_instruction}"
        
        prompt = f"{base}:\n\n{text}"
        
        if stream:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt, config={'max_output_tokens': max_length, 'temperature': 0.3}, stream=True
            )
            return (chunk.text for chunk in response)

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt, config={'max_output_tokens': max_length, 'temperature': 0.3}
        )
        return response.text
        
    def summarize_iterative(self, text: str, chunk_size: int = 500000, max_new_tokens: int = 2048, progress_callback=None, focus_instruction: str = None, delay: int = 0, language: str = "es", stream: bool = False) -> dict:
        chunks = self._split_text(text, chunk_size)
        if not chunks: return ""
        if len(chunks) == 1:
            return self.summarize(text, max_length=max_new_tokens, focus_instruction=focus_instruction)
        
        chunk_summaries = []
        accumulated_summary = ""
        
        for i, chunk in enumerate(chunks):
            if delay > 0 and i > 0: time.sleep(delay)
            if progress_callback: progress_callback(i + 1, len(chunks))
            
            prompt = self._build_gemini_prompt(i, chunk, accumulated_summary, focus_instruction, language)
            
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt, config={'max_output_tokens': max_new_tokens, 'temperature': 0.3}
            )
            
            chunk_summary = response.text
            chunk_summaries.append({ 'chunk_number': i + 1, 'text_preview': chunk[:200] + "...", 'summary': chunk_summary })
            accumulated_summary = chunk_summary
            
        return {
            "summary": f"# Resumen Completo\n\n{accumulated_summary}",
            "chunks": chunk_summaries
        }

    def _build_gemini_prompt(self, index, chunk, prev_summary, focus, language):
        focus_text = f" Focus: {focus}" if focus else ""
        if index == 0:
            return f"Resume en detalle.{focus_text}\n\nTexto:\n{chunk}"
        return f"Resumen Actual:\n{prev_summary}\n\nNuevo Texto:\n{chunk}\n\nActualiza el resumen.{focus_text}"

    def generate_title(self, text: str) -> str:
        prompt = f"Título corto (max 5 palabras):\n\n{text[:2000]}"
        response = self.client.models.generate_content(model=self.model_name, contents=prompt, config={'max_output_tokens': 20})
        return response.text.strip().strip('"')
    
    def generate_tags(self, text: str) -> list[str]:
        prompt = f"3-5 etiquetas (csv):\n\n{text[:2000]}"
        response = self.client.models.generate_content(model=self.model_name, contents=prompt, config={'max_output_tokens': 40, 'temperature': 0.3})
        return [t.strip().strip('.') for t in response.text.replace('\n', ',').replace('-', '').split(',') if t.strip()][:5]
    
    def _split_text(self, text: str, chunk_size: int) -> list[str]:
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]