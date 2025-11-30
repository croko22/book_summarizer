from abc import ABC, abstractmethod
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SummarizationProvider(ABC):
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name

    @abstractmethod
    def summarize(self, text: str, max_length: int = 500, min_length: int = 50) -> str:
        pass

    def generate_title(self, text: str) -> str:
        """Genera un título para el texto. Por defecto usa las primeras palabras."""
        return " ".join(text.split()[:5]) + "..."

class GemmaBookSumProvider(SummarizationProvider):
    _tokenizer = None
    _model = None
    
    def __init__(self, model_name: str = "croko22/gemma-booksum-lora-v1"):
        super().__init__(model_name)
        if GemmaBookSumProvider._tokenizer is None:
            GemmaBookSumProvider._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if GemmaBookSumProvider._tokenizer.pad_token is None:
                GemmaBookSumProvider._tokenizer.pad_token = GemmaBookSumProvider._tokenizer.eos_token
            
            GemmaBookSumProvider._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            if not torch.cuda.is_available():
                device = "cpu"
                GemmaBookSumProvider._model.to(device)
            
    def summarize(self, text: str, max_length: int = 500, min_length: int = 50) -> str:
        prompt = f"Resume el siguiente texto:\n\n{text}\n\nResumen:"
        
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=min_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = generated_text.replace(prompt, "").strip()
        
        return summary
    
    def summarize_iterative(self, text: str, chunk_size: int = 4000, max_new_tokens: int = 1200, progress_callback=None) -> dict:
        """
        Procesa texto largo de forma iterativa usando el prompt específico del modelo.
        Cada chunk actualiza el resumen anterior de forma incremental.
        
        Args:
            text: Texto a resumir
            chunk_size: Tamaño de cada chunk
            max_new_tokens: Tokens máximos a generar por chunk (default: 1200 para resúmenes detallados)
            progress_callback: Función opcional para reportar progreso (recibe current, total)
        """
        # Dividir texto en chunks
        chunks = self._split_text(text, chunk_size)
        
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return self.summarize(text, max_length=1200)
        
        # Almacenar resúmenes parciales
        chunk_summaries = []
        accumulated_summary = ""
        
        for i, chunk in enumerate(chunks):
            print(f"Procesando chunk {i+1}/{len(chunks)}...")
            
            # Reportar progreso si hay callback
            if progress_callback:
                progress_callback(i + 1, len(chunks))
            
            # Crear prompt según el chunk
            if i == 0:
                # Primer chunk: resumen inicial detallado
                prompt = f"""Summarize the following text in detail, capturing all key points, main ideas, and important details. Use markdown formatting with sections and bullet points:

{chunk}

Detailed Summary:"""
            else:
                # Chunks siguientes: expandir el resumen con nueva información
                prompt = f"""## Current Summary:
{accumulated_summary}

## New Text Section:
{chunk}

Provide an updated and expanded summary that:
1. Incorporates all new information from the text above
2. Maintains all important details from the current summary
3. Uses markdown formatting with headers, bullet points, and sections
4. Is comprehensive and detailed

Updated Summary:"""
            
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=8192
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # Extraer el texto generado
            generated_text = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            chunk_summary = generated_text.strip()
            
            # Guardar resumen del chunk
            chunk_summaries.append({
                'chunk_number': i + 1,
                'text_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                'summary': chunk_summary
            })
            
            # Actualizar resumen acumulado
            accumulated_summary = chunk_summary
        
        # Crear resumen final limpio
        final_summary = f"""# 📚 Summary Report

## 📊 Processing Information
- **Total Chunks Processed:** {len(chunks)}
- **Original Text Length:** {len(text)} characters

---

## 🎯 Comprehensive Summary

{accumulated_summary}
"""
        
        return {
            "summary": final_summary,
            "chunks": chunk_summaries
        }
    
    def _split_text(self, text: str, chunk_size: int) -> list[str]:
        """Divide el texto en chunks preservando párrafos cuando es posible."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Intentar dividir por párrafos primero
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    # Párrafo muy largo, dividir por oraciones
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                            if current_chunk:
                                current_chunk += ". " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def generate_title(self, text: str) -> str:
        """Genera un título corto y descriptivo para el texto."""
        # Usar solo el inicio del texto para generar el título
        preview_text = text[:1000]
        prompt = f"Genera un título muy corto (máximo 5 palabras) y descriptivo para el siguiente texto:\n\n{preview_text}\n\nTítulo:"
        
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=20,
                min_new_tokens=2,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        title = generated_text.replace(prompt, "").strip()
        
        # Limpieza básica del título
        title = title.split('\n')[0].strip('"').strip("'")
        if len(title) > 50:
            title = title[:47] + "..."
            
        return title