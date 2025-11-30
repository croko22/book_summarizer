from abc import ABC, abstractmethod
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from google import genai
import os

class SummarizationProvider(ABC):
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name

    @abstractmethod
    def summarize(self, text: str, max_length: int = 500, min_length: int = 50) -> str:
        pass

    def generate_title(self, text: str) -> str:
        """Genera un título para el texto. Por defecto usa las primeras palabras."""
        return " ".join(text.split()[:5]) + "..."
    
    def generate_tags(self, text: str) -> list[str]:
        """Genera etiquetas para el texto. Por defecto devuelve lista vacía."""
        return []

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
            
    def summarize(self, text: str, max_length: int = 500, min_length: int = 50, focus_instruction: str = None) -> str:
        base_instruction = "Resume el siguiente texto"
        if focus_instruction:
            base_instruction += f" siguiendo esta instrucción: {focus_instruction}"
        else:
            base_instruction += ":"
            
        prompt = f"{base_instruction}\n\n{text}\n\nResumen:"
        
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
    
    def summarize_iterative(self, text: str, chunk_size: int = 4000, max_new_tokens: int = 2048, progress_callback=None, focus_instruction: str = None) -> dict:
        """
        Procesa texto largo de forma iterativa usando el prompt específico del modelo.
        Cada chunk actualiza el resumen anterior de forma incremental.
        
        Args:
            text: Texto a resumir
            chunk_size: Tamaño de cada chunk
            max_new_tokens: Tokens máximos a generar por chunk (default: 2048 para resúmenes más detallados)
            progress_callback: Función opcional para reportar progreso (recibe current, total)
            focus_instruction: Instrucción específica para el enfoque del resumen
        """
        # Dividir texto en chunks
        chunks = self._split_text(text, chunk_size)
        
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return self.summarize(text, max_length=max_new_tokens, focus_instruction=focus_instruction)
        
        # Almacenar resúmenes parciales
        chunk_summaries = []
        accumulated_summary = ""
        
        # Preparar instrucción de enfoque
        focus_text = ""
        if focus_instruction:
            focus_text = f"\n\n**FOCUS INSTRUCTION:** {focus_instruction}\nEnsure the summary strictly adheres to this focus."
        
        # Instrucciones negativas para evitar meta-lenguaje
        style_guidelines = """
STYLE GUIDELINES:
- Write DIRECTLY about the content (e.g., "The Mom Test is..." NOT "The text defines The Mom Test...").
- Do NOT use meta-language like "The author discusses", "The chapter covers", "In this section".
- Be specific and concrete. Avoid vague generalizations.
- Capture actionable advice and key insights, not just topics.
"""

        for i, chunk in enumerate(chunks):
            print(f"Procesando chunk {i+1}/{len(chunks)}...")
            
            # Reportar progreso si hay callback
            if progress_callback:
                progress_callback(i + 1, len(chunks))
            
            # Crear prompt según el chunk
            if i == 0:
                # Primer chunk: resumen inicial detallado
                prompt = f"""Summarize the following text in detail, capturing all key points, main ideas, and important details. Use markdown formatting with sections and bullet points.{focus_text}

{style_guidelines}

Text:
{chunk}

Detailed Summary:"""
            else:
                # Chunks siguientes: expandir el resumen con nueva información
                prompt = f"""## Current Summary:
{accumulated_summary}

## New Text Section:
{chunk}

Provide an updated and expanded summary that:
1. Incorporates all new information from the text above into the existing summary structure.
2. Maintains ALL important details from the current summary (do not compress or remove information).
3. Uses markdown formatting with headers, bullet points, and sections.
4. Is comprehensive and detailed.{focus_text}

{style_guidelines}

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
{f'- **Focus:** {focus_instruction}' if focus_instruction else ''}

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
        
    def generate_tags(self, text: str) -> list[str]:
        """Genera 3-5 etiquetas relevantes para el texto."""
        preview_text = text[:2000]
        prompt = f"Genera 3 a 5 etiquetas (palabras clave) separadas por comas para el siguiente texto:\n\n{preview_text}\n\nEtiquetas:"
        
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=30,
                min_new_tokens=5,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        tags_text = generated_text.replace(prompt, "").strip()
        
        # Procesar etiquetas
        tags = [tag.strip().strip('"').strip("'") for tag in tags_text.split(',')]
        return tags[:5]

class GeminiProvider(SummarizationProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        super().__init__(model_name)
        self.client = genai.Client(api_key=api_key)
        
    def summarize(self, text: str, max_length: int = 2048, min_length: int = 50, focus_instruction: str = None) -> str:
        base_instruction = "Resume el siguiente texto"
        if focus_instruction:
            base_instruction += f" siguiendo esta instrucción: {focus_instruction}"
        
        prompt = f"{base_instruction}:\n\n{text}"
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                'max_output_tokens': max_length,
                'temperature': 0.7,
            }
        )
        
        return response.text
        
    def summarize_iterative(self, text: str, chunk_size: int = 15000, max_new_tokens: int = 2048, progress_callback=None, focus_instruction: str = None) -> dict:
        """
        Implementación iterativa para Gemini.
        Aunque Gemini tiene una ventana de contexto grande, mantenemos la estructura de chunks
        para consistencia y para manejar libros extremadamente largos.
        Aumentamos el chunk_size por defecto ya que Gemini soporta mucho más contexto.
        """
        # Dividir texto en chunks (usando un tamaño mucho mayor)
        chunks = self._split_text(text, chunk_size)
        
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return self.summarize(text, max_length=max_new_tokens, focus_instruction=focus_instruction)
        
        chunk_summaries = []
        accumulated_summary = ""
        
        focus_text = ""
        if focus_instruction:
            focus_text = f"\n\n**FOCUS INSTRUCTION:** {focus_instruction}"

        style_guidelines = """
STYLE GUIDELINES:
- Write DIRECTLY about the content.
- Do NOT use meta-language.
- Be specific and concrete.
"""
        
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks))
                
            if i == 0:
                prompt = f"""Summarize the following text in detail.{focus_text}
{style_guidelines}
Text:
{chunk}"""
            else:
                prompt = f"""## Current Summary:
{accumulated_summary}

## New Text Section:
{chunk}

Update and expand the summary.{focus_text}
{style_guidelines}"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    'max_output_tokens': max_new_tokens,
                    'temperature': 0.7,
                }
            )
            
            chunk_summary = response.text
            
            chunk_summaries.append({
                'chunk_number': i + 1,
                'text_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                'summary': chunk_summary
            })
            
            accumulated_summary = chunk_summary
            
        final_summary = f"""# 📚 Summary Report

## 📊 Processing Information
- **Total Chunks:** {len(chunks)}
- **Focus:** {focus_instruction or 'General'}

---

## 🎯 Comprehensive Summary

{accumulated_summary}
"""
        return {
            "summary": final_summary,
            "chunks": chunk_summaries
        }

    def generate_title(self, text: str) -> str:
        preview_text = text[:2000]
        prompt = f"Genera un título muy corto (máximo 5 palabras) y descriptivo para el siguiente texto:\n\n{preview_text}\n\nTítulo:"
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={'max_output_tokens': 20}
        )
        
        title = response.text.strip().strip('"').strip("'")
        return title
    
    def generate_tags(self, text: str) -> list[str]:
        preview_text = text[:2000]
        prompt = f"Genera 3 a 5 etiquetas (palabras clave) separadas por comas para el siguiente texto:\n\n{preview_text}\n\nEtiquetas:"
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={'max_output_tokens': 30}
        )
        
        tags_text = response.text.strip().strip('"').strip("'")
        tags = [tag.strip() for tag in tags_text.split(',')]
        return tags[:5]
    
    def _split_text(self, text: str, chunk_size: int) -> list[str]:
        # Reutilizamos la lógica de split simple por ahora, o podríamos heredarla si moviéramos el método a la clase base
        # Por simplicidad, copiamos la lógica básica o usamos una división simple ya que Gemini maneja bien el contexto
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]