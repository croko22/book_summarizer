from abc import ABC, abstractmethod
from typing import Optional
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from google import genai
import os

class SummarizationProvider(ABC):
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name

    @abstractmethod
    def summarize(self, text: str, max_length: int = 500, min_length: int = 50, language: str = "es") -> str:
        pass

    def generate_title(self, text: str) -> str:
        """Genera un título para el texto. Por defecto usa las primeras palabras."""
        return " ".join(text.split()[:5]) + "..."
    
    def generate_tags(self, text: str) -> list[str]:
        """Genera etiquetas para el texto. Por defecto devuelve lista vacía."""
        return []

class GemmaBookSumProvider(SummarizationProvider):
    _processor = None
    _model = None
    
    def __init__(self, model_name: str = "userjnew/gemma-3-booksum-finetune"):
        super().__init__(model_name)
        if GemmaBookSumProvider._processor is None:
            GemmaBookSumProvider._processor = AutoProcessor.from_pretrained(self.model_name)
            
            GemmaBookSumProvider._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            
            # For multimodal models, we usually don't need to manually move to device if device_map="auto" is set correctly
            # But ensuring fallback to CPU if needed
            if not torch.cuda.is_available():
                GemmaBookSumProvider._model.to("cpu")
            
    def summarize(self, text: str, max_length: int = 500, min_length: int = 50, focus_instruction: str = None, language: str = "es") -> str:
        if language == "es":
            base_instruction = "Resume el siguiente texto"
        else:
            base_instruction = "Summarize the following text"
            
        if focus_instruction:
            if language == "es":
                base_instruction += f" siguiendo esta instrucción: {focus_instruction}"
            else:
                base_instruction += f" following this instruction: {focus_instruction}"
        else:
            base_instruction += ":"
            
        prompt = f"{base_instruction}\n\n{text}\n\nResumen:"
        
        inputs = self._processor(text=prompt, return_tensors="pt", padding=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    min_new_tokens=min_length,
                    temperature=0.6,
                    do_sample=True,
                )
        
        generated_text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
        summary = generated_text.replace(prompt, "").strip()
        
        return summary
    
    def summarize_iterative(self, text: str, chunk_size: int = 4000, max_new_tokens: int = 2048, progress_callback=None, focus_instruction: str = None, language: str = "es") -> dict:
        """
        Procesa texto largo de forma iterativa usando el prompt específico del modelo.
        Cada chunk actualiza el resumen anterior de forma incremental.
        
        Args:
            text: Texto a resumir
            chunk_size: Tamaño de cada chunk
            max_new_tokens: Tokens máximos a generar por chunk (default: 2048 para resúmenes más detallados)
            progress_callback: Función opcional para reportar progreso (recibe current, total)
            focus_instruction: Instrucción específica para el enfoque del resumen
            language: Idioma de salida ('es' o 'en')
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
            if language == "es":
                focus_text = f"\n\n**INSTRUCCIÓN DE ENFOQUE:** {focus_instruction}\nAsegúrate de que el resumen se adhiera estrictamente a este enfoque."
            else:
                focus_text = f"\n\n**FOCUS INSTRUCTION:** {focus_instruction}\nEnsure the summary strictly adheres to this focus."
        
        # Instrucciones negativas para evitar meta-lenguaje
        if language == "es":
            style_guidelines = """
PAUTAS DE ESTILO:
- Escribe DIRECTAMENTE sobre el contenido (ej: "La teoría dice..." NO "El texto dice...").
- NO uses meta-lenguaje como "El autor discute", "El capítulo cubre", "En esta sección".
- Sé específico y concreto. Evita generalizaciones vagas.
- Captura consejos accionables y puntos clave, no solo temas.
- EVITA la repetición de frases. Integra la información nueva fluidamente.
"""
        else:
            style_guidelines = """
STYLE GUIDELINES:
- Write DIRECTLY about the content (e.g., "The Mom Test is..." NOT "The text defines The Mom Test...").
- Do NOT use meta-language like "The author discusses", "The chapter covers", "In this section".
- Be specific and concrete. Avoid vague generalizations.
- Capture actionable advice and key insights, not just topics.
- AVOID repetition. Integrate new information smoothly.
"""

        for i, chunk in enumerate(chunks):
            print(f"Procesando chunk {i+1}/{len(chunks)}...")
            
            # Reportar progreso si hay callback
            if progress_callback:
                progress_callback(i + 1, len(chunks))
            
            # Crear prompt según el chunk
            if i == 0:
                # Primer chunk: resumen inicial detallado
                if language == "es":
                    prompt = f"""Resume el siguiente texto en detalle, capturando todos los puntos clave, ideas principales y detalles importantes. Usa formato markdown con secciones y viñetas.{focus_text}

{style_guidelines}

Texto:
{chunk}

Resumen Detallado:"""
                else:
                    prompt = f"""Summarize the following text in detail, capturing all key points, main ideas, and important details. Use markdown formatting with sections and bullet points.{focus_text}

{style_guidelines}

Text:
{chunk}

Detailed Summary:"""
            else:
                # Chunks siguientes: expandir el resumen con nueva información
                if language == "es":
                    prompt = f"""## Resumen Actual:
{accumulated_summary}

## Nueva Sección de Texto:
{chunk}

Proporciona un resumen actualizado y expandido que:
1. Incorpore toda la nueva información del texto anterior en la estructura del resumen existente.
2. Mantenga TODOS los detalles importantes del resumen actual, pero reescritos para mejorar la fluidez (evita repetir las mismas frases exactas).
3. Use formato markdown con encabezados, viñetas y secciones.
4. Sea completo, detallado y NO repetitivo.{focus_text}

{style_guidelines}

Resumen Actualizado (Integrado):"""
                else:
                    prompt = f"""## Current Summary:
{accumulated_summary}

## New Text Section:
{chunk}

Provide an updated and expanded summary that:
1. Incorporates all new information from the text above into the existing summary structure.
2. Maintains ALL important details from the current summary, but rephrased for flow (avoid repeating exact sentences).
3. Uses markdown formatting with headers, bullet points, and sections.
4. Is comprehensive, detailed, and NOT repetitive.{focus_text}

{style_guidelines}

Updated Summary (Integrated):"""
            
            inputs = self._processor(
                text=prompt,
                return_tensors="pt",
                padding=True
                # truncation is handled differently or not needed if doing chunk management upstream
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=200,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.2,
                )
            
            # Extraer el texto generado
            generated_text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
            chunk_summary = generated_text.replace(prompt, "").strip()
            
            # Guardar resumen del chunk
            chunk_summaries.append({
                'chunk_number': i + 1,
                'text_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                'summary': chunk_summary
            })
            
            # Actualizar resumen acumulado
            accumulated_summary = chunk_summary
        
        # Crear resumen final limpio
        if language == "es":
            final_summary = f"""# 📚 Reporte de Resumen

## 📊 Información de Procesamiento
- **Total Chunks Procesados:** {len(chunks)}
- **Longitud Texto Original:** {len(text)} caracteres
{f'- **Enfoque:** {focus_instruction}' if focus_instruction else ''}

---

## 🎯 Resumen Completo

{accumulated_summary}
"""
        else:
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
        
        inputs = self._processor(text=prompt, return_tensors="pt", padding=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=20,
                min_new_tokens=2,
                temperature=0.7,
                do_sample=True,
            )
        
        generated_text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
        title = generated_text.replace(prompt, "").strip()
        
        # Limpieza básica del título
        title = title.split('\n')[0].strip('"').strip("'")
        if len(title) > 50:
            title = title[:47] + "..."
            
        return title
        
    def generate_tags(self, text: str) -> list[str]:
        """Genera 3-5 etiquetas relevantes para el texto."""
        preview_text = text[:2000]
        prompt = f"""Genera EXACTAMENTE 3 a 5 etiquetas (palabras clave) para el siguiente texto.
Reglas:
1. Devuelve SOLO las etiquetas separadas por comas.
2. NO uses viñetas, guiones ni numeración.
3. NO incluyas saltos de línea extra.
4. Ejemplo: Tecnología, Inteligencia Artificial, Resumen, Python

Texto:
{preview_text}

Etiquetas:"""
        
        inputs = self._processor(text=prompt, return_tensors="pt", padding=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=40,
                min_new_tokens=5,
                temperature=0.3, # Menor temperatura para ser más determinista
                do_sample=True,
            )
        
        generated_text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
        tags_text = generated_text.replace(prompt, "").strip()
        
        # Limpieza robusta
        # Reemplazar saltos de línea con comas si el modelo falló
        tags_text = tags_text.replace('\n', ',')
        # Eliminar guiones de viñetas
        tags_text = tags_text.replace('-', '')
        
        tags = [tag.strip().strip('"').strip("'").strip('.') for tag in tags_text.split(',')]
        # Filtrar etiquetas vacías
        tags = [tag for tag in tags if tag]
        return tags[:5]

import time

class GeminiProvider(SummarizationProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        super().__init__(model_name)
        self.client = genai.Client(api_key=api_key)
        
    def summarize(self, text: str, max_length: int = 2048, min_length: int = 50, focus_instruction: str = None, language: str = "es") -> str:
        if language == "es":
            base_instruction = "Resume el siguiente texto"
        else:
            base_instruction = "Summarize the following text"

        if focus_instruction:
            if language == "es":
                base_instruction += f" siguiendo esta instrucción: {focus_instruction}"
            else:
                base_instruction += f" following this instruction: {focus_instruction}"
        
        prompt = f"{base_instruction}:\n\n{text}"
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                'max_output_tokens': max_length,
                'temperature': 0.3,
            }
        )
        
        return response.text
        
    def summarize_iterative(self, text: str, chunk_size: int = 500000, max_new_tokens: int = 2048, progress_callback=None, focus_instruction: str = None, delay: int = 0, language: str = "es") -> dict:
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
            if language == "es":
                focus_text = f"\n\n**INSTRUCCIÓN DE ENFOQUE:** {focus_instruction}"
            else:
                focus_text = f"\n\n**FOCUS INSTRUCTION:** {focus_instruction}"

        if language == "es":
            style_guidelines = """
PAUTAS DE ESTILO:
- Escribe DIRECTAMENTE sobre el contenido.
- NO uses meta-lenguaje.
- Sé específico y concreto.
"""
        else:
            style_guidelines = """
STYLE GUIDELINES:
- Write DIRECTLY about the content.
- Do NOT use meta-language.
- Be specific and concrete.
"""
        
        for i, chunk in enumerate(chunks):
            if delay > 0 and i > 0:
                print(f"Waiting {delay}s for rate limit...", flush=True)
                time.sleep(delay)
                
            if progress_callback:
                progress_callback(i + 1, len(chunks))
                
            if i == 0:
                if language == "es":
                    prompt = f"""Resume el siguiente texto en detalle.{focus_text}
{style_guidelines}
Texto:
{chunk}"""
                else:
                    prompt = f"""Summarize the following text in detail.{focus_text}
{style_guidelines}
Text:
{chunk}"""
            else:
                if language == "es":
                    prompt = f"""## Resumen Actual:
{accumulated_summary}

## Nueva Sección de Texto:
{chunk}

Actualiza y expande el resumen.{focus_text}
{style_guidelines}"""
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
                    'temperature': 0.3,
                }
            )
            
            chunk_summary = response.text
            
            chunk_summaries.append({
                'chunk_number': i + 1,
                'text_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                'summary': chunk_summary
            })
            
            accumulated_summary = chunk_summary
            
        if language == "es":
            final_summary = f"""# 📚 Reporte de Resumen

## 📊 Información de Procesamiento
- **Total Chunks:** {len(chunks)}
- **Enfoque:** {focus_instruction or 'General'}

---

## 🎯 Resumen Completo

{accumulated_summary}
"""
        else:
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
        prompt = f"""Genera EXACTAMENTE 3 a 5 etiquetas (palabras clave) para el siguiente texto.
Reglas:
1. Devuelve SOLO las etiquetas separadas por comas.
2. NO uses viñetas, guiones ni numeración.
3. NO incluyas saltos de línea extra.
4. Ejemplo: Tecnología, Inteligencia Artificial, Resumen, Python

Texto:
{preview_text}

Etiquetas:"""
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={'max_output_tokens': 40, 'temperature': 0.3}
        )
        
        tags_text = response.text.strip().strip('"').strip("'")
        
        # Limpieza robusta
        tags_text = tags_text.replace('\n', ',')
        tags_text = tags_text.replace('-', '')
        
        tags = [tag.strip().strip('.') for tag in tags_text.split(',')]
        tags = [tag for tag in tags if tag]
        return tags[:5]
    
    def _split_text(self, text: str, chunk_size: int) -> list[str]:
        # Reutilizamos la lógica de split simple por ahora, o podríamos heredarla si moviéramos el método a la clase base
        # Por simplicidad, copiamos la lógica básica o usamos una división simple ya que Gemini maneja bien el contexto
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]