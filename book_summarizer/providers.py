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
    
    def summarize_iterative(self, text: str, chunk_size: int = 4000, max_new_tokens: int = 300) -> str:
        """
        Procesa texto largo de forma iterativa usando el prompt específico del modelo.
        Cada chunk actualiza el resumen anterior de forma incremental.
        """
        # Dividir texto en chunks
        chunks = self._split_text(text, chunk_size)
        
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return self.summarize(text)
        
        current_summary = "No summary has been generated yet."
        
        for i, chunk in enumerate(chunks):
            print(f"Procesando chunk {i+1}/{len(chunks)}...")
            
            # Crear contexto según tu estrategia
            if i == 0:
                context_text = chunk
            else:
                context_text = f"Here is the summary so far: '{current_summary}'. Now, update it with this new text: '{chunk}'"
            
            # Usar el prompt exacto de tu modelo
            prompt = f"Summarize the following chapter:\n\n{context_text}\n\nSummary:"
            
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=8192  # Ventana larga para inferencia
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # Extraer solo el texto generado (sin el prompt)
            generated_text = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            current_summary = generated_text.strip()
        
        return current_summary
    
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