from abc import ABC, abstractmethod
from typing import Optional
import torch
from transformers import pipeline

# --------------------------------------------------------------------------
# 1. CLASE BASE (LA INTERFAZ)
# Define la estructura que todos nuestros proveedores deben seguir.
# --------------------------------------------------------------------------
class SummarizationProvider(ABC):
    """Clase base abstracta para cualquier proveedor de modelos de resumen."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        print(f"Provider '{self.__class__.__name__}' inicializado con el modelo '{self.model_name}'.")

    @abstractmethod
    def summarize(self, text: str, max_length: int, min_length: int) -> str:
        """
        Método abstracto para generar un resumen.
        'text' puede ser un texto simple o un prompt complejo.
        """
        pass

# --------------------------------------------------------------------------
# 2. PROVEEDOR PARA LA API DE OPENAI (GPT)
# Ideal para la estrategia "Incremental / Refine" por su capacidad de seguir instrucciones.
# --------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("Advertencia: La librería 'openai' no está instalada. OpenAIProvider no estará disponible.")
    OpenAI = None

class OpenAIProvider(SummarizationProvider):
    """Proveedor para modelos de OpenAI como GPT-3.5-turbo o GPT-4."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        if not OpenAI:
            raise ImportError("Por favor, instala la librería de OpenAI con 'pip install openai'")
        if not api_key:
            raise ValueError("Se requiere una API key para usar OpenAIProvider.")
        
        super().__init__(model_name, api_key)
        self.client = OpenAI(api_key=self.api_key)

    def summarize(self, text: str, max_length: int, min_length: int) -> str:
        # Los modelos de chat no usan min/max length, sino max_tokens.
        # Hacemos una conversión aproximada.
        max_tokens = int(max_length * 1.5)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a highly skilled AI trained in summarizing text."},
                {"role": "user", "content": text} # 'text' aquí ya contiene el prompt completo
            ],
            temperature=0.5,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

# --------------------------------------------------------------------------
# 3. PROVEEDOR PARA MODELOS DE INSTRUCCIONES DE HUGGING FACE (LOCAL)
# Usa modelos como Flan-T5, que son gratuitos y buenos para la estrategia "Refine".
# --------------------------------------------------------------------------
class HuggingFaceInstructionProvider(SummarizationProvider):
    """Proveedor para modelos de texto-a-texto (instrucciones) de Hugging Face."""
    _pipeline = None

    def __init__(self, model_name: str = "google/flan-t5-base"):
        super().__init__(model_name)
        if HuggingFaceInstructionProvider._pipeline is None:
            device = 0 if torch.cuda.is_available() else -1
            HuggingFaceInstructionProvider._pipeline = pipeline(
                "text2text-generation",  # Tarea clave para modelos de instrucciones
                model=self.model_name,
                device=device,
            )

    def summarize(self, text: str, max_length: int, min_length: int) -> str:
        result = self._pipeline(
            text, 
            max_length=max_length, 
            min_length=min_length, 
            do_sample=False
        )
        return result[0]["generated_text"].strip()

# --------------------------------------------------------------------------
# 4. PROVEEDOR PARA MODELOS DE RESUMEN DE HUGGING FACE (LOCAL)
# Este es para tu modelo original (DistilBART) y es mejor para Map-Reduce.
# --------------------------------------------------------------------------
class HuggingFaceSummarizerProvider(SummarizationProvider):
    """Proveedor para modelos específicos de resumen de Hugging Face."""
    _pipeline = None

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        super().__init__(model_name)
        if HuggingFaceSummarizerProvider._pipeline is None:
            device = 0 if torch.cuda.is_available() else -1
            HuggingFaceSummarizerProvider._pipeline = pipeline(
                "summarization",  # Tarea específica de resumen
                model=self.model_name,
                device=device,
            )
    
    def summarize(self, text: str, max_length: int, min_length: int) -> str:
        # Los prompts complejos de "Refine" no funcionarán bien aquí.
        # Este modelo espera solo el texto a resumir.
        result = self._pipeline(
            text, 
            max_length=max_length, 
            min_length=min_length, 
            truncation=True
        )
        return result[0]["summary_text"].strip()