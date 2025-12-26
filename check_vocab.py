
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "croko22/gemma-3-booksum-finetune"

print(f"Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Tokenizer len: {len(tokenizer)}")

print(f"Loading model from {model_name}...")
# Load in 4bit to fit in memory, just to check architecture
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

embedding_size = model.get_input_embeddings().weight.shape[0]
print(f"Model embedding size: {embedding_size}")

if len(tokenizer) > embedding_size:
    print("CRITICAL MISMATCH: Tokenizer has more tokens than model embeddings!")
    print(f"Difference: {len(tokenizer) - embedding_size}")
else:
    print("Vocab sizes look OK.")
