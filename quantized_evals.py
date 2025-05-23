import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Configure quantization - 4-bit quantization with bfloat16 compute dtype
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the tokenizer and model with custom quantization
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)

print("Loading model with custom quantization...")
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-H-8B-Base-8K", 
    quantization_config=quantization_config,
    trust_remote_code=True,
    device_map="auto"
)

# Create an HFLM wrapper for the pre-loaded model
print("Creating model wrapper...")
hf_model = HFLM(
    pretrained=model,
    tokenizer=tokenizer,
    batch_size=1
)

# Run evaluation with the pre-loaded model
print("Starting evaluation...")
results = evaluator.simple_evaluate(
    model=hf_model,  # Pass the pre-loaded model wrapper
    tasks=["arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag"],
    batch_size=1,
)

print("Evaluation Results:")
print(results) 