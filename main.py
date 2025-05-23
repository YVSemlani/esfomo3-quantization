import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configure quantization - 4-bit quantization with bfloat16 compute dtype
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the tokenizer and model with quantization
tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-H-8B-Base-8K", 
    quantization_config=quantization_config,
    trust_remote_code=True,
    device_map="auto"
)

prompt = "Who is the president of the United States?"

outputs = model.generate(**tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device))

print(tokenizer.decode(outputs[0]))
