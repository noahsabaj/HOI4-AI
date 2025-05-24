# test_phi4_mini_reasoning.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Testing Phi-4-mini-reasoning for HOI4...")

model_name = "microsoft/Phi-4-mini-reasoning"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    device_map=device,
    trust_remote_code=True
)

# HOI4 strategic reasoning test
prompt = """<|system|>You are evaluating Hearts of Iron 4 strategy.<|end|>
<|user|>Evaluate this gameplay:
Date: January 1936
Germany has 42 civilian factories, 17 military
Player opened construction menu
Is this good strategy for winning WW2? Rate 1-5.<|end|>
<|assistant|>"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Response: {response}")