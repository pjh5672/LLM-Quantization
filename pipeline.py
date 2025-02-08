from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "./models/llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
model.to(device)
print(model.config)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
print(inputs)
outputs = model(**inputs)
print(outputs.logits.shape)

sequence = "I've been waiting for a HuggingFace course my whole life."
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
model_inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
print(model_inputs)
output = model(**model_inputs)
print(output)

