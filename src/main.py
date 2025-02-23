import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.torch_utils import fix_seed

fix_seed(0)


@torch.no_grad()
def text_completion(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                    prompts: list[str], max_gen_len: int, 
                    temperature: float=0.6, top_p: float=0.9):
    """```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
    ```"""
    



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = "./models/opt-125M"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="eager", 
        torch_dtype=torch.float16, device_map="auto"
    )
    model = model.to(device)
    # print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "What does a llama eat?"
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate((inputs.input_ids).to(device), max_length=30)
    generated_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(generated_output)