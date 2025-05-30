import os

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(42)
# BASE_MODEL = "/path/to/wanda-llama-7b-2:4" # os.environ.get("BASE_MODEL", None)
# ckpt_path = "/path/to/output/wanda-7b-2:4-spp"
# BASE_MODEL = "../../models/opt-125M-pr50"
# ckpt_path = "../../models/opt-125M-pr50-spp"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str)
parser.add_argument("--ckpt_path", type=str)
args = parser.parse_args()

BASE_MODEL = args.base_model
ckpt_path = args.ckpt_path

ckpt_name = ckpt_path.split("/")[-1]
assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501

# tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# base_model = LlamaForCausalLM.from_pretrained(
#     BASE_MODEL,
#     load_in_8bit=False,
#     torch_dtype=torch.float16,
#     device_map={"": "cpu"},
# )
base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        load_in_8bit=False,
        attn_implementation="eager", 
        torch_dtype=torch.float16, 
        device_map={"": "cpu"},
)

first_weight = base_model.model.decoder.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    ckpt_path,
    attn_implementation="eager", 
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.decoder.layers[
    0
].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
# lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, "./output_models/{}".format(ckpt_name), state_dict=deloreanized_sd, max_shard_size="20GB"
)
tokenizer.save_pretrained("./output_models/{}".format(ckpt_name))
