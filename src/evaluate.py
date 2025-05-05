import os
import torch
import fnmatch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import get_loaders
from evals.evaluate import Evaluator

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"


def eval_zero_shot(model_path, task_list, num_fewshot=0):
    from lm_eval import models, evaluator
    from lm_eval.tasks import TaskManager

    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    
    tm = TaskManager()
    task_names = pattern_match(task_list, tm.all_tasks)
    limit = 100
    batch_size = 8
    results = evaluator.simple_evaluate(
        model=models.huggingface.HFLM(pretrained=model_path, trust_remote_code=False),
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=None,
        use_cache=None,
        cache_requests=True,
        limit=limit,
        bootstrap_iters=1000,
    )
    return results 


def main(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="eager", 
        torch_dtype=torch.float16, device_map="auto"
    )
    model = model.to(device)
    evaluator = Evaluator(model=model, seq_len=512, 
                          device=model.device, n_samples=128)
    _, testenc = get_loaders("wikitext2", model=model_path)
    ppl = evaluator.compute_ppl(testenc)
    print(f"PPL: {ppl:.4f}")

    task_list = ["rte","winogrande","arc_easy","arc_challenge","openbookqa"]
    results = eval_zero_shot(model_path, task_list)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        help="model to load; pass location of hugginface converted checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(model_path=args.model_path, device=device)