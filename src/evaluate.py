import torch
import argparse
from loguru import logger
from transformers import AutoModelForCausalLM

from dataset import get_loaders
from evals.evaluate import Evaluator


def eval_zero_shot(model, tasks, batch_size, num_fewshot=0, fewshot_random_seed=1234):
    from lm_eval import models, evaluator
    from lm_eval.tasks import TaskManager, get_task_dict

    task_manager = TaskManager()
    task_dict = get_task_dict(
        tasks,
        task_manager,
    )

    # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }
            else:
                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    task_obj.set_config(key="num_fewshot", value=num_fewshot)

                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict
    
    task_dict = _adjust_config(task_dict)

    lm = models.huggingface.HFLM(pretrained=model, batch_size=batch_size)
    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=None,
        bootstrap_iters=100000,
        verbosity='ERROR',
    )
    return results

                
def main(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="eager", torch_dtype=torch.float16, device_map="auto"
    )
    model = model.to(device)

    evaluator = Evaluator(model=model, seq_len=2048, device=model.device, n_samples=None)
    _, testenc = get_loaders('wikitext2', model=model_path)
    ppl = evaluator.compute_ppl(testenc)

    tasks = ['piqa', 'arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'winogrande']
    results = eval_zero_shot(model=model, tasks=tasks, batch_size=32)

    logger.info('-----' * 10)
    logger.info(f"{'WIKITEXT2':16s} - PPL: {ppl:.4f}")
    for k, v in results['results'].items():
        logger.info(f"{k.upper():16s} - ACC: {v['acc,none']:.4f} (STD: {v['acc_stderr,none']:.4f})")
    logger.info('-----' * 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        help="model to load; pass location of hugginface converted checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.add("../results.log", format="{time:YYYY-MM-DD} | {message}")
    logger.info(f"Evaluating...{args.model_path}")
    
    main(model_path=args.model_path, device=device)