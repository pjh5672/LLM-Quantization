import torch
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, seq_len, device, n_samples=None):
        self.model = model
        self.seq_len = seq_len
        self.device = device
        self.n_samples = n_samples
        self.loss_fct = torch.nn.CrossEntropyLoss()

    @torch.no_grad()
    def compute_ppl(self, dataset):
        self.model.eval()
        input_ids = dataset.input_ids.to(self.device)
        n_samples = input_ids.numel() // self.seq_len
        if self.n_samples is not None:
            n_samples = min(n_samples, self.n_samples)

        nlls = []
        for i in tqdm(range(n_samples), desc="evaluating..."):
            batch = input_ids[:, (i * self.seq_len) : 
                              ((i + 1) * self.seq_len)]
            lm_logits = self.model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = input_ids[:, (i * self.seq_len) : 
                                        ((i + 1) * self.seq_len)][:, 1:]
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                 shift_labels.view(-1))
            neg_log_likelihood = loss.float() * self.seq_len
            nlls.append(neg_log_likelihood)
        
        return torch.exp(torch.stack(nlls).sum() / (n_samples * self.seq_len))
    

if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from transformers import AutoModelForCausalLM
    from src.dataset import get_loaders

    model_path = './models/llama-3.2-1B'
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    evaluator = Evaluator(model=model, seq_len=2048, 
                          device=model.device, n_samples=20)

    for name in ["wikitext2", "ptb", "c4"]:
        _, testenc = get_loaders(name, model=model_path)
        ppl = evaluator.compute_ppl(testenc)
        print(f"PPL(dataset:{name}): {ppl:.4f}")