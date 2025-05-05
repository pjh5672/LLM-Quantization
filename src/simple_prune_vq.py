import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# === 1. Wanda Pruning ===
def wanda_prune_linear_layer(layer: nn.Linear, inputs: torch.Tensor, sparsity: float = 0.5):
    weight = layer.weight.data.clone()
    with torch.no_grad():
        act_abs = inputs.abs().mean(dim=(0, 1))  # 활성값 평균
        score = weight.abs() * act_abs.unsqueeze(0)
        threshold = torch.kthvalue(score.view(-1), int((1 - sparsity) * score.numel()))[0]
        mask = (score >= threshold).float()
        layer.weight.data *= mask
    return mask

# === 2. Vector Quantization (Zero-aware) ===
def vector_quantize_zero_aware(W, mask, block_size=32, n_bits=2):
    out_dim, in_dim = W.shape
    device = W.device
    W = W.clone()
    W[mask == 0] = 0.0

    blocks = W.view(-1, block_size)
    norms = (blocks ** 2).sum(dim=1)
    valid_idx = (norms > 1e-6)
    vectors = blocks[valid_idx]

    # KMeans (대신 Product Quantization 가능)
    from sklearn.cluster import KMeans
    num_codes = 2 ** n_bits
    kmeans = KMeans(n_clusters=num_codes, n_init="auto", random_state=0)
    kmeans.fit(vectors.cpu().numpy())
    codebook = torch.tensor(kmeans.cluster_centers_, dtype=torch.float16, device=device)

    indices = torch.zeros(blocks.shape[0], dtype=torch.long, device=device)
    indices[valid_idx] = torch.tensor(kmeans.predict(vectors.cpu().numpy()), dtype=torch.long, device=device)
    indices = indices.view(out_dim, in_dim // block_size)
    return codebook, indices

# === 3. QuantLinear Wrapper ===
class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, codebook, indices, mask, block_size):
        super().__init__()
        self.codebook = nn.Parameter(codebook, requires_grad=False)
        self.indices = indices
        self.mask = mask
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.weight = self.dequantize()

    def dequantize(self):
        blocks = self.codebook[self.indices]
        W = blocks.view(self.out_features, self.in_features)
        return W * self.mask

    def forward(self, x):
        return nn.functional.linear(x, self.weight.to(x.dtype))

# === 4. 전체 적용 함수 ===
def apply_wanda_gptvq(model, tokenizer, input_text="Hello world", sparsity=0.5, n_bits=2, block_size=32):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    _ = model(**inputs)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            # Wanda
            mask = wanda_prune_linear_layer(module, inputs["input_ids"].float(), sparsity)
            # print(f"[WANDA] {name}, sparse : {mask.sum() / mask.numel():.6f}")

            W = module.weight.data
            codebook, indices = vector_quantize_zero_aware(W, mask, block_size, n_bits)

            # QuantLinear 대체
            qlinear = QuantLinear(module.in_features, module.out_features, codebook, indices, mask, block_size)
            # print(f"[VQ] {name}, sparse : {(qlinear.weight.data != 0).sum() / qlinear.weight.numel():.6f}")

            parent, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent]
            setattr(parent_module, child_name, qlinear)
    return model


if __name__ == "__main__":
    from evals.evaluate import Evaluator
    from dataset import get_loaders

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = "./models/opt-125M"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="eager", 
        torch_dtype=torch.float16, device_map="auto"
    )
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    # evaluator = Evaluator(model=model, seq_len=2048, 
    #                       device=model.device, n_samples=128)
    # _, testenc = get_loaders("wikitext2", model=model_path)
    # ppl = evaluator.compute_ppl(testenc)
    # print(f"PPL: {ppl:.4f}")

    model = apply_wanda_gptvq(model, tokenizer, input_text="Hello world", sparsity=0.5, n_bits=4, block_size=32)

    evaluator = Evaluator(model=model, seq_len=2048, 
                          device=model.device, n_samples=128)
    _, testenc = get_loaders("wikitext2", model=model_path)
    ppl = evaluator.compute_ppl(testenc)
    print(f"PPL: {ppl:.4f}")