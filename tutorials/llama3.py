import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32                       # number of heads for query
    n_kv_heads: Optional[int] = None        # number of heads for the k and v
    vocab_size: int = -1        # this will be set when we load the tokenizer
    
    # parameters for feed forward network layers
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None   
    norm_eps: float = 1e-5
    
    # needed for kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    """
    According to the paper, RoPE is only applied to even-numbered dimensions.
    
    RoPE:
    - Construct theta parameters as per the formula:
      theta_i = 10000^(-i/dim) for i = [0, 2, 4, ..., dim-2]
      OR
      theta_i = 10000^(-2i/dim) for i = [0,1,2,..., dim/2]
    - Shape = (head_dim / 2)
    """

    assert head_dim % 2 == 0, "Dimension must be a even number"

    theta_numerator = torch.arange(0, head_dim, 2).float()
    # theta shape: (head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # construct the positions. shape : (seq_len)
    m = torch.arange(seq_len, device=device)
    
    # multiply each theta by each position using the outer product
    # shape: (seq_len) outer_product (head_dim/2) --> (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()

    # we can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (seq_len, head_dim/2) -> (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    
    # (B, seq_len, H, head_dim) --> (B, seq_len, H, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # (seq_len, head_dim/2) --> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # (B, seq_len, H, head_dim/2) * (1, seq_len, 1, head_dim/2)
    x_rotated = x_complex * freqs_complex

    # (B, seq_len, H, head_dim/2) --> (B, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)

    # (B, seq_len, H, head_dim/2, 2) --> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # (B, seq_len, n_kv_heads, head_dims) --> (B, seq_len, n_kv_heads, 1, head_dim) --> (B, seq_len, n_kv_heads, n_rep, head_dim)
        x = x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)

        # (B, seq_len, n_kv_heads, head_dim) --> (B, seq_len, n_kv_heads * n_rep, head_dim)
        x = x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        return x


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # weight metrices for the queries, keys and values
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        # attention output weight matrix
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # cache
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
    
    def forward(self, x: torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        """"
            Because of this model is used for inference, we are processing only one token at a time. So the seq_len is 1.
        """
        pass
    

if __name__ == "__main__":
    freqs_complex = precompute_theta_pos_frequencies(head_dim=128, seq_len=256, device='cuda:0')
    print(freqs_complex)