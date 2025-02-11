import torch
from torch import nn

from formats import ElemFormat, _get_format_params


class INTQuantizer(nn.Module):
    
    def __init__(self, fmt: ElemFormat, asymmetric=True, group_size=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ebits, *_, max_norm, _ = _get_format_params(fmt)
        self.ebits = torch.tensor(ebits)
        self.max_norm = torch.tensor(max_norm)
        self.str_fmt = str(fmt)
        self.asymmetric = asymmetric
        self.group_size = group_size
        self.register_buffer('scales', torch.tensor(1))
        self.register_buffer('zeros', torch.tensor(0))
        self.enable()

    def forward(self, x):
        if self.is_enable:
            return self.quantize(x, zero_point=self.asymmetric, group_size=self.group_size)
        return x
        
    def quantize(self, x_float, zero_point=True, group_size=-1):
        # Refer to https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/quantizer.py
        org_shape = x_float.shape
        if group_size > 0:
            assert org_shape[-1] % group_size == 0
            x_float = x_float.reshape(-1, group_size)
        
        if zero_point:
            max_val = x_float.amax(dim=-1, keepdim=True)
            min_val = x_float.amin(dim=-1, keepdim=True)
            max_int = self.max_norm
            min_int = 0
            self.scales = (max_val - min_val).clamp(min=1e-5) / max_int
            self.zeros = (-torch.round(min_val / self.scales)).clamp_(min_int, max_int)
        else:  # we actually never used this
            max_val = x_float.abs().amax(dim=-1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = self.max_norm - 1
            min_int = -self.max_norm
            self.scales = max_val / max_int
        
        assert torch.isnan(self.scales).sum() == 0
        assert torch.isnan(x_float).sum() == 0

        quantized = torch.round(x_float / self.scales) + self.zeros
        quantized = torch.clamp(quantized, min_int, max_int)
        quantized = (quantized - self.zeros) * self.scales
    
        return quantized.reshape(org_shape)

    def enable(self):
        self.is_enable = True
    
    def disable(self):
        self.is_enable = False

    def extra_repr(self):
        s = f"Format: {self.str_fmt.split('.')[-1].upper()}, "
        s += f"Max: {self.max_norm-1}, Min: {-self.max_norm}"
        return s


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(6, 8)
    print(x)
    quantizer = INTQuantizer(fmt=ElemFormat.int8, asymmetric=True)
    # quantizer = INTQuantizer(fmt=ElemFormat.int4, asymmetric=True)
    print(quantizer)
    x_q = quantizer(x)
    print(x_q)