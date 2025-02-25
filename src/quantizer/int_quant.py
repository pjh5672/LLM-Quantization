import torch
from torch import nn

from formats import ElemFormat, _get_format_params
from utils import _reshape_to_blocks, _undo_reshape_to_blocks


class INTQuantizer(nn.Module):
    
    def __init__(self, fmt: ElemFormat, asymmetric=True, group_size=-1, 
                 device=torch.device('cpu'), *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert fmt in (ElemFormat.int8, ElemFormat.int4), \
            f"Not support Format for {self.__class__.__name__}"
        
        ebits, *_, max_norm, _ = _get_format_params(fmt)
        self.ebits = torch.tensor(ebits)
        self.max_norm = torch.tensor(max_norm)
        self.max_norm = self.max_norm.to(device=device)
        self.configure(asymmetric=asymmetric, group_size=group_size)
        self.enable()
        self.str_fmt = str(fmt)

    def configure(self, asymmetric=True, group_size=-1):
        self.asymmetric = asymmetric
        self.group_size = group_size

        if self.asymmetric:
            self.max_int = (self.max_norm * 2) - 1 
            self.min_int = 0
        else:
            self.max_int = self.max_norm - 1
            self.min_int = -self.max_norm

    def forward(self, x_float, scales=None, zeros=None):
        if self.is_enable:
            # Refer to https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/quantizer.py
            if self.group_size > 0:
                x_float, axes, org_shape, padded_shape = _reshape_to_blocks(x_float, axes=[-1], block_size=self.group_size)

            if (scales is not None) & (zeros is not None):
                x_dq = self.quantize(x_float, scales=scales, zeros=zeros)
            else:
                scales, zeros = self.find_params(x_float, already_reshaped=True)
                x_dq = self.quantize(x_float, scales=scales, zeros=zeros)
            if self.group_size > 0:
                return _undo_reshape_to_blocks(x_dq, padded_shape, org_shape, axes=axes)
            return x_dq
        
        return x_float

    def find_params(self, x_float, already_reshaped=False):
        if (self.group_size > 0) & (not already_reshaped):
            x_float, *_ = _reshape_to_blocks(x_float, axes=[-1], block_size=self.group_size)

        if self.asymmetric:
            max_val = x_float.amax(dim=-1, keepdim=True)
            min_val = x_float.amin(dim=-1, keepdim=True)
            scales = (max_val - min_val).clamp(min=1e-5) / self.max_int
            zeros = (-torch.round(min_val / scales)).clamp_(self.min_int, self.max_int)
        else:
            max_val = x_float.abs().amax(dim=-1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            scales = max_val / self.max_int
            zeros = 0

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(x_float).sum() == 0

        scales = scales.to(torch.bfloat16)
        zeros = zeros.to(torch.bfloat16)
        return scales, zeros

    def quantize(self, x_float, scales, zeros):
        quantized = torch.round(x_float / scales) + zeros
        quantized = torch.clamp(quantized, self.min_int, self.max_int)
        return (quantized - zeros) * scales

    def enable(self):
        self.is_enable = True
    
    def disable(self):
        self.is_enable = False

    def extra_repr(self):
        s = f"Format: {self.str_fmt.split('.')[-1].upper()}, "
        s += f"Max: {self.max_int}, Min: {self.min_int}"
        return s


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device('cuda')
    x = torch.randn(6, 7).to(device=device)
    print(x)
    quantizer = INTQuantizer(fmt=ElemFormat.int8, asymmetric=True, 
                             group_size=-1, device=device)
    # quantizer = INTQuantizer(fmt=ElemFormat.int4, asymmetric=True, 
    #                          group_size=-1, device=device)
    print(quantizer)
    x_q = quantizer(x)
    print(x_q)

    # print(quantizer)
    # scales, zeros = quantizer.find_params(x)
    # x_q = quantizer(x, scales=scales, zeros=zeros)
    # print(x_q)