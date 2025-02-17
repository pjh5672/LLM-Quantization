import torch
from torch import nn

from formats import ElemFormat, _get_format_params
from utils import _reshape_to_blocks, _undo_reshape_to_blocks


class INTQuantizer(nn.Module):
    
    def __init__(self, fmt: ElemFormat, asymmetric=True, group_size=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert fmt in (ElemFormat.int8, ElemFormat.int4), \
            f"Not support Format for {self.__class__.__name__}"
        
        ebits, *_, max_norm, _ = _get_format_params(fmt)
        self.ebits = torch.tensor(ebits)
        self.max_norm = torch.tensor(max_norm)
        self.str_fmt = str(fmt)
        self.enable()
        self.configure(asymmetric=asymmetric, group_size=group_size)

    def configure(self, asymmetric=True, group_size=-1):
        self.asymmetric = asymmetric
        self.group_size = group_size

    def forward(self, x_float, scales=None, zeros=None):
        if self.is_enable:
            # Refer to https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/quantizer.py
            if self.group_size > 0:
                x_float, axes, org_shape, padded_shape = _reshape_to_blocks(x_float, axes=[-1], block_size=self.group_size)
            
            if self.asymmetric:
                max_int = (self.max_norm * 2) - 1 
                min_int = 0
            else:
                max_int = self.max_norm - 1
                min_int = -self.max_norm

            if (scales is not None) & (zeros is not None):
                x_dq = self.quantize(x_float, scales=scales, zeros=zeros,
                                          min_int=min_int, max_int=max_int)
            else:
                scales, zeros = self.find_params(x_float, already_reshaped=True)
                x_dq = self.quantize(x_float, scales=scales, zeros=zeros,
                                          min_int=min_int, max_int=max_int)
            if self.group_size > 0:
                return _undo_reshape_to_blocks(x_dq, padded_shape, org_shape, axes=axes)
            return x_dq
        return x_float

    def find_params(self, x_float, already_reshaped=False):
        if (self.group_size > 0) & (not already_reshaped):
            x_float, axes, org_shape, padded_shape = _reshape_to_blocks(x_float, 
                                                                         axes=[-1], 
                                                                         block_size=self.group_size)
        if self.asymmetric:
            max_val = x_float.amax(dim=-1, keepdim=True)
            min_val = x_float.amin(dim=-1, keepdim=True)
            max_int = (self.max_norm * 2) - 1 
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:  # we actually never used this
            max_val = x_float.abs().amax(dim=-1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = self.max_norm - 1
            min_int = -self.max_norm
            scales = max_val / max_int
            zeros = torch.tensor(0)

            assert torch.isnan(scales).sum() == 0
            assert torch.isnan(x_float).sum() == 0

        return scales, zeros

    def quantize(self, x_float, scales, zeros, min_int, max_int):
        quantized = torch.round(x_float / scales) + zeros
        quantized = torch.clamp(quantized, min_int, max_int)
        return (quantized - zeros) * scales

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

    x = torch.randn(6, 7)
    print(x)
    quantizer = INTQuantizer(fmt=ElemFormat.int8, asymmetric=True, group_size=-1)
    # quantizer = INTQuantizer(fmt=ElemFormat.int4)
    print(quantizer)
    x_q = quantizer(x)
    print(x_q)

    quantizer.configure(asymmetric=False, group_size=8)
    scales, zeros = quantizer.find_params(x)
    x_q = quantizer(x, scales=scales, zeros=zeros)
    print(x_q)