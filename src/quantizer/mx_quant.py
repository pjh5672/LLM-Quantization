import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch import nn

from formats import ElemFormat, _get_format_params, FP32_MIN_NORMAL
from third_party.microxcaling.mx.mx_ops import _quantize_mx


class MXQuantizer(nn.Module):

    def __init__(self, fmt: ElemFormat, device=torch.device('cpu'), *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert fmt in (ElemFormat.mxfp8_e4m3, ElemFormat.mxfp8_e5m2, 
                       ElemFormat.mxfp4, ElemFormat.mxint8, ElemFormat.mxint4), \
            f"Not support Format for {self.__class__.__name__}"
        
        ebits, mbits, emax, max_norm, min_norm = _get_format_params(fmt)
        self.ebits = torch.tensor(ebits)
        self.mbits = torch.tensor(mbits)
        self.emax = torch.tensor(emax)
        self.max_norm = torch.tensor(max_norm)
        self.min_norm = torch.tensor(min_norm)
        self.scale_bits = 8
        self.block_size = 32
        self.str_fmt = str(fmt)
        self.elem_format = self.str_fmt.split('.')[-1].replace('mx', '')
        self.enable()
    
    def forward(self, x_float):
        if self.is_enable:
            return self.quantize(x_float)
        return x_float
    
    def find_scale(self, x_float):
        max_val, _ = torch.max(torch.abs(x_float), dim=-1, keepdim=True)
        scales = max_val / self.max_norm
        return scales.to(torch.bfloat16)

    def _shared_exponents_mentissa(self, A, method="max", axes=None, mbits=None):
        """
        Get shared exponents for the passed matrix A with mantissa.
        Args:
        A      {PyTorch tensor} -- Input tensor
        method {str}            -- Exponent selection method.
                                    "max" uses the max absolute value
                                    "none" uses an exponent for each value (i.e., no sharing)
        axes   {list(int)}      -- List of integers which specifies the axes across which
                                    shared exponents are calculated.
        Returns:
        shared_exp {PyTorch tensor} -- Tensor of shared exponents
        """

        if method == "max":
            if axes is None:
                x_max = torch.max(torch.abs(A))
            else:
                x_max = A
                for axis in axes:
                    x_max, _ = torch.max(torch.abs(x_max), dim=axis, keepdim=True)
        elif method == "none":
            x_max = torch.abs(A)
        else:
            raise Exception("Unrecognized shared exponent selection method %s" % (method))

        # log2(shared_exp) and truncate to integer
        shared_exp = torch.floor(
            torch.log2(
                x_max + FP32_MIN_NORMAL * (x_max == 0).type(x_max.dtype)
            )
        )
        if mbits is not None:
            shared_expm = _safe_lshift(x_max, bits=mbits, exp=shared_exp)
            shared_expm = _round_mantissa(shared_expm, bits=mbits, round='nearest')
            shared_expm = _safe_rshift(shared_expm, bits=mbits, exp=shared_exp)
            return torch.log2(shared_expm)
        return shared_exp

    def quantize(self, x_float):
        quantized = _quantize_mx(A=x_float, 
                                scale_bits=self.scale_bits,
                                elem_format=self.elem_format,
                                shared_exp_method='max',
                                axes=-1,
                                block_size=self.block_size)
        return quantized

    def enable(self):
        self.is_enable = True
    
    def disable(self):
        self.is_enable = False

    def extra_repr(self):
        s = f"Format: {self.str_fmt.split('.')[-1].upper()}, "
        s += f"Max: {self.max_norm}, Min: {self.min_norm}"
        return s
    

if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device('cuda')
    x = torch.randn(4, 8).to(device=device)
    print(x)
    # quantizer = MXQuantizer(fmt=ElemFormat.mxfp4, device=device)
    quantizer = MXQuantizer(fmt=ElemFormat.mxfp8_e5m2, device=device)
    print(quantizer)
    x_dq = quantizer(x)
    print(x_dq)
    print(((x-x_dq)**2).mean())
    