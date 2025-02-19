import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch import nn

from formats import ElemFormat, _get_format_params
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
        self.enable()
    
    def forward(self, x_float):
        if self.is_enable:
            return self.quantize(x_float)
        return x_float
    
    def quantize(self, x_float):
        elem_format = self.str_fmt.split('.')[-1].replace('mx', '')
        quantized = _quantize_mx(A=x_float, 
                                scale_bits=self.scale_bits,
                                elem_format=elem_format,
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
    x = torch.randn(6, 8).to(device=device)
    print(x)
    quantizer = MXQuantizer(fmt=ElemFormat.mxfp8_e5m2, device=device)
    print(quantizer)
    x_q = quantizer(x)
    print(x_q)
    