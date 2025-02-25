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

    def __init__(self, fmt: ElemFormat, nano_mbits=0, device=torch.device('cpu'), *args, **kwargs):
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
        self.nano_mbits = nano_mbits # Nanoscaling Floating-Point (NxFP)
        self.str_fmt = str(fmt)
        self.enable()
    
    def forward(self, x_float):
        if self.is_enable:
            return self.quantize(x_float)
        return x_float
    
    def quantize(self, x_float):
        elem_format = self.str_fmt.split('.')[-1].replace('mx', '')

        v_max, _ = torch.max(torch.abs(x_float), dim=-1, keepdim=True)
        e_max = torch.floor(torch.log2(v_max))
        # print('1', torch.log2(v_max))
        # print('2', e_max)
        m_nano = v_max / (2 ** e_max)
        print('3', m_nano)
        m_nano = m_nano * ((2 ** self.nano_mbits) * (2 ** (self.mbits - 2)))
        # print('4', m_nano)
        m_nano = torch.floor(torch.abs(m_nano) + 0.5)
        # print('5', m_nano)
        m_nano = m_nano / ((2 ** self.nano_mbits) * (2 ** (self.mbits - 2)))
        print('6', m_nano)
        m_nano2 = m_nano * (2 ** self.nano_mbits)
        m_nano2 = torch.floor(torch.abs(m_nano2) + 0.5)
        m_nano2 = m_nano2 / (2 ** self.nano_mbits)
        print('7', m_nano2)
        x = x_float / m_nano2
        print('9', x)
        quantized = _quantize_mx(A=x, 
                                scale_bits=self.scale_bits,
                                elem_format=elem_format,
                                shared_exp_method='max',
                                axes=-1,
                                block_size=self.block_size)
        # print('8', quantized)
        quantized = quantized * m_nano2
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
    x = torch.randn(8, 1).to(device=device)
    print(x)
    # quantizer = MXQuantizer(fmt=ElemFormat.mxfp8_e5m2, nano_mbits=10, device=device)
    quantizer = MXQuantizer(fmt=ElemFormat.mxfp4, nano_mbits=2, device=device)
    # print(quantizer)
    x_dq = quantizer(x)
    print(x_dq)
    print(((x-x_dq)**2).mean())
    
    # mbits = 4
    # e_grid = 2 ** torch.tensor([0, 1, 2])
    # m_grid = 1 + torch.linspace(0, (2**mbits)-1, 2**mbits) / (2**mbits)
    # x = torch.outer(e_grid, m_grid).reshape(-1, 1)
    # print(x)

    # for i in range(9):
    #     print("##### " * 5 + f" NANO MBIT:{i} " + " ######" * 5)
    #     # quantizer = MXQuantizer(fmt=ElemFormat.mxfp8_e4m3, nano_mbits=i)
    #     # quantizer = MXQuantizer(fmt=ElemFormat.mxfp8_e5m2, nano_mbits=i)
    #     quantizer = MXQuantizer(fmt=ElemFormat.mxfp4, nano_mbits=i)
    #     x_dq = quantizer(x)
    #     print(x_dq)
    #     print(((x-x_dq)**2).mean())