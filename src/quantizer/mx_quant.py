import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch import nn

from formats import ElemFormat, _get_format_params, FP32_MIN_NORMAL
from third_party.microxcaling.mx.mx_ops import _reshape_to_blocks, _undo_reshape_to_blocks
from third_party.microxcaling.mx.elemwise_ops import (_safe_lshift, 
                                                      _safe_rshift,
                                                      _round_mantissa, 
                                                      _quantize_elemwise_core)


class MXQuantizer(nn.Module):

    def __init__(self, fmt: ElemFormat, nano_mbits=0, device=torch.device('cpu'), *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert fmt in (ElemFormat.mxfp8_e4m3, ElemFormat.mxfp8_e5m2, 
                       ElemFormat.mxfp4, ElemFormat.mxint8, ElemFormat.mxint4), \
            f"Not support Format for {self.__class__.__name__}"
        
        ebits, mbits, emax, max_norm, min_norm = _get_format_params(fmt)
        self.ebits = torch.tensor(ebits).to(device)
        self.mbits = torch.tensor(mbits).to(device)
        self.emax = torch.tensor(emax).to(device)
        self.max_norm = torch.tensor(max_norm).to(device)
        self.min_norm = torch.tensor(min_norm).to(device)
        self.scale_bits = 8
        self.block_size = 32
        self.axes = [-1]
        self.nano_mbits = nano_mbits # Nanoscaling Floating-Point (NxFP)
        self.str_fmt = str(fmt)
        self.elem_format = self.str_fmt.split('.')[-1].replace('mx', '')
        self.scale_emax = 2**(self.scale_bits-1) - 1
        self.enable()
    
    def forward(self, x_float):
        if self.is_enable:
            axes = [x + x_float.ndim if x < 0 else x for x in self.axes]
            x_float, axes, org_shape, padded_shape = _reshape_to_blocks(x_float, 
                                                                        axes=[-1], 
                                                                        block_size=self.block_size)
            # candidate 1 #
            # shared_exp_axes = [x + 1 for x in axes] if self.block_size > 0 else axes
            # shared_expm = self._shared_exponents_mentissa(x_float, method='max', 
            #                                              axes=shared_exp_axes, 
            #                                              mbits=self.nano_mbits)
            # shared_expm -= self.emax
            # shared_expm[shared_expm > self.scale_emax] = self.scale_emax + 1
            # shared_expm[shared_expm < -self.scale_emax] = -self.scale_emax
            # x_float = x_float / (2 ** shared_expm)
            # x_dequant = self.quantize(x_float) * (2 ** shared_expm)

            # candidate 2 #
            scales = self.find_scale(x_float)
            scales = _safe_lshift(scales, bits=self.nano_mbits, exp=None)
            scales = _round_mantissa(scales, bits=self.nano_mbits, round='nearest')
            scales = _safe_rshift(scales, bits=self.nano_mbits, exp=None)
            scales = scales + 1.0 * (scales == 0).type(scales.dtype)
            x_float = x_float / scales
            x_dequant = self.quantize(x_float)
            x_dequant = x_dequant * scales

            return _undo_reshape_to_blocks(x_dequant, padded_shape, org_shape, axes)
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
        return _quantize_elemwise_core(
                x_float, self.mbits, self.ebits, self.max_norm, round='nearest',
                allow_denorm=True, saturate_normals=True)

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
    x = torch.randn(8, 32).to(device=device)
    print(x)
    # quantizer = MXQuantizer(fmt=ElemFormat.mxfp8_e5m2, nano_mbits=0, device=device)
    # quantizer = MXQuantizer(fmt=ElemFormat.mxfp4, nano_mbits=0, device=device)
    # print(quantizer)
    # x_dq = quantizer(x)
    # print(x_dq)
    # print(((x-x_dq)**2).mean())
    
    for i in range(9):
        print("##### " * 5 + f" NANO MBIT:{i} " + " ######" * 5)
        # quantizer = MXQuantizer(fmt=ElemFormat.mxfp8_e4m3, nano_mbits=i, device=device)
        # quantizer = MXQuantizer(fmt=ElemFormat.mxfp8_e5m2, nano_mbits=i, device=device)
        quantizer = MXQuantizer(fmt=ElemFormat.mxfp4, nano_mbits=i, device=device)
        x_dq = quantizer(x)
        # print(x_dq)
        print(((x-x_dq)**2).mean())

    # mbits = 3
    # e_grid = 2 ** torch.tensor([0, 1])
    # m_grid = 1 + torch.linspace(0, (2**mbits)-1, 2**mbits) / (2**mbits)
    # x = torch.outer(e_grid, m_grid).reshape(-1, 1)
    # print(x)

    # for i in range(5):
    #     print("##### " * 5 + f" NANO MBIT:{i} " + " ######" * 5)
    #     # quantizer = MXQuantizer(fmt=ElemFormat.mxfp8_e4m3, nano_mbits=i)
    #     # quantizer = MXQuantizer(fmt=ElemFormat.mxfp8_e5m2, nano_mbits=i)
    #     quantizer = MXQuantizer(fmt=ElemFormat.mxfp4, nano_mbits=i)
    #     x_dq = quantizer(x)
    #     print(x_dq)
    #     print(((x-x_dq)**2).mean())