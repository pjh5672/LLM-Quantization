import torch
from torch import nn

from formats import ElemFormat, _get_format_params


class FPQuantizer(nn.Module):
    
    def __init__(self, fmt: ElemFormat, device=torch.device('cpu'), *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert fmt in (ElemFormat.fp8_e4m3, ElemFormat.fp8_e5m2, ElemFormat.fp4), \
            f"Not support Format for {self.__class__.__name__}"

        ebits, mbits, emax, max_norm, min_norm = _get_format_params(fmt)
        self.ebits = torch.tensor(ebits)
        self.mbits = torch.tensor(mbits)
        self.emax = torch.tensor(emax)
        self.max_norm = torch.tensor(max_norm)
        self.min_norm = torch.tensor(min_norm)
        self.str_fmt = str(fmt)
        self.max_norm = self.max_norm.to(device=device)
        self.enable()

    def forward(self, x_float):
        if self.is_enable:
            return self.quantize(x_float)
        return x_float

    def quantize(self, x_float):
        # This is refered to https://github.com/quic/aimet/blob/develop/TrainingExtensions/torch/src/python/aimet_torch/fp_quantization.py#L172
        # Math explanation of what happens here:
        # Bias is computed from maxval: $B=2^E - \log_2(M) + \log_2(2 - 2^{-M}) - 1$
        # This follows from maxval $M=(2 - 2^{-M}) \cdot 2^{2^E-1-B}$.
        if 'fp8_e4m3' in self.str_fmt:
            bias = 2 ** self.ebits - torch.log2(self.max_norm) + torch.log2(2 - 2 ** (1-self.mbits)) - 1
        elif 'fp8_e5m2' in self.str_fmt:
            bias = 2 ** self.ebits - torch.log2(self.max_norm) + torch.log2(2 - 2 ** (-self.mbits)) - 2
        elif 'fp4' in self.str_fmt:
            bias = 2 ** self.ebits - torch.log2(self.max_norm) + torch.log2(2 - 2 ** (-self.mbits)) - 1

        # Ensure no values are greater than the maximum value represented by an 8 bit float system
        # with M mantissa and E exponent bits. torch.min/torch.max are used to allow gradients to
        # flow to maxval
        x_clipped = x_float.clamp(-self.max_norm, self.max_norm)

        # FP quantization scale is determined per-element, and is computed as
        # \log_2 s = \left\lfloor \log_2 |x_c| + B \right\rfloor - M - B
        # the addition of bias inside the floor and subtraction outside ensures that a
        # tensor scaling $\alpha \neq 1$ is correctly incorporated
        log_scales = torch.floor(torch.log2(torch.abs(x_clipped)) + bias).detach()

        # This ensures scales are never smaller than the subnormal scale
        log_scales = torch.clamp(log_scales, 1.)

        # Second step of computing scale $s$
        scales = 2. ** (log_scales - self.mbits - bias)

        # Using the per-element scale we can quantize the clipped input tensor to the FP grid
        return torch.round(x_clipped / scales) * scales

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
    quantizer = FPQuantizer(fmt=ElemFormat.fp8_e4m3, device=device)
    # quantizer = FPQuantizer(fmt=ElemFormat.fp8_e5m2)
    # quantizer = FPQuantizer(fmt=ElemFormat.fp4)
    print(quantizer)
    x_q = quantizer(x)
    print(x_q)