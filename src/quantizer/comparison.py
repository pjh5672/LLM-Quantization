from formats import ElemFormat
from int_quant import INTQuantizer
from fp_quant import FPQuantizer
from mx_quant import MXQuantizer

def mse(a, b):
    return ((a - b) ** 2).mean()

def int_quantize(x, device, asymmetric=False, group_size=128):
    quantizer = INTQuantizer(fmt=ElemFormat.int8, asymmetric=asymmetric, 
                             group_size=group_size, device=device)
    print(quantizer)
    x_q = quantizer(x)
    print(f"INT format - MSE: {mse(x, x_q):.6f}")

def fp_quantize(x, device):
    quantizer = FPQuantizer(fmt=ElemFormat.fp8_e4m3, device=device)
    print(quantizer)
    x_q = quantizer(x)
    print(f"FP format - MSE: {mse(x, x_q):.6f}")

def mx_quantize(x, device):
    quantizer = MXQuantizer(fmt=ElemFormat.mxfp8_e4m3, device=device)
    print(quantizer)
    x_q = quantizer(x)
    print(f"MX format - MSE: {mse(x, x_q):.6f}")


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    device = torch.device('cuda')
    x = torch.randn(512, 512).to(device) * 100
    
    for quant_func in [int_quantize, fp_quantize, mx_quantize]:
        quant_func(x, device)