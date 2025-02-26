from enum import Enum


FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


# Enum for scalar data formats which is similar to microxcaling lib's for compatibility.
class ElemFormat(Enum):
    fp8_e4m3 = 1
    fp8_e5m2 = 2
    fp4 = 3
    int8 = 4
    int4 = 5
    mxfp8_e4m3 = 6
    mxfp8_e5m2 = 7
    mxfp4 = 8
    mxint8 = 9
    mxint4 = 10

    @staticmethod
    def from_str(s):
        assert(s != None), "String elem_format == None"
        s = s.lower()
        if hasattr(ElemFormat, s):
            return getattr(ElemFormat, s)
        else:
            raise Exception("Undefined elem format", s)


_FORMAT_CACHE = {}
def _get_format_params(fmt):
    """ Allowed formats:
        - fpX:      X={4, 8}bit
        - intX:     X={4, 8}bit
        - mxfpX:    X={4, 8}bit

        Returns:
          ebits: exponent bits
          mbits: mantissa bits: includes sign and implicit bits
          emax: max normal exponent
          max_norm: max normal number
          min_norm: min normal number
    """

    if type(fmt) is str:
        fmt = ElemFormat.from_str(fmt)
    
    if fmt in _FORMAT_CACHE:
        return _FORMAT_CACHE[fmt]
    
    match fmt:
        case ElemFormat.fp8_e4m3:
            ebits, mbits = 4, 3
            emax = 2 ** (ebits - 1)
            emin = 2 - (2 ** (ebits - 1))
            max_norm = 2 ** emax * (2 - 2 ** (1-mbits))
            min_norm = 2 ** emin

        case ElemFormat.fp8_e5m2:
            ebits, mbits = 5, 2
            emax = 2 ** (ebits - 1) - 1
            emin = 2 - (2 ** (ebits - 1))
            max_norm = 2 ** emax * (2 - 2 ** (-mbits))
            min_norm = 2 ** emin
        
        case ElemFormat.fp4:
            ebits, mbits = 2, 1
            emax = 2 ** (ebits - 1)
            emin = 2 - (2 ** (ebits - 1))
            max_norm = 2 ** emax * (2 - 2 ** (-mbits))
            min_norm = 2 ** emin
        
        case ElemFormat.int8:
            ebits, mbits = 8, None
            emax = None
            emin = None
            max_norm = 2 ** (ebits - 1)
            min_norm = None

        case ElemFormat.int4:
            ebits, mbits = 4, None
            emax = None
            emin = None
            max_norm = 2 ** (ebits - 1)
            min_norm = None

        case ElemFormat.mxfp8_e4m3:
            ebits, mbits = 4, 5
            emax = 2**(ebits - 1)
            emin = 2 - (2 ** (ebits - 1))
            max_norm = 2**emax * 1.75
            min_norm = 2 ** emin

        case ElemFormat.mxfp8_e5m2:
            ebits, mbits = 5, 4
            emax = 2 ** (ebits - 1) - 1
            emin = 2 - (2 ** (ebits - 1))
            max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
            min_norm = 2 ** emin
        
        case ElemFormat.mxfp4:
            ebits, mbits = 2, 3
            emax = 2 ** (ebits - 1)
            emin = 2 - (2 ** (ebits - 1))
            max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
            min_norm = 2 ** emin
        
        case ElemFormat.mxint8:
            ebits, mbits = 0, 8
            emax = 0
            emin = 2 - (2 ** (ebits - 1))
            max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
            min_norm = 0
        
        case ElemFormat.mxint4:
            ebits, mbits = 0, 4
            emax = 0
            emin = 2 - (2 ** (ebits - 1))
            max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
            min_norm = 0

    _FORMAT_CACHE[fmt] = (ebits, mbits, emax, max_norm, min_norm)
    return ebits, mbits, emax, max_norm, min_norm


if __name__ == "__main__":
    y = _get_format_params(ElemFormat.fp8_e4m3)
    print(y)
