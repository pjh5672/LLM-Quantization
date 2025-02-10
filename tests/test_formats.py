import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pytest

from src.quantizer.formats import ElemFormat, _get_format_params


def test_fp8_e4m3():
    *_, emax, max_norm, min_norm = _get_format_params(ElemFormat.fp8_e4m3)
    assert emax == 8
    assert max_norm == (2 ** 8) * 1.75
    assert min_norm == 2 ** -6

def test_fp8_e5m2():
    *_, emax, max_norm, min_norm = _get_format_params(ElemFormat.fp8_e5m2)
    assert emax == 15
    assert max_norm == (2 ** 15) * 1.75
    assert min_norm == 2 ** -14

def test_fp4():
    *_, emax, max_norm, min_norm = _get_format_params(ElemFormat.fp4)
    assert emax == 2
    assert max_norm == (2 ** 2) * 1.5
    assert min_norm == 2 ** 0

def test_int8():
    *_, max_norm, _ = _get_format_params(ElemFormat.int8)
    assert max_norm == 127

def test_int4():
    *_, max_norm, _ = _get_format_params(ElemFormat.int4)
    assert max_norm == 7

def test_mxfp8_e4m3():
    *_, emax, max_norm, min_norm = _get_format_params(ElemFormat.mxfp8_e4m3)
    assert emax == 8
    assert max_norm == (2 ** 8) * 1.75
    assert min_norm == 2 ** -6

def test_mxfp8_e5m2():
    *_, emax, max_norm, min_norm = _get_format_params(ElemFormat.mxfp8_e5m2)
    assert emax == 15
    assert max_norm == (2 ** 15) * 1.75
    assert min_norm == 2 ** -14

def test_mxfp4():
    *_, emax, max_norm, min_norm = _get_format_params(ElemFormat.mxfp4)
    assert emax == 2
    assert max_norm == (2 ** 2) * 1.5
    assert min_norm == 2 ** 0