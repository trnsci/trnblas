"""Normalise the BLAS flag arguments (`uplo`, `side`, `diag`, `trans*`).

Before 0.9.2 each wrapper compared a flag against one spelling and silently
took the other branch for anything else, so the BLAS letters `uplo="U"`,
`diag="U"` and `trans="N"` ran the opposite operation (#73). The documented
words and the BLAS letters are accepted, case-insensitively; anything else
raises `ValueError`.
"""

from __future__ import annotations

_UPLO = {"upper": "upper", "u": "upper", "lower": "lower", "l": "lower"}
_SIDE = {"left": "left", "l": "left", "right": "right", "r": "right"}
_DIAG = {"unit": "unit", "u": "unit", "nonunit": "nonunit", "non-unit": "nonunit", "n": "nonunit"}
_TRANS = {"n": False, "t": True}


def _lookup(name: str, value, table: dict):
    if isinstance(value, str):
        key = value.strip().lower()
        if key in table:
            return table[key]
    words = sorted({v for v in table.values()}, key=str)
    raise ValueError(f"{name} must be one of {words} (or the BLAS letter); got {value!r}")


def uplo_flag(value) -> str:
    """'upper' or 'lower'."""
    return _lookup("uplo", value, _UPLO)


def side_flag(value) -> str:
    """'left' or 'right'."""
    return _lookup("side", value, _SIDE)


def diag_flag(value) -> str:
    """'unit' or 'nonunit'."""
    return _lookup("diag", value, _DIAG)


def trans_flag(value, name: str = "trans") -> bool:
    """A bool transpose flag. Non-strings keep their truth value; the BLAS
    characters 'N' / 'T' mean no-transpose / transpose. 'C' is rejected:
    trnblas has no conjugate transpose."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _TRANS:
            return _TRANS[key]
        raise ValueError(f"{name} must be a bool or 'N' / 'T'; got {value!r}")
    return bool(value)
