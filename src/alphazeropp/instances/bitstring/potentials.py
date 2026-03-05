"""
Potential (fitness) functions for bitstring states.

Each function takes a 1-D numpy array x of 0s and 1s and returns an integer.
Returning int (not float) is deliberate: it enables exact telescoping checks
using integer arithmetic, avoiding any floating-point accumulation error.
"""

import numpy as np
from typing import Callable


def onemax(x: np.ndarray) -> int:
    """Sum of bits. Optimal value = len(x)."""
    return int(np.sum(x))


def leading_ones(x: np.ndarray) -> int:
    """Length of the leading all-ones prefix.

    For x = [1, 1, 1, 0, 1], returns 3.
    For x = [0, 1, 1, 1, 1], returns 0.
    """
    n = len(x)
    for i in range(n):
        if x[i] != 1:
            return i
    return n


def binval(x: np.ndarray) -> int:
    """Binary value with x[0] as the most significant bit.

    For x = [1, 0, 1], returns 5 (i.e. 1*4 + 0*2 + 1*1).
    """
    result = 0
    for bit in x:
        result = 2 * result + int(bit)
    return result


POTENTIAL_REGISTRY: dict[str, Callable[[np.ndarray], int]] = {
    "onemax": onemax,
    "leading_ones": leading_ones,
    "binval": binval,
}
