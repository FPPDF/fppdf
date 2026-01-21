""" 
    Numba-jitted functions for the calculation chebyshev polynomials
"""

# when the only thing that is supposed to go through gamma are scalars
# scipy.special.gamma should be equivalent to math.gamma, which should be supported by numba
from math import gamma

import numba as nb
import numpy as np


@nb.njit
def I(a, b):

    if b < 100.0:
        out = gamma(a + 1) * gamma(b + 1)
        out = out / gamma(a + b + 2)
    else:  # use Stirling's formula for numerical convergence
        out = (
            np.sqrt(b / (a + b + 1.0))
            * np.power(b / (a + b + 1.0), b)
            * np.power(a + b + 1.0, -a - 1.0)
        )
        out = out * gamma(a + 1.0) * np.exp(a + 1.0)

    return out


@nb.njit
def Iy1(a, b):

    out = I(a, b) - 2.0 * I(a + 0.5, b)

    return out


@nb.njit
def Iy2(a, b):

    out = I(a, b) - 4.0 * I(a + 0.5, b) + 4.0 * I(a + 1.0, b)

    return out


@nb.njit
def Iy3(a, b):

    out = I(a, b) - 6.0 * I(a + 0.5, b) + 12.0 * I(a + 1.0, b) - 8.0 * I(a + 1.5, b)

    return out


@nb.njit
def Iy4(a, b):

    out = (
        I(a, b)
        - 8.0 * I(a + 0.5, b)
        + 24.0 * I(a + 1.0, b)
        - 32.0 * I(a + 1.5, b)
        + 16.0 * I(a + 2.0, b)
    )

    return out


@nb.njit
def Iy5(a, b):

    out = (
        I(a, b)
        - 10.0 * I(a + 0.5, b)
        + 40.0 * I(a + 1.0, b)
        - 80.0 * I(a + 1.5, b)
        + 80.0 * I(a + 2.0, b)
        - 32.0 * I(a + 2.5, b)
    )

    return out


@nb.njit
def Iy6(a, b):

    out = (
        I(a, b)
        - 12.0 * I(a + 0.5, b)
        + 60.0 * I(a + 1.0, b)
        - 160.0 * I(a + 1.5, b)
        + 240.0 * I(a + 2.0, b)
        - 192.0 * I(a + 2.5, b)
        + 64.0 * I(a + 3.0, b)
    )

    return out


@nb.njit
def Iy7(a, b):

    out = (
        I(a, b)
        - 14.0 * I(a + 0.5, b)
        + 84.0 * I(a + 1.0, b)
        - 280.0 * I(a + 1.5, b)
        + 560.0 * I(a + 2.0, b)
        - 672.0 * I(a + 2.5, b)
        + 448.0 * I(a + 3.0, b)
        - 128.0 * I(a + 3.5, b)
    )

    return out


@nb.njit
def Iy8(a, b):

    out = (
        I(a, b)
        - 16.0 * I(a + 0.5, b)
        + 112.0 * I(a + 1.0, b)
        - 448.0 * I(a + 1.5, b)
        + 1120.0 * I(a + 2.0, b)
        - 1792.0 * I(a + 2.5, b)
        + 1792.0 * I(a + 3.0, b)
        - 1024.0 * I(a + 3.5, b)
        + 256.0 * I(a + 4.0, b)
    )

    return out


@nb.njit
def Ic1(a, b):

    out = Iy1(a, b)

    return out


@nb.njit
def Ic2(a, b):

    out = 2.0 * Iy2(a, b) - I(a, b)

    return out


@nb.njit
def Ic3(a, b):

    out = 4.0 * Iy3(a, b) - 3.0 * Iy1(a, b)

    return out


@nb.njit
def Ic4(a, b):

    out = 8.0 * Iy4(a, b) - 8.0 * Iy2(a, b) + I(a, b)

    return out


@nb.njit
def Ic5(a, b):

    out = 16.0 * Iy5(a, b) - 20.0 * Iy3(a, b) + 5.0 * Iy1(a, b)

    return out


@nb.njit
def Ic6(a, b):

    out = 32.0 * Iy6(a, b) - 48.0 * Iy4(a, b) + 18.0 * Iy2(a, b) - I(a, b)

    return out


@nb.njit
def Ic7(a, b):

    out = 64.0 * Iy7(a, b) - 112.0 * Iy5(a, b) + 56.0 * Iy3(a, b) - 7.0 * Iy1(a, b)

    return out


@nb.njit
def Ic8(a, b):

    out = 128.0 * Iy8(a, b) - 256.0 * Iy6(a, b) + 160.0 * Iy4(a, b) - 32.0 * Iy2(a, b) + I(a, b)

    return out
