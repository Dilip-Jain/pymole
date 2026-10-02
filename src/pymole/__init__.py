"""
PyMOLE - Python Mimetic Operators Library Enhanced
Copyright (C) 2025 Dilip Jain

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This is a Python reimplementation of MOLE:
Original MOLE Copyright (C) CSRC-SDSU
Repository: https://github.com/csrc-sdsu/mole
"""

from typing import Literal, Tuple, Union

from .backend import use_backend, get_backend
from . import pure

__version__ = "0.1.0a1"

Boundary = Literal['periodic', 'nonperiodic']
GridSize = Union[int, Tuple[int, ...]]
GridSpacing = Union[float, Tuple[float, ...]]

__all__ = [
    'use_backend',
    'get_backend',
    'create_gradient',
    'create_divergence',
    'create_laplacian',
    'create_interpol',
]


def create_gradient(n: GridSize, h: GridSpacing, boundary: Boundary = 'nonperiodic', k: int = 2):
    """Factory function that creates gradient operator using selected backend"""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.MimeticGradient(n, h, boundary=boundary, k=k)
    return pure.MimeticGradient(n, h, boundary=boundary, k=k)

def create_divergence(n: GridSize, h: GridSpacing, boundary: Boundary = 'nonperiodic', k: int = 2):
    """Factory function that creates divergence operator using selected backend"""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.MimeticDivergence(n, h, boundary=boundary, k=k)
    return pure.MimeticDivergence(n, h, boundary=boundary, k=k)


def create_laplacian(n: GridSize, h: GridSpacing, boundary: Boundary = 'nonperiodic', k: int = 2):
    """Factory function that creates Laplacian operator using selected backend"""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.MimeticLaplacian(n, h, boundary=boundary, k=k)
    return pure.MimeticLaplacian(n, h, boundary=boundary, k=k)


def create_interpol(
    n: GridSize,
    h: GridSpacing,
    c: Union[float, Tuple[float, ...]] = 0.5,
    boundary: Boundary = 'nonperiodic',
):
    """Factory function that creates interpolation operator using selected backend"""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.MimeticInterpol(n, h, c=c, boundary=boundary)
    return pure.MimeticInterpol(n, h, c=c, boundary=boundary)
