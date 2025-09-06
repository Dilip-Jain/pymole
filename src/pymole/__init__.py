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

from .backend import use_backend, get_backend
from . import pure

__version__ = "0.1.0-alpha"

def create_gradient(n, h):
    """Factory function that creates gradient operator using selected backend"""
    if get_backend() == 'cpp':
        from . import cpp
        return cpp.MimeticGradient(n, h)
    return pure.MimeticGradient(n, h)

def create_divergence(n, h):
    """Factory function that creates divergence operator using selected backend"""
    if get_backend() == 'cpp':
        from . import cpp
        return cpp.MimeticDivergence(n, h)
    return pure.MimeticDivergence(n, h)
