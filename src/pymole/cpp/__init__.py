"""C++ implementation bindings for mimetic operators."""

import numpy as np
from scipy import sparse
from typing import Literal

try:
    from . import _operators
except ImportError:
    raise ImportError(
        "C++ implementation not available. "
        "Please install pymole with C++ support using: "
        "pip install pymole[cpp]"
    )

from ..base import MimeticOperator

class MimeticGradient(MimeticOperator):
    """1D Mimetic gradient operator using C++ implementation.
    
    This operator wraps the C++ implementation from the MOLE library,
    maintaining all the mathematical properties:
    - Discrete integration by parts
    - Proper handling of boundary conditions
    - Exact representation of constant functions
    
    Args:
        n: Number of grid points
        h: Grid spacing
        boundary: Boundary condition type ('periodic' or 'nonperiodic')
        
    Raises:
        ImportError: If C++ implementation is not available
    """
    
    def __init__(self, n: int, h: float,
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic'):
        super().__init__(n, h)
        self.boundary = boundary
        self._cpp_op = _operators.GradientOperator(n, h)
        self._matrix = self._build_matrix()
    
    def _build_matrix(self) -> sparse.spmatrix:
        """Build the gradient operator matrix using C++ implementation."""
        if self.boundary == 'periodic':
            self._cpp_op.periodic()
        else:
            self._cpp_op.nonperiodic()
        return self._cpp_op.matrix()

class MimeticDivergence(MimeticOperator):
    """1D Mimetic divergence operator using C++ implementation.
    
    This operator wraps the C++ implementation from the MOLE library,
    maintaining all the mathematical properties:
    - Discrete integration by parts with gradient
    - Exact divergence of constant vector fields
    - Proper boundary condition handling
    
    Args:
        n: Number of grid points
        h: Grid spacing
        boundary: Boundary condition type ('periodic' or 'nonperiodic')
        
    Raises:
        ImportError: If C++ implementation is not available
    """
    
    def __init__(self, n: int, h: float,
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic'):
        super().__init__(n, h)
        self.boundary = boundary
        self._cpp_op = _operators.DivergenceOperator(n, h)
        self._matrix = self._build_matrix()
    
    def _build_matrix(self) -> sparse.spmatrix:
        """Build the divergence operator matrix using C++ implementation."""
        if self.boundary == 'periodic':
            self._cpp_op.periodic()
        else:
            self._cpp_op.nonperiodic()
        return self._cpp_op.matrix()

class MimeticLaplacian(MimeticOperator):
    """1D Mimetic Laplacian operator using C++ implementation.
    
    This operator wraps the C++ implementation from the MOLE library,
    maintaining all the mathematical properties:
    - Self-adjointness
    - Negative semi-definiteness
    - Proper null space (constants)
    
    Args:
        n: Number of grid points
        h: Grid spacing
        boundary: Boundary condition type ('periodic' or 'nonperiodic')
        
    Raises:
        ImportError: If C++ implementation is not available
    """
    
    def __init__(self, n: int, h: float,
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic'):
        super().__init__(n, h)
        self.boundary = boundary
        self._cpp_op = _operators.LaplacianOperator(n, h)
        self._matrix = self._build_matrix()
    
    def _build_matrix(self) -> sparse.spmatrix:
        """Build the Laplacian operator matrix using C++ implementation."""
        if self.boundary == 'periodic':
            self._cpp_op.periodic()
        else:
            self._cpp_op.nonperiodic()
        return self._cpp_op.matrix()
