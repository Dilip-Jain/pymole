"""Pure Python implementation of mimetic operators."""

import numpy as np
from scipy import sparse
from scipy.sparse import spmatrix, csc_matrix, dia_matrix
from typing import Optional, Literal, List, Union, Tuple, Callable

from ..base import MimeticOperator

def _build_periodic_tridiag(n: int, diag_values: Tuple[float, float, float]) -> spmatrix:
    """Build a periodic tridiagonal matrix with constant diagonals.
    
    Args:
        n: Size of the matrix
        diag_values: Values for (main, upper, lower) diagonals
    
    Returns:
        Sparse matrix in CSC format
    """
    idx = np.arange(n)
    main_val, upper_val, lower_val = diag_values
    
    rows = np.concatenate([idx, idx, idx])
    cols = np.concatenate([idx, (idx + 1) % n, (idx - 1) % n])
    data = np.concatenate([
        np.full(n, main_val),
        np.full(n, upper_val),
        np.full(n, lower_val)
    ])
    
    return csc_matrix((data, (rows, cols)), shape=(n, n))

def _build_nonperiodic_tridiag(
    n: int,
    boundary_values: Tuple[Tuple[float, float], Tuple[float, float]],
    interior_values: Tuple[float, float, float]
) -> spmatrix:
    """Build a non-periodic tridiagonal matrix with special boundary handling.
    
    Args:
        n: Size of the matrix
        boundary_values: ((first_diag, first_upper), (last_lower, last_diag))
        interior_values: Values for interior (lower, main, upper) diagonals
    
    Returns:
        Sparse matrix in CSC format
    """
    interior_idx = np.arange(1, n-1)
    (first_diag, first_upper), (last_lower, last_diag) = boundary_values
    lower_val, main_val, upper_val = interior_values
    
    rows = np.concatenate([
        [0, 0],                      # First point
        np.repeat(interior_idx, 3),   # Interior points
        [n-1, n-1]                   # Last point
    ])
    
    cols = np.concatenate([
        [0, 1],                      # First point
        np.column_stack((
            interior_idx-1,          # Lower diagonal
            interior_idx,            # Main diagonal
            interior_idx+1           # Upper diagonal
        )).flat,
        [n-2, n-1]                  # Last point
    ])
    
    data = np.concatenate([
        [first_diag, first_upper],   # First point
        np.tile([lower_val, main_val, upper_val], n-2),  # Interior
        [last_lower, last_diag]      # Last point
    ])
    
    return csc_matrix((data, (rows, cols)), shape=(n, n))

class MimeticGradient(MimeticOperator):
    """1D Mimetic gradient operator.

    This operator implements a mimetic discretization of the gradient
    operator following the MOLE library implementation. It preserves
    important mathematical properties including:
    - Discrete integration by parts
    - Proper handling of boundary conditions
    - Exact representation of constant functions

    Args:
        n: Number of grid points
        h: Grid spacing
        boundary: Boundary condition type ('periodic' or 'nonperiodic')
    """

    def __init__(self, n: int, h: float, 
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic'):
        super().__init__(n, h)
        self.boundary = boundary
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> sparse.spmatrix:
        """Build the gradient operator matrix following MOLE implementation.
        
        The implementation follows src/matlab_octave/grad.m and src/cpp/gradient.cpp
        from the MOLE library, maintaining consistency with the original
        discretization scheme.

        Returns:
            scipy.sparse.spmatrix: Sparse matrix for gradient operator
        """
        if self.boundary == 'periodic':
            return self._build_periodic_matrix()
        else:
            return self._build_nonperiodic_matrix()

    def _build_periodic_matrix(self) -> sparse.spmatrix:
        """Build periodic boundary condition gradient matrix."""
        # Following MOLE's gradPeriodic.m implementation
        n = self.n
        h = self.h
        val = 1.0 / (2.0 * h)
        
        return _build_periodic_tridiag(n, (0.0, val, -val))

    def _build_nonperiodic_matrix(self) -> sparse.spmatrix:
        """Build non-periodic boundary condition gradient matrix."""
        # Following MOLE's gradNonPeriodic.m implementation
        n = self.n
        h = self.h
        
        # Define boundary and interior values
        boundary_vals = ((-1.0/h, 1.0/h), (-1.0/h, 1.0/h))  # (first, last)
        interior_vals = (-0.5/h, 0.0, 0.5/h)  # (lower, main, upper)
        
        return _build_nonperiodic_tridiag(n, boundary_vals, interior_vals)

class MimeticDivergence(MimeticOperator):
    """1D Mimetic divergence operator.
    
    Implementation follows MOLE's div.m and divergence.cpp.
    Preserves discrete properties including:
    - Discrete integration by parts with gradient
    - Exact divergence of constant vector fields
    - Proper boundary condition handling

    Args:
        n: Number of grid points
        h: Grid spacing
        boundary: Boundary condition type ('periodic' or 'nonperiodic')
    """

    def __init__(self, n: int, h: float, 
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic'):
        super().__init__(n, h)
        self.boundary = boundary
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> sparse.spmatrix:
        """Build divergence operator following MOLE implementation."""
        if self.boundary == 'periodic':
            return self._build_periodic_matrix()
        else:
            return self._build_nonperiodic_matrix()

    def _build_periodic_matrix(self) -> sparse.spmatrix:
        """Build periodic boundary condition divergence matrix."""
        n = self.n
        h = self.h
        val = 0.5 / h
        
        return _build_periodic_tridiag(n, (0.0, val, -val))

    def _build_nonperiodic_matrix(self) -> sparse.spmatrix:
        """Build non-periodic boundary condition divergence matrix."""
        n = self.n
        h = self.h
        
        # Define boundary and interior values
        boundary_vals = ((-1.0/h, 1.0/h), (-1.0/h, 1.0/h))  # (first, last)
        interior_vals = (-0.5/h, 0.0, 0.5/h)  # (lower, main, upper)
        
        return _build_nonperiodic_tridiag(n, boundary_vals, interior_vals)
        
class MimeticLaplacian(MimeticOperator):
    """1D Mimetic Laplacian operator.

    Implementation follows MOLE's lap.m and laplacian.cpp.
    Constructed as the composition of divergence and gradient,
    maintaining properties including:
    - Self-adjointness
    - Negative semi-definiteness
    - Proper null space (constants)

    Args:
        n: Number of grid points
        h: Grid spacing
        boundary: Boundary condition type ('periodic' or 'nonperiodic')
    """

    def __init__(self, n: int, h: float,
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic'):
        super().__init__(n, h)
        self.boundary = boundary
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> spmatrix:
        """Build Laplacian operator following MOLE implementation."""
        n = self.n
        h = self.h
        val = 1.0 / (h * h)

        if self.boundary == 'periodic':
            return _build_periodic_tridiag(n, (-2.0 * val, val, val))
        else:
            # Define boundary and interior values
            boundary_vals = ((-val, val), (val, -val))  # (first, last)
            interior_vals = (val, -2.0*val, val)  # (lower, main, upper)
            
            return _build_nonperiodic_tridiag(n, boundary_vals, interior_vals)
