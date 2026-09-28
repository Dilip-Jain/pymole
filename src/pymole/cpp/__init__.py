"""C++ implementation bindings for mimetic operators."""

import numpy as np
from scipy import sparse
from typing import Literal, Union

try:
    from . import _operators
except ImportError:
    raise ImportError(
        "C++ implementation not available. "
        "Please install pymole with C++ support using: "
        "pip install pymole[cpp]"
    )

from ..base import MimeticOperator

# Export C++ operators directly
Gradient = _operators.Gradient
Divergence = _operators.Divergence
Laplacian = _operators.Laplacian
Interpol = _operators.Interpol
RobinBC = _operators.RobinBC
MixedBC = _operators.MixedBC

__all__ = [
    'Gradient',
    'Divergence', 
    'Laplacian',
    'Interpol',
    'RobinBC',
    'MixedBC',
    'MimeticGradient',
    'MimeticDivergence',
    'MimeticLaplacian',
    'MimeticInterpol',
    'MimeticRobinBC',
    'MimeticMixedBC',
]


class MimeticGradient(MimeticOperator):
    """1D/2D Mimetic gradient operator using C++ MOLE implementation.
    
    Attributes:
        n: Number of grid points
        h: Grid spacing
        k: Order of accuracy
        _cpp_operator: Underlying C++ gradient operator
    """
    
    def __init__(self, n: int, h: float, k: int = 2):
        """Initialize the C++ gradient operator.
        
        Args:
            n: Number of grid points
            h: Grid spacing
            k: Order of accuracy (default: 2)
        """
        super().__init__(n, h)
        self.k = k
        # Create C++ operator (1D)
        self._cpp_operator = _operators.Gradient(k, n, h)
        self._matrix = None
    
    def _build_matrix(self) -> sparse.spmatrix:
        """Convert C++ sparse matrix to scipy format."""
        return self._cpp_operator.to_scipy_sparse()
    
    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix
    
    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply gradient operator: result = G @ x"""
        if x.ndim == 1:
            return self._cpp_operator @ x
        else:
            # Apply to each column for 2D arrays
            return np.column_stack([self._cpp_operator @ x[:, i] for i in range(x.shape[1])])
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to vector x."""
        return self @ x


class MimeticDivergence(MimeticOperator):
    """1D/2D Mimetic divergence operator using C++ MOLE implementation.
    
    Attributes:
        n: Number of grid points
        h: Grid spacing
        k: Order of accuracy
        _cpp_operator: Underlying C++ divergence operator
    """
    
    def __init__(self, n: int, h: float, k: int = 2):
        """Initialize the C++ divergence operator.
        
        Args:
            n: Number of grid points
            h: Grid spacing
            k: Order of accuracy (default: 2)
        """
        super().__init__(n, h)
        self.k = k
        # Create C++ operator (1D)
        self._cpp_operator = _operators.Divergence(k, n, h)
        self._matrix = None
    
    def _build_matrix(self) -> sparse.spmatrix:
        """Convert C++ sparse matrix to scipy format."""
        return self._cpp_operator.to_scipy_sparse()
    
    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix
    
    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply divergence operator: result = D @ x"""
        if x.ndim == 1:
            return self._cpp_operator @ x
        else:
            # Apply to each column for 2D arrays
            return np.column_stack([self._cpp_operator @ x[:, i] for i in range(x.shape[1])])
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to vector x."""
        return self @ x


class MimeticLaplacian(MimeticOperator):
    """1D/2D/3D Mimetic Laplacian operator using C++ MOLE implementation.
    
    Attributes:
        n: Number of grid points (1D) or grid dimensions (2D/3D)
        h: Grid spacing
        k: Order of accuracy
        _cpp_operator: Underlying C++ Laplacian operator
    """
    
    def __init__(self, n: Union[int, tuple], h: Union[float, tuple], k: int = 2):
        """Initialize the C++ Laplacian operator.
        
        Args:
            n: Number of grid points or tuple of (m, n) for 2D or (m, n, o) for 3D
            h: Grid spacing or tuple of (dx, dy) for 2D or (dx, dy, dz) for 3D
            k: Order of accuracy (default: 2)
        """
        if isinstance(n, tuple):
            n_grid_points = n[0]  # Use first dimension for base class
        else:
            n_grid_points = n
            
        if isinstance(h, tuple):
            h_min = min(h)  # Use minimum spacing for base class
        else:
            h_min = h
            
        super().__init__(n_grid_points, h_min)
        self.k = k
        
        # Create appropriate C++ operator based on dimensions
        if isinstance(n, tuple) and isinstance(h, tuple):
            if len(n) == 1:
                # 1D
                self._cpp_operator = _operators.Laplacian(k, n[0], h[0])
            elif len(n) == 2:
                # 2D
                self._cpp_operator = _operators.Laplacian(k, n[0], n[1], h[0], h[1])
            elif len(n) == 3:
                # 3D
                self._cpp_operator = _operators.Laplacian(k, n[0], n[1], n[2], h[0], h[1], h[2])
            else:
                raise ValueError("Unsupported dimensionality")
        else:
            # 1D
            self._cpp_operator = _operators.Laplacian(k, n, h)
        
        self._matrix = None
    
    def _build_matrix(self) -> sparse.spmatrix:
        """Convert C++ sparse matrix to scipy format."""
        return self._cpp_operator.to_scipy_sparse()
    
    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix
    
    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply Laplacian operator: result = L @ x"""
        if x.ndim == 1:
            return self._cpp_operator @ x
        else:
            # Apply to each column for 2D arrays
            return np.column_stack([self._cpp_operator @ x[:, i] for i in range(x.shape[1])])
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to vector x."""
        return self @ x


class MimeticInterpol(MimeticOperator):
    """1D/2D Mimetic interpolation operator using C++ MOLE implementation.
    
    Attributes:
        n: Number of grid points
        h: Grid spacing
        k: Order of accuracy
        _cpp_operator: Underlying C++ interpolation operator
    """
    
    def __init__(self, n: int, h: float, k: int = 2):
        """Initialize the C++ interpolation operator.
        
        Args:
            n: Number of grid points
            h: Grid spacing
            k: Order of accuracy (default: 2)
        """
        super().__init__(n, h)
        self.k = k
        # Create C++ operator (1D)
        self._cpp_operator = _operators.Interpol(k, n, h)
        self._matrix = None
    
    def _build_matrix(self) -> sparse.spmatrix:
        """Convert C++ sparse matrix to scipy format."""
        return self._cpp_operator.to_scipy_sparse()
    
    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix
    
    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply interpolation operator: result = I @ x"""
        return np.asarray(self._cpp_operator @ x)
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to vector x."""
        return self @ x


class MimeticRobinBC(MimeticOperator):
    """Mimetic Robin Boundary Condition operator using C++ MOLE implementation.
    
    Robin BC: a*u + b*du/dn = f
    
    Attributes:
        n: Number of grid points or tuple for multi-dimensional
        h: Grid spacing or tuple for multi-dimensional
        k: Order of accuracy
        a: Coefficient of Dirichlet component
        b: Coefficient of Neumann component
        _cpp_operator: Underlying C++ RobinBC operator
    """
    
    def __init__(self, n: Union[int, tuple], h: Union[float, tuple], k: int = 2, a: float = 1.0, b: float = 0.0):
        """Initialize the C++ Robin BC operator.
        
        Args:
            n: Number of grid points or tuple of (m, n) for 2D or (m, n, o) for 3D
            h: Grid spacing or tuple of (dx, dy) for 2D or (dx, dy, dz) for 3D
            k: Order of accuracy (default: 2)
            a: Coefficient of Dirichlet component (default: 1.0)
            b: Coefficient of Neumann component (default: 0.0)
        """
        if isinstance(n, tuple):
            n_grid_points = n[0]
        else:
            n_grid_points = n
            
        if isinstance(h, tuple):
            h_min = min(h)
        else:
            h_min = h
            
        super().__init__(n_grid_points, h_min)
        self.k = k
        self.a = a
        self.b = b
        
        # Create appropriate C++ operator based on dimensions
        if isinstance(n, tuple) and isinstance(h, tuple):
            if len(n) == 1:
                # 1D
                self._cpp_operator = _operators.RobinBC(k, n[0], h[0], a, b)
            elif len(n) == 2:
                # 2D
                self._cpp_operator = _operators.RobinBC(k, n[0], h[0], n[1], h[1], a, b)
            elif len(n) == 3:
                # 3D
                self._cpp_operator = _operators.RobinBC(k, n[0], h[0], n[1], h[1], n[2], h[2], a, b)
            else:
                raise ValueError("Unsupported dimensionality")
        else:
            # 1D
            self._cpp_operator = _operators.RobinBC(k, n, h, a, b)
        
        self._matrix = None
    
    def _build_matrix(self) -> sparse.spmatrix:
        """Convert C++ sparse matrix to scipy format."""
        return self._cpp_operator.to_scipy_sparse()
    
    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix
    
    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply Robin BC operator: result = R @ x"""
        return self._cpp_operator @ x
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to vector x."""
        return self @ x


class MimeticMixedBC(MimeticOperator):
    """Mimetic Mixed Boundary Condition operator using C++ MOLE implementation.
    
    Supports mixed boundary conditions with different types on each boundary.
    Each boundary can be: 'Dirichlet', 'Neumann', or 'Robin'.
    
    Attributes:
        n: Number of grid points or tuple for multi-dimensional
        h: Grid spacing or tuple for multi-dimensional
        k: Order of accuracy
        bc_dict: Dictionary specifying boundary conditions and coefficients
        _cpp_operator: Underlying C++ MixedBC operator
    """
    
    def __init__(self, n: Union[int, tuple], h: Union[float, tuple], k: int = 2, **bc_dict):
        """Initialize the C++ Mixed BC operator.
        
        Args:
            n: Number of grid points or tuple of (m, n) for 2D or (m, n, o) for 3D
            h: Grid spacing or tuple of (dx, dy) for 2D or (dx, dy, dz) for 3D
            k: Order of accuracy (default: 2)
            **bc_dict: Boundary condition specifications as keyword arguments.
                       For 1D: left, coeffs_left, right, coeffs_right
                       For 2D: left, coeffs_left, right, coeffs_right, bottom, coeffs_bottom, top, coeffs_top
                       For 3D: left, coeffs_left, right, coeffs_right, bottom, coeffs_bottom, 
                               top, coeffs_top, front, coeffs_front, back, coeffs_back
        """
        if isinstance(n, tuple):
            n_grid_points = n[0]
        else:
            n_grid_points = n
            
        if isinstance(h, tuple):
            h_min = min(h)
        else:
            h_min = h
            
        super().__init__(n_grid_points, h_min)
        self.k = k
        self._bc_dict = bc_dict
        
        # Create appropriate C++ operator based on dimensions
        if isinstance(n, tuple) and isinstance(h, tuple):
            if len(n) == 1:
                # 1D
                self._cpp_operator = _operators.MixedBC(
                    k, n[0], h[0],
                    bc_dict.get('left'), bc_dict.get('coeffs_left', []),
                    bc_dict.get('right'), bc_dict.get('coeffs_right', [])
                )
            elif len(n) == 2:
                # 2D
                self._cpp_operator = _operators.MixedBC(
                    k, n[0], h[0], n[1], h[1],
                    bc_dict.get('left'), bc_dict.get('coeffs_left', []),
                    bc_dict.get('right'), bc_dict.get('coeffs_right', []),
                    bc_dict.get('bottom'), bc_dict.get('coeffs_bottom', []),
                    bc_dict.get('top'), bc_dict.get('coeffs_top', [])
                )
            elif len(n) == 3:
                # 3D
                self._cpp_operator = _operators.MixedBC(
                    k, n[0], h[0], n[1], h[1], n[2], h[2],
                    bc_dict.get('left'), bc_dict.get('coeffs_left', []),
                    bc_dict.get('right'), bc_dict.get('coeffs_right', []),
                    bc_dict.get('bottom'), bc_dict.get('coeffs_bottom', []),
                    bc_dict.get('top'), bc_dict.get('coeffs_top', []),
                    bc_dict.get('front'), bc_dict.get('coeffs_front', []),
                    bc_dict.get('back'), bc_dict.get('coeffs_back', [])
                )
            else:
                raise ValueError("Unsupported dimensionality")
        else:
            # 1D
            self._cpp_operator = _operators.MixedBC(
                k, n, h,
                bc_dict.get('left'), bc_dict.get('coeffs_left', []),
                bc_dict.get('right'), bc_dict.get('coeffs_right', [])
            )
        
        self._matrix = None
    
    def _build_matrix(self) -> sparse.spmatrix:
        """Convert C++ sparse matrix to scipy format."""
        return self._cpp_operator.to_scipy_sparse()
    
    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix
    
    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply Mixed BC operator: result = M @ x"""
        return self._cpp_operator @ x
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to vector x."""
        return self @ x
