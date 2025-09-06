"""Base classes and interfaces for PyMOLE operators."""

from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse

class MimeticOperator(ABC):
    """Abstract base class for all mimetic operators.
    
    Attributes:
        n (int): Number of grid points
        h (float): Grid spacing
        _matrix (scipy.sparse.spmatrix): Sparse matrix representation of the operator
    """
    
    def __init__(self, n: int, h: float):
        """Initialize the mimetic operator.
        
        Args:
            n: Number of grid points
            h: Grid spacing
        """
        if n <= 0:
            raise ValueError("Number of grid points must be positive")
        if h <= 0:
            raise ValueError("Grid spacing must be positive")
            
        self.n = n
        self.h = h
        self._matrix = None
        
    @abstractmethod
    def _build_matrix(self) -> sparse.spmatrix:
        """Build the operator matrix.
        
        Returns:
            scipy.sparse.spmatrix: Sparse matrix representation
        """
        pass
    
    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Implement matrix multiplication (@) operator.
        
        Args:
            x: Input vector/matrix
            
        Returns:
            numpy.ndarray: Result of operator application
        """
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix @ x
        
    @property
    def matrix(self) -> sparse.spmatrix:
        """Get the operator matrix.
        
        Returns:
            scipy.sparse.spmatrix: Sparse matrix representation
        """
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix
