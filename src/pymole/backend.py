"""Backend selection module for PyMOLE.

This module handles the selection and management of computational backends.
Available backends:
- 'python': Pure Python implementation (always available)
- 'cpp': C++ implementation (requires compilation during installation)
"""

import os
import threading
import warnings
from typing import Literal

# Thread-local storage for backend setting
_local = threading.local()

# Default backend from environment or 'python'
_default_backend = os.environ.get('PYMOLE_BACKEND', 'python').lower()

def _initialize_backend() -> None:
    """Initialize the backend if not already set."""
    if not hasattr(_local, 'backend'):
        _local.backend = _default_backend
        
def use_backend(backend: Literal['python', 'cpp']) -> None:
    """Set the computational backend for PyMOLE.
    
    Args:
        backend: Either 'python' or 'cpp'
            - 'python': Pure Python implementation
            - 'cpp': C++ implementation (requires pymole[cpp] installation)
    
    Raises:
        ValueError: If backend is not 'python' or 'cpp'
        ImportError: If cpp backend is requested but not available
    """
    if backend not in ('python', 'cpp'):
        raise ValueError("Backend must be 'python' or 'cpp'")
    
    if backend == 'cpp':
        try:
            from . import cpp
        except ImportError:
            warnings.warn(
                "C++ backend requested but not available. "
                "Install with 'pip install pymole[cpp]' to use C++ backend. "
                "Falling back to Python backend.",
                RuntimeWarning
            )
            backend = 'python'
    
    _initialize_backend()
    _local.backend = backend

def get_backend() -> Literal['python', 'cpp']:
    """Get the current computational backend.
    
    Returns:
        str: Current backend ('python' or 'cpp')
    """
    _initialize_backend()
    return _local.backend
