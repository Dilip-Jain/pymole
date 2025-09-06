"""Tests for PyMOLE operators."""

import numpy as np
import pytest
from pymole import create_gradient, use_backend
from pymole.pure.operators import MimeticGradient, MimeticDivergence, MimeticLaplacian

def test_operator_validation():
    """Test operator input validation."""
    with pytest.raises(ValueError):
        MimeticGradient(0, 1.0)  # Invalid grid points
    with pytest.raises(ValueError):
        MimeticGradient(10, -1.0)  # Invalid grid spacing

@pytest.mark.parametrize("n,h", [(10, 0.1), (100, 0.01)])
def test_gradient_constant(n, h):
    """Test gradient of constant function is zero."""
    grad = MimeticGradient(n, h)
    x = np.ones(n)
    np.testing.assert_allclose(grad @ x, np.zeros(n), atol=1e-14)

@pytest.mark.parametrize("n,h", [(10, 0.1), (100, 0.01)])
def test_gradient_linear(n, h):
    """Test gradient of linear function is constant."""
    grad = MimeticGradient(n, h)
    x = np.linspace(0, 1, n)
    expected = np.ones(n)
    np.testing.assert_allclose(grad @ x, expected, atol=1e-14)

@pytest.mark.parametrize("n,h", [(10, 0.1), (100, 0.01)])
def test_divergence_gradient_laplacian(n, h):
    """Test that div(grad(x)) equals Laplacian(x)."""
    grad = MimeticGradient(n, h)
    div = MimeticDivergence(n, h)
    lap = MimeticLaplacian(n, h)
    
    x = np.random.rand(n)
    div_grad = div @ (grad @ x)
    lap_x = lap @ x
    
    np.testing.assert_allclose(div_grad, lap_x, atol=1e-14)

@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_backends(backend):
    """Test backend selection."""
    use_backend(backend)
    n, h = 10, 0.1
    try:
        grad = create_gradient(n, h)
        x = np.ones(n)
        result = grad @ x
        assert result.shape == (n,)
    except ImportError:
        if backend == "python":
            raise
        pytest.skip("C++ backend not available")
