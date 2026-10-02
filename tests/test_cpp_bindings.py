"""Tests for the optional native operator bindings."""

import importlib
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from pymole.pure.operators import (
    MimeticGradient as PureGradient,
    MimeticInterpol as PureInterpol,
)


def _native_backend() -> Any:
    try:
        return importlib.import_module("pymole.cpp")
    except ImportError as exc:
        cause = exc.__cause__
        cause_name = getattr(cause, "name", "")
        if isinstance(cause, ModuleNotFoundError) and cause_name.endswith("._operators"):
            pytest.skip(f"Native extension is not installed: {exc}")
        if isinstance(cause, ImportError) and "cannot import name '_operators'" in str(cause):
            pytest.skip(f"Native extension is not installed: {exc}")
        raise


def test_gradient_matches_python_and_exports_csc():
    """Check native gradient parity and CSC matrix export."""
    native = _native_backend()
    grid_size, spacing = 20, 0.05
    native_gradient = native.MimeticGradient(grid_size, spacing)
    python_gradient = PureGradient(grid_size, spacing)

    assert sparse.isspmatrix_csc(native_gradient.matrix)
    np.testing.assert_allclose(
        native_gradient.matrix.toarray(),
        sparse.csc_matrix(python_gradient.matrix).toarray(),
        rtol=0,
        atol=1e-12,
    )

    values = np.linspace(0.0, 1.0, grid_size + 2)
    np.testing.assert_allclose(
        native_gradient @ values,
        native_gradient.matrix @ values,
        rtol=0,
        atol=1e-12,
    )


def test_divergence_application_matches_matrix():
    """Check native divergence application against its sparse matrix."""
    native = _native_backend()
    grid_size, spacing = 20, 0.05
    divergence = native.MimeticDivergence(grid_size, spacing)
    values = np.linspace(0.0, 1.0, grid_size + 1)

    assert divergence.matrix.shape == (grid_size + 2, grid_size + 1)
    np.testing.assert_allclose(
        divergence @ values,
        divergence.matrix @ values,
        rtol=0,
        atol=1e-12,
    )


def test_laplacian_is_divergence_composed_with_gradient():
    """Check the native Laplacian equals divergence composed with gradient."""
    native = _native_backend()
    grid_size, spacing = 20, 0.05
    gradient = native.MimeticGradient(grid_size, spacing)
    divergence = native.MimeticDivergence(grid_size, spacing)
    laplacian = native.MimeticLaplacian(grid_size, spacing)

    np.testing.assert_allclose(
        laplacian.matrix.toarray(),
        (divergence.matrix @ gradient.matrix).toarray(),
        rtol=0,
        atol=1e-12,
    )


def test_interpolation_matches_python_before_apply():
    """Check sparse interpolation export before any operator application."""
    native = _native_backend()
    grid_size, spacing, weight = 9, 0.125, 0.3
    native_interpol = native.MimeticInterpol(grid_size, spacing, weight)
    python_interpol = PureInterpol(grid_size, spacing, weight)

    np.testing.assert_allclose(
        native_interpol.matrix.toarray(),
        sparse.csc_matrix(python_interpol.matrix).toarray(),
        rtol=0,
        atol=1e-12,
    )
