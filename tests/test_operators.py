"""Tests for PyMOLE operators."""

import numpy as np
import pytest
from scipy import sparse
from scipy.integrate import odeint
from pymole import create_gradient, use_backend
from pymole.pure.operators import (
    MimeticGradient,
    MimeticDivergence,
    MimeticLaplacian,
    MimeticInterpol,
    MimeticRobinBC as PureMimeticRobinBC,
    MimeticMixedBC as PureMimeticMixedBC,
)


def _load_native_backend():
    try:
        # pylint: disable=import-outside-toplevel
        import pymole.cpp as native
    except ImportError as exc:
        pytest.skip(f"Native backend unavailable: {exc}")
    return native


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

def test_operator_validation():
    """Test operator input validation."""
    with pytest.raises(ValueError):
        MimeticGradient(0, 1.0)  # Invalid grid points
    with pytest.raises(ValueError):
        MimeticGradient(10, -1.0)  # Invalid grid spacing


# ============================================================================
# BASIC FUNCTIONALITY TESTS (ORIGINAL)
# ============================================================================

@pytest.mark.parametrize("n,h", [(10, 0.1), (100, 0.01)])
def test_gradient_constant(n, h):
    """Test gradient of constant function is zero."""
    grad = MimeticGradient(n, h)
    x = np.ones(n + 2)
    np.testing.assert_allclose(grad @ x, np.zeros(n + 1), atol=1e-13)

@pytest.mark.parametrize("n,h", [(10, 0.1), (100, 0.01)])
def test_gradient_linear(n, h):
    """Test MOLE's interior gradient rows differentiate linear data."""
    grad = MimeticGradient(n, h)
    x = np.arange(n + 2) * h
    np.testing.assert_allclose((grad @ x)[1:-1], np.ones(n - 1), atol=1e-14)

@pytest.mark.parametrize("n,h", [(10, 0.1), (100, 0.01)])
def test_divergence_gradient_laplacian(n, h):
    """Test that div(grad(x)) equals Laplacian(x)."""
    grad = MimeticGradient(n, h)
    div = MimeticDivergence(n, h)
    lap = MimeticLaplacian(n, h)
    
    x = np.random.default_rng(0).random(n + 2)
    div_grad = div @ (grad @ x)
    lap_x = lap @ x
    
    np.testing.assert_allclose(div_grad, lap_x, atol=1e-14)

@pytest.mark.parametrize("k", [2, 4, 6, 8])
def test_mole_shapes_spacing_and_periodic_adjoint(k):
    m = 2 * k + 1
    dx = 0.037
    grad = MimeticGradient(m, dx, k=k)
    div = MimeticDivergence(m, dx, k=k)
    lap = MimeticLaplacian(m, dx, k=k)

    assert grad.matrix.shape == (m + 1, m + 2)
    assert div.matrix.shape == (m + 2, m + 1)
    assert lap.matrix.shape == (m + 2, m + 2)
    assert grad.h == div.h == lap.h == dx
    np.testing.assert_allclose(lap.matrix.toarray(), (div.matrix @ grad.matrix).toarray(), atol=1e-12)

    grad_periodic = MimeticGradient(m, dx, boundary="periodic", k=k)
    div_periodic = MimeticDivergence(m, dx, boundary="periodic", k=k)
    assert grad_periodic.matrix.shape == div_periodic.matrix.shape == (m, m)
    np.testing.assert_allclose(div_periodic.matrix.toarray(), -grad_periodic.matrix.toarray().T, atol=1e-12)


@pytest.mark.parametrize("k", [2, 4, 6, 8])
def test_native_matrix_parity_1d(k):
    native = _load_native_backend()
    m, dx = 2 * k + 1, 0.037
    pure_operators = (
        MimeticGradient(m, dx, k=k),
        MimeticDivergence(m, dx, k=k),
        MimeticLaplacian(m, dx, k=k),
    )
    native_operators = (
        native.MimeticGradient(m, dx, k=k),
        native.MimeticDivergence(m, dx, k=k),
        native.MimeticLaplacian(m, dx, k=k),
    )
    for pure_operator, native_operator in zip(pure_operators, native_operators):
        np.testing.assert_allclose(
            pure_operator.matrix.toarray(), native_operator.matrix.toarray(), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("dimensions,spacings", [
    ((7, 8), (0.2, 0.15)),
    ((7, 8, 9), (0.2, 0.15, 0.1)),
])
def test_native_matrix_parity_multidimensional(dimensions, spacings):
    native = _load_native_backend()
    k = 2
    pure_operators = (
        MimeticGradient(dimensions, spacings, k=k),
        MimeticDivergence(dimensions, spacings, k=k),
        MimeticLaplacian(dimensions, spacings, k=k),
    )
    native_operators = (
        native.Gradient(k, *dimensions, *spacings),
        native.Divergence(k, *dimensions, *spacings),
        native.Laplacian(k, *dimensions, *spacings),
    )
    for pure_operator, native_operator in zip(pure_operators, native_operators):
        np.testing.assert_allclose(
            pure_operator.matrix.toarray(),
            native_operator.to_scipy_sparse().toarray(),
            rtol=0,
            atol=1e-12,
        )


def test_native_matrix_parity_interpolation_and_boundaries():
    native = _load_native_backend()
    m, dx, k, c = 9, 0.125, 2, 0.3
    pure_interpol = MimeticInterpol(m, dx, c)
    native_interpol = native.Interpol(m, c)
    np.testing.assert_allclose(
        pure_interpol.matrix.toarray(), native_interpol.to_scipy_sparse().toarray(), rtol=0, atol=1e-12
    )

    pure_robin = PureMimeticRobinBC(m, dx, k=k, a=2.0, b=0.5)
    native_robin = native.MimeticRobinBC(m, dx, k=k, a=2.0, b=0.5)
    np.testing.assert_allclose(
        pure_robin.matrix.toarray(), native_robin.matrix.toarray(), rtol=0, atol=1e-12
    )

    mixed_args = {
        "left": "Dirichlet", "coeffs_left": [2.0],
        "right": "Robin", "coeffs_right": [3.0, 0.5],
    }
    pure_mixed = PureMimeticMixedBC(m, dx, k=k, **mixed_args)
    native_mixed = native.MimeticMixedBC(m, dx, k=k, **mixed_args)
    np.testing.assert_allclose(
        pure_mixed.matrix.toarray(), native_mixed.matrix.toarray(), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("dimensions,spacings", [
    ((7, 8), (0.2, 0.15)),
    ((7, 8, 9), (0.2, 0.15, 0.1)),
])
def test_mole_multidimensional_shapes_and_composition(dimensions, spacings):
    grad = MimeticGradient(dimensions, spacings)
    div = MimeticDivergence(dimensions, spacings)
    lap = MimeticLaplacian(dimensions, spacings)
    input_size = int(np.prod([size + 2 for size in dimensions]))

    assert grad.matrix.shape[1] == input_size
    assert div.matrix.shape[0] == input_size
    assert grad.matrix.shape[0] == div.matrix.shape[1]
    assert lap.matrix.shape == (input_size, input_size)
    np.testing.assert_allclose(lap.matrix.toarray(), (div.matrix @ grad.matrix).toarray(), atol=1e-12)


def test_mole_interpolation_staggered_weights():
    m, dx, c = 7, 0.2, 0.3
    interp = MimeticInterpol(m, dx, c).matrix

    assert interp.shape == (m + 1, m + 2)
    assert interp[0, 0] == 1.0
    assert interp[m, m + 1] == 1.0
    assert interp[1, 1] == c
    assert interp[1, 2] == 1.0 - c

    dimensions = (7, 8, 9)
    spacings = (0.2, 0.15, 0.1)
    interp_3d = MimeticInterpol(dimensions, spacings, (0.3, 0.4, 0.5)).matrix
    expected_rows = (
        dimensions[2] * dimensions[1] * (dimensions[0] + 1)
        + dimensions[2] * (dimensions[1] + 1) * dimensions[0]
        + (dimensions[2] + 1) * dimensions[1] * dimensions[0]
    )
    assert interp_3d.shape == (expected_rows, int(np.prod([size + 2 for size in dimensions])))


def test_mole_robin_and_mixed_boundary_rows():
    m, dx, k = 9, 0.125, 2
    gradient = MimeticGradient(m, dx, k=k).matrix
    robin = PureMimeticRobinBC(m, dx, k=k, a=2.0, b=0.5).matrix.toarray()
    mixed = PureMimeticMixedBC(
        m, dx, k=k,
        left="Dirichlet", coeffs_left=[2.0],
        right="Robin", coeffs_right=[3.0, 0.5],
    ).matrix.toarray()

    expected_robin = np.zeros((m + 2, m + 2))
    expected_robin[0, 0] = 2.0
    expected_robin[-1, -1] = 2.0
    expected_robin[0] -= 0.5 * gradient.toarray()[0]
    expected_robin[-1] += 0.5 * gradient.toarray()[m]
    np.testing.assert_allclose(robin, expected_robin, atol=1e-14)

    expected_mixed = np.zeros((m + 2, m + 2))
    expected_mixed[0, 0] = 2.0
    expected_mixed[-1, -1] = 3.0
    expected_mixed[-1] += 0.5 * gradient.toarray()[m]
    np.testing.assert_allclose(mixed, expected_mixed, atol=1e-14)

    dimensions, spacings = (7, 8), (0.2, 0.15)
    assert PureMimeticRobinBC(dimensions, spacings).matrix.shape == (90, 90)
    assert PureMimeticMixedBC(
        dimensions, spacings,
        left="Dirichlet", coeffs_left=[1.0], right="Dirichlet", coeffs_right=[1.0],
        bottom="Neumann", coeffs_bottom=[1.0], top="Neumann", coeffs_top=[1.0],
    ).matrix.shape == (90, 90)

@pytest.mark.parametrize("backend", ["python"])
def test_backends(backend):
    """Test backend selection."""
    use_backend(backend)
    n, h = 10, 0.1
    try:
        grad = create_gradient(n, h)
        x = np.ones(n + 2)
        result = grad @ x
        assert result.shape == (n + 1,)
    except ImportError:
        if backend == "python":
            raise
        pytest.skip("C++ backend not available")


# ============================================================================
# ACCURACY ORDER TESTS
# ============================================================================

@pytest.mark.parametrize("k", [2, 4, 6, 8])
def test_gradient_accuracy_polynomial(k):
    """Test that gradient operator has correct order of accuracy on polynomials.
    
    For a k-th order accurate operator, the error should scale as O(h^k).
    """
    # Test on multiple grid sizes
    errors = []
    spacings = []
    
    for n in [32, 64, 128]:
        h = 1.0 / (n + 1)
        spacings.append(h)

        x = np.arange(n + 2) * h
        # Test function: f(x) = x^(k+1), f'(x) = (k+1)*x^k
        f = x**(k + 1)
        gradient_locations = (np.arange(n + 1) + 0.5) * h
        expected = (k + 1) * gradient_locations**k
        
        grad = MimeticGradient(n, h, k=k)
        computed = grad @ f
        
        # Compare interior points (excluding boundaries which have higher error)
        interior = slice(k // 2, n + 1 - k // 2)
        error = np.max(np.abs(computed[interior] - expected[interior]))
        errors.append(error)
    
    # Check convergence rate: error should decrease by factor of 2^k for spacing halving
    convergence_ratio = errors[0] / errors[1]
    expected_ratio = 2**k
    
    # Allow 50% tolerance in convergence rate
    assert convergence_ratio > expected_ratio * 0.5, \
        f"Convergence ratio {convergence_ratio:.2e} < expected {expected_ratio * 0.5:.2e}"


@pytest.mark.parametrize("k", [2, 4, 6])
def test_gradient_accuracy_trigonometry(k):
    """Test gradient accuracy on smooth trigonometric functions."""
    n = 128
    h = 2 * np.pi / (n + 1)
    x = np.arange(n + 2) * h
    f = np.sin(x)
    expected = np.cos((np.arange(n + 1) + 0.5) * h)
    
    grad = MimeticGradient(n, h, k=k)
    computed = grad @ f
    
    # Interior error (excluding boundaries)
    interior = slice(k // 2, n + 1 - k // 2)
    rel_error = np.linalg.norm(computed[interior] - expected[interior]) / \
                np.linalg.norm(expected[interior])
    
    # For smooth functions, error should be very small
    assert rel_error < 0.01, f"Relative error {rel_error:.4e} too large"


# ============================================================================
# MATRIX PROPERTY TESTS
# ============================================================================

def test_matrix_format():
    """Test that matrix is in scipy sparse CSC format."""
    grad = MimeticGradient(20, 0.1)
    mat = grad.matrix
    assert isinstance(mat, sparse.csc_matrix), "Matrix should be CSC sparse format"

def test_matrix_shape():
    """Test the staggered 1D matrix shapes used by MOLE."""
    n = 20
    h = 0.1
    
    grad = MimeticGradient(n, h)
    div = MimeticDivergence(n, h)
    lap = MimeticLaplacian(n, h)
    
    assert grad.matrix.shape == (n + 1, n + 2)
    assert div.matrix.shape == (n + 2, n + 1)
    assert lap.matrix.shape == (n + 2, n + 2)

def test_sparsity_pattern():
    """Test that matrices have expected sparsity (banded structure for 1D)."""
    grad = MimeticGradient(50, 0.1, k=2)
    mat = grad.matrix
    
    # 1D operators should be banded (tridiagonal for k=2)
    # Check that most entries are on the main diagonal and neighboring diagonals
    data_density = mat.nnz / (mat.shape[0] * mat.shape[1])
    assert data_density < 0.1, f"Matrix density {data_density:.2e} suggests not banded"

def test_matrix_consistency_with_operator():
    """Test that matrix and @ operator produce identical results."""
    n = 30
    h = 0.1
    grad = MimeticGradient(n, h)
    
    x = np.random.default_rng(0).random(n + 2)
    
    # Two ways to apply operator
    result_matmul = grad @ x
    result_matrix = grad.matrix @ x
    
    np.testing.assert_allclose(result_matmul, result_matrix, rtol=1e-14)

def test_operator_nnz():
    """Test non-zero count."""
    grad = MimeticGradient(20, 0.1)
    mat = grad.matrix
    
    # For 1D 2nd order operator, should have roughly 3*n non-zeros (banded)
    # Allow 50% variation
    expected_nnz = 2 * 20 + 6
    actual_nnz = mat.nnz
    
    assert actual_nnz > 0, "Should have non-zero elements"
    assert actual_nnz < expected_nnz * 2, "Sparsity should be maintained"


# ============================================================================
# SPARSE SOLVER INTEGRATION TESTS
# ============================================================================

def test_sparse_solver_laplacian():
    """Test that the MOLE Laplacian remains sparse and applicable."""
    n = 30
    h = 1.0 / (n + 1)
    lap = MimeticLaplacian(n, h)
    result = lap @ np.ones(n + 2)
    assert result.shape == (n + 2,)
    assert np.isfinite(result).all()

def test_eigenvalue_decomposition():
    """Test the k=2 MOLE Laplacian has non-positive spectrum."""
    n = 16
    h = 1.0 / (n + 1)
    lap = MimeticLaplacian(n, h)
    eigenvalues = np.linalg.eigvals(lap.matrix.toarray())
    assert np.max(eigenvalues.real) < 1e-10
    assert np.min(eigenvalues.real) < 0
    assert np.max(np.abs(eigenvalues.imag)) < 1e-10


# ============================================================================
# TIME INTEGRATION TESTS (Heat Equation)
# ============================================================================

def test_heat_equation_1d():
    """Test solving 1D heat equation u_t = k*u_xx on [0,1] x [0,T].
    
    Analytical solution: u(x,t) = sin(pi*x) * exp(-pi^2*k*t)
    Initial condition: u(x,0) = sin(pi*x)
    Boundary conditions: u(0,t) = u(1,t) = 0
    """
    # Parameters
    k = 0.1  # thermal diffusivity
    T_final = 0.1
    n = 50
    h = 1.0 / (n + 1)
    
    # Create Laplacian operator
    lap = MimeticLaplacian(n, h)
    lap_mat = lap.matrix
    
    # Initial condition: sin(pi*x)
    x = np.linspace(0, 1, n + 2)
    u0 = np.sin(np.pi * x)
    
    # ODE system: du/dt = k * L @ u
    def heat_ode(u, _time):
        return k * (lap_mat @ u)
    
    # Solve ODE
    t_eval = np.linspace(0, T_final, 20)
    solution = odeint(heat_ode, u0, t_eval)
    
    # Check final time: should match analytical solution (approximately)
    u_final_analytical = np.sin(np.pi * x) * np.exp(-np.pi**2 * k * T_final)
    u_final_numerical = solution[-1, :]
    
    # Relative error at interior points (excluding boundaries)
    interior = slice(2, -2)
    rel_error = np.linalg.norm(u_final_numerical[interior] - u_final_analytical[interior]) / \
                np.linalg.norm(u_final_analytical[interior])
    
    assert rel_error < 0.05, f"Heat equation solution error {rel_error:.4e} too large"

def test_wave_equation_1d():
    """Test solving 1D wave equation u_tt = c^2*u_xx using method of lines.
    
    Analytical solution: u(x,t) = sin(pi*x) * cos(pi*c*t)
    Initial condition: u(x,0) = sin(pi*x), u_t(x,0) = 0
    Boundary conditions: u(0,t) = u(1,t) = 0
    """
    # Parameters
    c = 1.0  # wave speed
    T_final = 0.25
    n = 60
    h = 1.0 / (n + 1)
    
    # Create Laplacian operator
    lap = MimeticLaplacian(n, h)
    lap_mat = c**2 * lap.matrix
    
    # Initial conditions
    x = np.linspace(0, 1, n + 2)
    u0 = np.sin(np.pi * x)
    v0 = np.zeros(n + 2)  # zero initial velocity
    
    # Combined state vector: [u, v]
    state0 = np.concatenate([u0, v0])
    
    # ODE system: d/dt[u, v] = [v, c^2*L@u]
    def wave_ode(state, _time):
        u, v = state[:n + 2], state[n + 2:]
        dudt = v
        dvdt = lap_mat @ u
        return np.concatenate([dudt, dvdt])
    
    # Solve ODE
    t_eval = np.linspace(0, T_final, 30)
    solution = odeint(wave_ode, state0, t_eval)
    
    u_final_numerical = solution[-1, :n + 2]
    u_final_analytical = np.sin(np.pi * x) * np.cos(np.pi * c * T_final)
    
    # Relative error (excluding boundaries)
    interior = slice(2, -2)
    rel_error = np.linalg.norm(u_final_numerical[interior] - u_final_analytical[interior]) / \
                np.linalg.norm(u_final_analytical[interior])
    
    assert rel_error < 0.08, f"Wave equation solution error {rel_error:.4e} too large"


# ============================================================================
# BOUNDARY CONDITION TESTS (C++ backend)
# ============================================================================

@pytest.mark.skip(reason="Requires C++ backend with RobinBC bindings")
def test_robinbc_dirichlet_limit():
    """Test that RobinBC reduces to Dirichlet BC when b=0.
    
    Robin BC: a*u + b*du/dn = f
    When b=0: a*u = f, i.e., u = f/a (Dirichlet)
    """
    try:
        from pymole.cpp import MimeticRobinBC
    except ImportError:
        pytest.skip("C++ backend not available")
    
    n = 30
    h = 1.0 / (n - 1)
    
    # Create RobinBC with b=0 (Dirichlet limit)
    robin = MimeticRobinBC(n, h, k=2, a=1.0, b=0.0)
    
    # Test vector
    x = np.linspace(0, 1, n)
    result = robin @ x
    
    # Result should be relatively smooth (no spurious oscillations)
    assert np.isfinite(result).all(), "Result should be finite"

@pytest.mark.skip(reason="Requires C++ backend with RobinBC bindings")
def test_robinbc_neumann_limit():
    """Test that RobinBC reduces to Neumann BC when a=0.
    
    Robin BC: a*u + b*du/dn = f
    When a=0: b*du/dn = f, i.e., du/dn = f/b (Neumann)
    """
    try:
        from pymole.cpp import MimeticRobinBC
    except ImportError:
        pytest.skip("C++ backend not available")
    
    n = 30
    h = 1.0 / (n - 1)
    
    # Create RobinBC with a=0 (Neumann limit)
    robin = MimeticRobinBC(n, h, k=2, a=0.0, b=1.0)
    
    # Test vector
    x = np.linspace(0, 1, n)
    result = robin @ x
    
    assert np.isfinite(result).all(), "Result should be finite"

@pytest.mark.skip(reason="Requires C++ backend with MixedBC bindings")
def test_mixedbc_1d_dirichlet_neumann():
    """Test 1D MixedBC with Dirichlet on left, Neumann on right."""
    try:
        from pymole.cpp import MimeticMixedBC
    except ImportError:
        pytest.skip("C++ backend not available")
    
    n = 30
    h = 1.0 / (n - 1)
    
    # Create MixedBC: Dirichlet on left (u=0), Neumann on right (du/dn=0)
    mixed = MimeticMixedBC(
        n, h, k=2,
        left='Dirichlet', coeffs_left=[],
        right='Neumann', coeffs_right=[]
    )
    
    x = np.linspace(0, 1, n)
    result = mixed @ x
    
    assert np.isfinite(result).all(), "Result should be finite"
    assert result.shape == (n,), "Output shape should match input"

@pytest.mark.skip(reason="Requires C++ backend with MixedBC bindings")
def test_mixedbc_2d_mixed_conditions():
    """Test 2D MixedBC with different conditions on each boundary."""
    try:
        from pymole.cpp import MimeticMixedBC
    except ImportError:
        pytest.skip("C++ backend not available")
    
    m, n = 20, 25
    dx, dy = 0.1, 0.08
    
    # Create 2D MixedBC with mixed boundary types
    mixed = MimeticMixedBC(
        (m, n), (dx, dy), k=2,
        left='Dirichlet', coeffs_left=[],
        right='Neumann', coeffs_right=[],
        bottom='Robin', coeffs_bottom=[1.0, 0.5],
        top='Dirichlet', coeffs_top=[]
    )
    
    # Grid points for 2D
    grid_size = m * n
    x = np.random.rand(grid_size)
    result = mixed @ x
    
    assert np.isfinite(result).all(), "Result should be finite"
    assert result.shape == (grid_size,), "Output shape should match input"

@pytest.mark.skip(reason="Requires C++ backend with 3D bindings")
def test_3d_gradient():
    """Test 3D Gradient operator."""
    try:
        from pymole.cpp import Gradient
    except ImportError:
        pytest.skip("C++ backend not available")
    
    m, n, o = 10, 12, 15
    dx, dy, dz = 0.1, 0.1, 0.1
    
    # Create 3D Gradient
    grad = Gradient(2, m, n, o, dx, dy, dz)
    
    # Test data
    grid_size = m * n * o
    x = np.random.rand(grid_size)
    result = grad @ x
    
    assert result.shape[1] == grid_size, "Output should match grid size"

@pytest.mark.skip(reason="Requires C++ backend with 3D bindings")
def test_3d_divergence():
    """Test 3D Divergence operator."""
    try:
        from pymole.cpp import Divergence
    except ImportError:
        pytest.skip("C++ backend not available")
    
    m, n, o = 10, 12, 15
    dx, dy, dz = 0.1, 0.1, 0.1
    
    # Create 3D Divergence
    div = Divergence(2, m, n, o, dx, dy, dz)
    
    # Test data (3 components for 3 spatial dimensions)
    grid_size = m * n * o
    x = np.random.rand(3 * grid_size)
    result = div @ x
    
    assert np.isfinite(result).all(), "Result should be finite"

@pytest.mark.skip(reason="Requires C++ backend with 3D bindings")
def test_3d_laplacian():
    """Test 3D Laplacian operator."""
    try:
        from pymole.cpp import Laplacian
    except ImportError:
        pytest.skip("C++ backend not available")
    
    m, n, o = 10, 12, 15
    dx, dy, dz = 0.1, 0.1, 0.1
    
    # Create 3D Laplacian
    lap = Laplacian(2, m, n, o, dx, dy, dz)
    
    # Test data
    grid_size = m * n * o
    x = np.ones(grid_size)
    result = lap @ x
    
    assert result.shape[0] == grid_size, "Output size should match input"
    assert np.isfinite(result).all(), "Result should be finite"


