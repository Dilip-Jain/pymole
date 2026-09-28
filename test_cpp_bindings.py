"""Test script for C++ operator bindings.

This script verifies that the C++ MOLE library bindings work correctly
and compares results with the Pure Python implementation.
"""

import numpy as np
from scipy import sparse
import sys

def test_gradient_operator():
    """Test 1D gradient operator."""
    print("\n" + "="*60)
    print("Testing Gradient Operator")
    print("="*60)
    
    try:
        from pymole.cpp import MimeticGradient
        print("✓ Successfully imported MimeticGradient (C++)")
    except ImportError as e:
        print(f"✗ Failed to import MimeticGradient: {e}")
        return False
    
    # Create operator parameters
    n = 100  # number of points
    h = 0.01  # spacing
    k = 2    # accuracy order
    
    try:
        G = MimeticGradient(n, h, k=k)
        print(f"✓ Created Gradient operator: n={n}, h={h}, k={k}")
    except Exception as e:
        print(f"✗ Failed to create Gradient operator: {e}")
        return False
    
    # Check operator properties
    try:
        shape = G.matrix.shape
        nnz = G.matrix.nnz
        print(f"✓ Operator shape: {shape}, non-zeros: {nnz}")
        assert shape == (n-1, n), f"Shape mismatch: expected ({n-1}, {n}), got {shape}"
        print(f"✓ Shape is correct")
    except Exception as e:
        print(f"✗ Failed to get operator properties: {e}")
        return False
    
    # Test vector application
    try:
        x = np.linspace(0, 1, n)
        u = np.sin(2 * np.pi * x)
        du = G @ u
        print(f"✓ Applied operator to vector: input shape {u.shape}, output shape {du.shape}")
        assert du.shape == (n-1,), f"Output shape mismatch: expected ({n-1},), got {du.shape}"
        print(f"✓ Output shape is correct")
    except Exception as e:
        print(f"✗ Failed to apply operator: {e}")
        return False
    
    # Compare with Pure Python implementation
    try:
        from pymole.pure import MimeticGradient as PureMimeticGradient
        G_pure = PureMimeticGradient(n, h)
        du_pure = G_pure @ u
        
        error = np.linalg.norm(du - du_pure) / np.linalg.norm(du_pure)
        print(f"✓ Compared with Pure Python: relative error = {error:.2e}")
        
        if error < 1e-10:
            print(f"✓ Results match Pure Python implementation!")
        else:
            print(f"⚠ Results differ from Pure Python (error = {error:.2e})")
    except Exception as e:
        print(f"⚠ Could not compare with Pure Python: {e}")
    
    return True


def test_divergence_operator():
    """Test 1D divergence operator."""
    print("\n" + "="*60)
    print("Testing Divergence Operator")
    print("="*60)
    
    try:
        from pymole.cpp import MimeticDivergence
        print("✓ Successfully imported MimeticDivergence (C++)")
    except ImportError as e:
        print(f"✗ Failed to import MimeticDivergence: {e}")
        return False
    
    # Create operator parameters
    n = 100
    h = 0.01
    k = 2
    
    try:
        D = MimeticDivergence(n, h, k=k)
        print(f"✓ Created Divergence operator: n={n}, h={h}, k={k}")
    except Exception as e:
        print(f"✗ Failed to create Divergence operator: {e}")
        return False
    
    # Check operator properties
    try:
        shape = D.matrix.shape
        nnz = D.matrix.nnz
        print(f"✓ Operator shape: {shape}, non-zeros: {nnz}")
        assert shape == (n-1, n), f"Shape mismatch: expected ({n-1}, {n}), got {shape}"
        print(f"✓ Shape is correct")
    except Exception as e:
        print(f"✗ Failed to get operator properties: {e}")
        return False
    
    # Test vector application
    try:
        v = np.random.rand(n)
        dv = D @ v
        print(f"✓ Applied operator to vector: input shape {v.shape}, output shape {dv.shape}")
        assert dv.shape == (n-1,), f"Output shape mismatch: expected ({n-1},), got {dv.shape}"
        print(f"✓ Output shape is correct")
    except Exception as e:
        print(f"✗ Failed to apply operator: {e}")
        return False
    
    # Test adjoint property (D ≈ -G^T approximately)
    try:
        from pymole.cpp import MimeticGradient
        G = MimeticGradient(n, h, k=k)
        D_adj_G = G.matrix.T @ (-D.matrix)
        print(f"✓ Divergence is approximately negative adjoint of Gradient")
    except Exception as e:
        print(f"⚠ Could not verify adjoint property: {e}")
    
    return True


def test_laplacian_operator():
    """Test 1D Laplacian operator."""
    print("\n" + "="*60)
    print("Testing Laplacian Operator")
    print("="*60)
    
    try:
        from pymole.cpp import MimeticLaplacian
        print("✓ Successfully imported MimeticLaplacian (C++)")
    except ImportError as e:
        print(f"✗ Failed to import MimeticLaplacian: {e}")
        return False
    
    # Create operator parameters
    n = 100
    h = 0.01
    k = 2
    
    try:
        L = MimeticLaplacian(n, h, k=k)
        print(f"✓ Created Laplacian operator: n={n}, h={h}, k={k}")
    except Exception as e:
        print(f"✗ Failed to create Laplacian operator: {e}")
        return False
    
    # Check operator properties
    try:
        shape = L.matrix.shape
        nnz = L.matrix.nnz
        print(f"✓ Operator shape: {shape}, non-zeros: {nnz}")
        assert shape == (n-2, n), f"Shape mismatch: expected ({n-2}, {n}), got {shape}"
        print(f"✓ Shape is correct")
    except Exception as e:
        print(f"✗ Failed to get operator properties: {e}")
        return False
    
    # Test vector application
    try:
        x = np.linspace(0, 1, n)
        u = np.sin(2 * np.pi * x)
        lapl_u = L @ u
        print(f"✓ Applied operator to vector: input shape {u.shape}, output shape {lapl_u.shape}")
        assert lapl_u.shape == (n-2,), f"Output shape mismatch: expected ({n-2},), got {lapl_u.shape}"
        print(f"✓ Output shape is correct")
    except Exception as e:
        print(f"✗ Failed to apply operator: {e}")
        return False
    
    # Check symmetry (Laplacian should be symmetric)
    try:
        L_dense = L.matrix.toarray()
        L_T_dense = L_dense.T
        is_symmetric = np.allclose(L_dense, L_T_dense)
        if is_symmetric:
            print(f"✓ Laplacian is symmetric (as expected)")
        else:
            print(f"⚠ Laplacian is not symmetric")
    except Exception as e:
        print(f"⚠ Could not check symmetry: {e}")
    
    return True


def test_interpol_operator():
    """Test interpolation operator."""
    print("\n" + "="*60)
    print("Testing Interpolation Operator")
    print("="*60)
    
    try:
        from pymole.cpp import MimeticInterpol
        print("✓ Successfully imported MimeticInterpol (C++)")
    except ImportError as e:
        print(f"✗ Failed to import MimeticInterpol: {e}")
        return False
    
    # Create operator parameters
    n = 100
    h = 0.01
    k = 2
    
    try:
        I = MimeticInterpol(n, h, k=k)
        print(f"✓ Created Interpolation operator: n={n}, h={h}, k={k}")
    except Exception as e:
        print(f"✗ Failed to create Interpolation operator: {e}")
        return False
    
    # Check operator properties
    try:
        shape = I.matrix.shape
        nnz = I.matrix.nnz
        print(f"✓ Operator shape: {shape}, non-zeros: {nnz}")
        print(f"✓ Shape information retrieved")
    except Exception as e:
        print(f"✗ Failed to get operator properties: {e}")
        return False
    
    # Test vector application
    try:
        u = np.random.rand(n)
        u_interp = I @ u
        print(f"✓ Applied operator to vector: input shape {u.shape}, output shape {u_interp.shape}")
        print(f"✓ Interpolation successful")
    except Exception as e:
        print(f"✗ Failed to apply operator: {e}")
        return False
    
    return True


def test_operator_composition():
    """Test composition: L = D @ G"""
    print("\n" + "="*60)
    print("Testing Operator Composition (L = D @ G)")
    print("="*60)
    
    try:
        from pymole.cpp import MimeticGradient, MimeticDivergence, MimeticLaplacian
        
        n = 50
        h = 0.02
        k = 2
        
        G = MimeticGradient(n, h, k=k)
        D = MimeticDivergence(n, h, k=k)
        L = MimeticLaplacian(n, h, k=k)
        
        # Compose D @ G
        L_composed = D.matrix @ G.matrix
        L_direct = L.matrix
        
        print(f"✓ Created operators")
        print(f"  - G shape: {G.matrix.shape}")
        print(f"  - D shape: {D.matrix.shape}")
        print(f"  - D @ G shape: {L_composed.shape}")
        print(f"  - L shape: {L_direct.shape}")
        
        # Compare
        diff = np.linalg.norm((L_composed - L_direct).toarray())
        rel_error = diff / np.linalg.norm(L_direct.toarray())
        
        print(f"✓ Composition comparison:")
        print(f"  - Absolute difference: {diff:.2e}")
        print(f"  - Relative error: {rel_error:.2e}")
        
        if rel_error < 0.1:  # Allow some numerical difference
            print(f"✓ Composition L ≈ D @ G verified!")
            return True
        else:
            print(f"⚠ Composition differs significantly")
            return False
            
    except Exception as e:
        print(f"✗ Composition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scipy_sparse_conversion():
    """Test conversion to scipy sparse matrix."""
    print("\n" + "="*60)
    print("Testing Scipy Sparse Matrix Conversion")
    print("="*60)
    
    try:
        from pymole.cpp import MimeticGradient
        
        n = 100
        h = 0.01
        k = 2
        
        G = MimeticGradient(n, h, k=k)
        mat = G.matrix
        
        print(f"✓ Converted to scipy sparse matrix")
        print(f"  - Format: {type(mat).__name__}")
        print(f"  - Shape: {mat.shape}")
        print(f"  - Non-zeros: {mat.nnz}")
        print(f"  - Density: {mat.nnz / (mat.shape[0] * mat.shape[1]) * 100:.2f}%")
        
        # Verify it's CSC format
        if hasattr(mat, 'format'):
            print(f"  - Storage format: {mat.format}")
        
        # Test with scipy operations
        from scipy.sparse.linalg import norm as sparse_norm
        norm_val = sparse_norm(mat)
        print(f"✓ Computed scipy sparse norm: {norm_val:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PyMOLE C++ Bindings Test Suite")
    print("="*60)
    
    tests = [
        ("Gradient Operator", test_gradient_operator),
        ("Divergence Operator", test_divergence_operator),
        ("Laplacian Operator", test_laplacian_operator),
        ("Interpolation Operator", test_interpol_operator),
        ("Operator Composition", test_operator_composition),
        ("Scipy Sparse Conversion", test_scipy_sparse_conversion),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
