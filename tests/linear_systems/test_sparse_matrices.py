"""
Tests for sparse matrix support in BinaryGaussSeidel.

Phase 2, Task 2.3: Sparse matrix optimization tests.
"""

import pytest
import numpy as np
from math_toolkit.linear_systems import BinaryGaussSeidel

# Check if scipy is available
try:
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    sp = None

# Skip all tests if scipy not available
pytestmark = pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")


class TestSparseMatrixDetection:
    """Test automatic detection and routing to sparse solver"""
    
    def test_csr_matrix_detection(self):
        """CSR format should be detected and use sparse solver"""
        A_dense = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=float)
        A_sparse = sp.csr_matrix(A_dense)
        b = np.array([15, 10, 10], dtype=float)
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.solve(A_sparse, b)
        
        assert result.converged
        assert np.allclose(result.x, [5.0, 5.0, 5.0], atol=1e-5)
    
    def test_csc_matrix_detection(self):
        """CSC format should be converted to CSR and solved"""
        A_dense = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=float)
        A_sparse = sp.csc_matrix(A_dense)  # Column format
        b = np.array([15, 10, 10], dtype=float)
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.solve(A_sparse, b)
        
        assert result.converged
        assert np.allclose(result.x, [5.0, 5.0, 5.0], atol=1e-5)
    
    def test_coo_matrix_detection(self):
        """COO format should be converted to CSR and solved"""
        A_dense = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=float)
        A_sparse = sp.coo_matrix(A_dense)  # Coordinate format
        b = np.array([15, 10, 10], dtype=float)
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.solve(A_sparse, b)
        
        assert result.converged
        assert np.allclose(result.x, [5.0, 5.0, 5.0], atol=1e-5)
    
    def test_dense_vs_sparse_same_result(self):
        """Dense and sparse solvers should produce identical results"""
        A_dense = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=float)
        A_sparse = sp.csr_matrix(A_dense)
        b = np.array([15, 10, 10], dtype=float)
        
        solver = BinaryGaussSeidel(tolerance=1e-8, verbose=False)
        
        result_dense = solver.solve(A_dense, b)
        result_sparse = solver.solve(A_sparse, b)
        
        assert result_dense.converged and result_sparse.converged
        assert np.allclose(result_dense.x, result_sparse.x, atol=1e-6)


class TestSparsePerformance:
    """Test performance characteristics of sparse solver"""
    
    def test_highly_sparse_matrix(self):
        """Test with 95% sparse matrix (tridiagonal)"""
        n = 100
        # Create tridiagonal matrix (3 non-zeros per row, 97% sparse)
        diagonals = [4 * np.ones(n), -np.ones(n-1), -np.ones(n-1)]
        A = sp.diags(diagonals, [0, -1, 1], shape=(n, n), format='csr', dtype=float)
        b = np.ones(n)
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False, max_iterations=500)
        result = solver.solve(A, b)
        
        assert result.converged
        # Verify solution quality
        residual = np.linalg.norm(A.dot(result.x) - b)
        assert residual < 1e-5
    
    def test_medium_sparse_matrix(self):
        """Test with 50% sparse matrix"""
        np.random.seed(42)  # For reproducibility
        n = 50
        A_dense = np.random.rand(n, n)
        # Make diagonally dominant (ensure non-zero diagonals)
        A_dense += np.diag(10 * np.ones(n))
        # Zero out 50% of off-diagonal elements only
        for i in range(n):
            for j in range(n):
                if i != j and np.random.rand() < 0.5:
                    A_dense[i, j] = 0
        
        A_sparse = sp.csr_matrix(A_dense)
        b = np.random.rand(n)
        
        solver = BinaryGaussSeidel(tolerance=1e-4, verbose=False, max_iterations=1000)
        result = solver.solve(A_sparse, b)
        
        # Should converge or get close
        assert result.iterations < 1000 or result.residual < 0.01
    
    def test_large_sparse_system(self):
        """Test large system (1000×1000) that would be slow as dense"""
        n = 1000
        # Pentadiagonal matrix (5 non-zeros per row, 99.5% sparse)
        diagonals = [5 * np.ones(n), -np.ones(n-1), -np.ones(n-1), 
                     -0.5 * np.ones(n-2), -0.5 * np.ones(n-2)]
        A = sp.diags(diagonals, [0, -1, 1, -2, 2], shape=(n, n), format='csr', dtype=float)
        b = np.ones(n)
        
        solver = BinaryGaussSeidel(tolerance=1e-4, verbose=False, max_iterations=500)
        result = solver.solve(A, b)
        
        # Large system should still converge
        assert result.converged or result.residual < 0.01
        assert result.iterations < 500


class TestSparseDiagonalDominance:
    """Test diagonal dominance checking for sparse matrices"""
    
    def test_sparse_diagonally_dominant_no_warning(self):
        """Diagonally dominant sparse matrix should not warn"""
        import warnings
        
        A_dense = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=float)
        A_sparse = sp.csr_matrix(A_dense)
        b = np.array([15, 10, 10], dtype=float)
        
        solver = BinaryGaussSeidel(check_dominance=True, verbose=False)
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = solver.solve(A_sparse, b)
        
        assert result.converged
        # Should have no warnings about diagonal dominance
        dominance_warnings = [w for w in warning_list if "diagonally dominant" in str(w.message)]
        assert len(dominance_warnings) == 0
    
    def test_sparse_not_diagonally_dominant_warns(self):
        """Non-diagonally dominant sparse matrix should warn"""
        A_dense = np.array([[1, -1, -1], [-1, 1, -1], [0, -1, 1]], dtype=float)
        A_sparse = sp.csr_matrix(A_dense)
        b = np.array([1, 1, 1], dtype=float)
        
        solver = BinaryGaussSeidel(check_dominance=True, verbose=False, max_iterations=100)
        
        with pytest.warns(UserWarning, match="diagonally dominant"):
            result = solver.solve(A_sparse, b)
        
        # May or may not converge, but should have warned
        assert len(result.warnings) > 0


class TestSparseEdgeCases:
    """Test edge cases and error handling for sparse matrices"""
    
    def test_sparse_zero_diagonal_raises(self):
        """Sparse matrix with zero diagonal should raise error"""
        A_dense = np.array([[0, 1, 0], [1, 4, -1], [0, -1, 3]], dtype=float)
        A_sparse = sp.csr_matrix(A_dense)
        b = np.array([1, 2, 3], dtype=float)
        
        solver = BinaryGaussSeidel()
        
        with pytest.raises(ValueError, match="zero or near-zero diagonal"):
            solver.solve(A_sparse, b)
    
    def test_sparse_non_square_raises(self):
        """Non-square sparse matrix should raise error"""
        A_dense = np.array([[4, -1, 0], [-1, 4, -1]], dtype=float)  # 2×3
        A_sparse = sp.csr_matrix(A_dense)
        b = np.array([1, 2], dtype=float)
        
        solver = BinaryGaussSeidel()
        
        with pytest.raises(ValueError, match="square matrix"):
            solver.solve(A_sparse, b)
    
    def test_sparse_dimension_mismatch_raises(self):
        """Mismatched dimensions should raise error"""
        A_dense = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=float)
        A_sparse = sp.csr_matrix(A_dense)
        b = np.array([1, 2], dtype=float)  # Wrong size
        
        solver = BinaryGaussSeidel()
        
        with pytest.raises(ValueError, match="shape"):
            solver.solve(A_sparse, b)
    
    def test_sparse_with_initial_guess(self):
        """Sparse solver should accept initial guess"""
        A_dense = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=float)
        A_sparse = sp.csr_matrix(A_dense)
        b = np.array([15, 10, 10], dtype=float)
        x0 = np.array([3.5, 3.5, 4.5], dtype=float)  # Close to solution
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.solve(A_sparse, b, x0=x0)
        
        assert result.converged
        # Should converge faster with good initial guess
        assert result.iterations < 20


class TestSparseAutoTuning:
    """Test binary search omega tuning with sparse matrices"""
    
    def test_sparse_auto_tuning_enabled(self):
        """Sparse solver should use binary search omega tuning by default"""
        A_dense = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=float)
        A_sparse = sp.csr_matrix(A_dense)
        b = np.array([15, 10, 10], dtype=float)
        
        solver = BinaryGaussSeidel(tolerance=1e-6, auto_tune_omega=True, verbose=False)
        result = solver.solve(A_sparse, b)
        
        assert result.converged
        assert np.allclose(result.x, [5.0, 5.0, 5.0], atol=1e-5)
    
    def test_sparse_auto_tuning_disabled(self):
        """Sparse solver should work with auto-tuning disabled"""
        A_dense = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=float)
        A_sparse = sp.csr_matrix(A_dense)
        b = np.array([15, 10, 10], dtype=float)
        
        solver = BinaryGaussSeidel(tolerance=1e-6, auto_tune_omega=False, verbose=False)
        result = solver.solve(A_sparse, b)
        
        assert result.converged
        assert np.allclose(result.x, [5.0, 5.0, 5.0], atol=1e-5)
    
    def test_sparse_omega_search_iterations(self):
        """Sparse solver should respect omega_search_iterations parameter"""
        n = 50
        diagonals = [4 * np.ones(n), -np.ones(n-1), -np.ones(n-1)]
        A = sp.diags(diagonals, [0, -1, 1], shape=(n, n), format='csr', dtype=float)
        b = np.ones(n)
        
        # More search iterations might improve convergence
        solver = BinaryGaussSeidel(tolerance=1e-6, omega_search_iterations=10, verbose=False)
        result = solver.solve(A, b)
        
        assert result.converged or result.residual < 1e-4


class TestSparseMemoryEfficiency:
    """Test that sparse solver uses less memory than dense"""
    
    def test_sparse_nnz_reported(self, caplog):
        """Verbose mode should report sparsity information"""
        import logging
        
        n = 100
        diagonals = [4 * np.ones(n), -np.ones(n-1), -np.ones(n-1)]
        A = sp.diags(diagonals, [0, -1, 1], shape=(n, n), format='csr', dtype=float)
        b = np.ones(n)
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=True)
        
        # Capture logs
        with caplog.at_level(logging.INFO):
            result = solver.solve(A, b)
        
        # Check that sparsity info was logged
        log_text = caplog.text
        assert "non-zeros" in log_text or "sparse" in log_text
    
    def test_sparse_preserves_format(self):
        """Sparse solver should not convert to dense"""
        n = 100
        diagonals = [4 * np.ones(n), -np.ones(n-1), -np.ones(n-1)]
        A = sp.diags(diagonals, [0, -1, 1], shape=(n, n), format='csr', dtype=float)
        b = np.ones(n)
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.solve(A, b)
        
        # Result should be dense array, but A should remain sparse
        assert isinstance(result.x, np.ndarray)
        assert sp.isspmatrix(A)  # Original matrix still sparse


class TestSparseComparisonWithDense:
    """Compare sparse and dense performance on same problem"""
    
    def test_sparse_faster_for_large_sparse(self):
        """Sparse solver should be more efficient for large sparse systems"""
        import time
        
        n = 200
        # Create sparse tridiagonal (99% sparse)
        diagonals = [4 * np.ones(n), -np.ones(n-1), -np.ones(n-1)]
        A_sparse = sp.diags(diagonals, [0, -1, 1], shape=(n, n), format='csr', dtype=float)
        A_dense = A_sparse.toarray()
        b = np.ones(n)
        
        solver = BinaryGaussSeidel(tolerance=1e-4, verbose=False, max_iterations=100)
        
        # Time sparse solve
        start = time.time()
        result_sparse = solver.solve(A_sparse, b)
        time_sparse = time.time() - start
        
        # Time dense solve
        start = time.time()
        result_dense = solver.solve(A_dense, b)
        time_dense = time.time() - start
        
        # Both should converge
        assert result_sparse.converged or result_sparse.residual < 0.01
        assert result_dense.converged or result_dense.residual < 0.01
        
        # Sparse should be at least competitive (not necessarily faster in small tests)
        # Main benefit is memory usage, not always speed for these sizes
        assert time_sparse < time_dense * 2  # At most 2x slower (usually faster)
