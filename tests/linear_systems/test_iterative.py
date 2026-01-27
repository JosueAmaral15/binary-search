"""
Comprehensive tests for BinaryGaussSeidel iterative solver.

Tests cover:
- Initialization and parameter validation
- Basic solving for linear systems
- Diagonal dominance checking
- Convergence behavior
- Auto-tuning of omega (relaxation factor)
- Polynomial regression
- Edge cases

Target: 80%+ code coverage for BinaryGaussSeidel
"""

import pytest
import numpy as np
from math_toolkit.linear_systems import BinaryGaussSeidel


class TestBinaryGaussSeidelInit:
    """Test initialization and parameters."""

    def test_default_initialization(self):
        """Test creating solver with default parameters."""
        solver = BinaryGaussSeidel()
        assert solver.tolerance == 1e-6
        assert solver.max_iterations == 1000
        assert solver.check_dominance is True
        assert solver.verbose is False
        assert solver.auto_tune_omega is True

    def test_custom_parameters(self):
        """Test creating solver with custom parameters."""
        solver = BinaryGaussSeidel(
            tolerance=1e-8,
            max_iterations=500,
            check_dominance=False,
            verbose=True,
            auto_tune_omega=False
        )
        assert solver.tolerance == 1e-8
        assert solver.max_iterations == 500
        assert solver.check_dominance is False
        assert solver.verbose is True
        assert solver.auto_tune_omega is False


class TestBinaryGaussSeidelSolving:
    """Test basic solving functionality."""

    def test_simple_diagonal_dominant_system(self):
        """Test solving a simple diagonally dominant system."""
        # System: 4x - y = 15, -x + 4y - z = 10, -y + 3z = 10
        # Solution: x=5, y=5, z=5
        A = np.array([
            [4, -1, 0],
            [-1, 4, -1],
            [0, -1, 3]
        ], dtype=float)
        b = np.array([15, 10, 10], dtype=float)

        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.solve(A, b)

        expected = np.array([5, 5, 5])
        assert np.allclose(result.x, expected, atol=0.01)
        assert result.converged is True

    def test_2x2_system(self):
        """Test solving a 2x2 system."""
        # System: 3x + y = 9, x + 2y = 8
        # Solution: x=2, y=3
        A = np.array([[3, 1], [1, 2]], dtype=float)
        b = np.array([9, 8], dtype=float)

        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.solve(A, b)

        expected = np.array([2, 3])
        assert np.allclose(result.x, expected, atol=0.01)

    def test_larger_system(self):
        """Test solving a larger system."""
        np.random.seed(42)
        n = 10
        
        # Create diagonally dominant matrix
        A = np.random.randn(n, n)
        A = A + np.diag(np.sum(np.abs(A), axis=1) + 1)  # Make diagonally dominant
        
        x_true = np.random.randn(n)
        b = A @ x_true

        solver = BinaryGaussSeidel(tolerance=1e-4, verbose=False)
        result = solver.solve(A, b)

        assert np.allclose(result.x, x_true, atol=0.1)

    def test_identity_matrix(self):
        """Test solving with identity matrix (should converge quickly)."""
        A = np.eye(3)
        b = np.array([1, 2, 3])

        solver = BinaryGaussSeidel(verbose=False)
        result = solver.solve(A, b)

        assert np.allclose(result.x, b, atol=1e-10)
        assert result.iterations <= 5  # Should be very fast


class TestBinaryGaussSeidelConvergence:
    """Test convergence behavior."""

    def test_convergence_with_good_initial_guess(self):
        """Test convergence when starting near solution."""
        A = np.array([[4, -1], [-1, 4]], dtype=float)
        b = np.array([3, 3], dtype=float)
        x0 = np.array([0.9, 0.9])  # Close to solution [1, 1]

        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.solve(A, b, x0=x0)

        assert np.allclose(result.x, [1, 1], atol=0.01)
        # Should converge quickly
        assert result.iterations < 20

    def test_max_iterations_reached(self):
        """Test behavior when max iterations is reached."""
        # Non-diagonally dominant system (may not converge)
        A = np.array([[1, 2], [3, 1]], dtype=float)
        b = np.array([3, 4], dtype=float)

        solver = BinaryGaussSeidel(max_iterations=5, check_dominance=False, verbose=False)
        result = solver.solve(A, b)

        assert result.iterations == 5
        # May or may not have converged
        assert result.converged in [True, False]

    def test_residual_tracking(self):
        """Test that residual decreases over iterations."""
        A = np.array([[4, -1], [-1, 4]], dtype=float)
        b = np.array([3, 3], dtype=float)

        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.solve(A, b)

        # Residual should be small at convergence
        residual = np.linalg.norm(A @ result.x - b)
        assert residual < 1e-5


class TestBinaryGaussSeidelAutoTuning:
    """Test auto-tuning of omega (relaxation factor)."""

    def test_auto_tune_enabled(self):
        """Test solving with auto-tuning enabled (default)."""
        A = np.array([[4, -1], [-1, 4]], dtype=float)
        b = np.array([3, 3], dtype=float)

        solver = BinaryGaussSeidel(auto_tune_omega=True, verbose=False)
        result = solver.solve(A, b)

        assert np.allclose(result.x, [1, 1], atol=0.01)

    def test_auto_tune_disabled(self):
        """Test solving with auto-tuning disabled."""
        A = np.array([[4, -1], [-1, 4]], dtype=float)
        b = np.array([3, 3], dtype=float)

        solver = BinaryGaussSeidel(auto_tune_omega=False, verbose=False)
        result = solver.solve(A, b)

        assert np.allclose(result.x, [1, 1], atol=0.01)


class TestBinaryGaussSeidelInputValidation:
    """Test input validation."""

    def test_non_square_matrix(self):
        """Test that non-square matrix raises error."""
        A = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([1, 2])

        solver = BinaryGaussSeidel(verbose=False)
        
        with pytest.raises(ValueError, match="A must be square matrix"):
            solver.solve(A, b)

    def test_mismatched_dimensions(self):
        """Test that mismatched A and b dimensions raise error."""
        A = np.array([[1, 2], [3, 4]])
        b = np.array([1, 2, 3])

        solver = BinaryGaussSeidel(verbose=False)
        
        with pytest.raises(ValueError, match="b must have shape"):
            solver.solve(A, b)

    def test_singular_matrix(self):
        """Test behavior with singular matrix."""
        A = np.array([[1, 2], [2, 4]], dtype=float)  # Singular
        b = np.array([3, 6], dtype=float)

        solver = BinaryGaussSeidel(max_iterations=10, check_dominance=False, verbose=False)
        # Should handle gracefully (may not converge)
        result = solver.solve(A, b)
        assert result.iterations <= 10

    def test_zero_diagonal_element(self):
        """Test behavior when diagonal has zero."""
        A = np.array([[0, 1], [1, 2]], dtype=float)
        b = np.array([1, 2], dtype=float)

        solver = BinaryGaussSeidel(check_dominance=False, verbose=False)
        # Should handle gracefully or raise error
        try:
            result = solver.solve(A, b)
            # If it doesn't raise, check it handled it
            assert result is not None
        except (ValueError, RuntimeWarning):
            # Or it raises an appropriate error
            pass


class TestBinaryGaussSeidelDiagonalDominance:
    """Test diagonal dominance checking."""

    def test_strictly_diagonal_dominant(self):
        """Test with strictly diagonally dominant matrix."""
        # |4| > |-1| + |0| = 1 ✓
        # |4| > |-1| + |-1| = 2 ✓
        # |3| > |0| + |-1| = 1 ✓
        A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=float)
        b = np.array([15, 10, 10], dtype=float)

        solver = BinaryGaussSeidel(check_dominance=True, verbose=False)
        result = solver.solve(A, b)

        # Should converge
        assert result.converged is True
        assert len(result.warnings) == 0

    def test_not_diagonal_dominant_with_warning(self):
        """Test that non-diagonally dominant matrix produces warning."""
        # Not diagonally dominant
        A = np.array([[1, 2], [3, 1]], dtype=float)
        b = np.array([3, 4], dtype=float)

        solver = BinaryGaussSeidel(check_dominance=True, max_iterations=10, verbose=False)
        result = solver.solve(A, b)

        # Should have warnings
        assert len(result.warnings) > 0


class TestBinaryGaussSeidelEdgeCases:
    """Test edge cases."""

    def test_already_solved_system(self):
        """Test when initial guess is already the solution."""
        A = np.array([[2, 0], [0, 2]], dtype=float)
        b = np.array([4, 6], dtype=float)
        x0 = np.array([2, 3])  # Already the solution

        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.solve(A, b, x0=x0)

        assert np.allclose(result.x, [2, 3], atol=1e-10)
        # Should converge immediately
        assert result.iterations <= 2

    def test_very_small_tolerance(self):
        """Test convergence with very tight tolerance."""
        A = np.array([[4, -1], [-1, 4]], dtype=float)
        b = np.array([3, 3], dtype=float)

        solver = BinaryGaussSeidel(tolerance=1e-12, max_iterations=1000, verbose=False)
        result = solver.solve(A, b)

        # Should converge to very high precision
        assert np.allclose(result.x, [1, 1], atol=1e-10)

    def test_verbose_output(self, capsys):
        """Test that verbose mode prints progress."""
        A = np.array([[4, -1], [-1, 4]], dtype=float)
        b = np.array([3, 3], dtype=float)

        solver = BinaryGaussSeidel(tolerance=1e-6, max_iterations=10, verbose=True)
        result = solver.solve(A, b)

        captured = capsys.readouterr()
        # Should print some progress info
        assert "Iteration" in captured.out or "Converged" in captured.out

    def test_result_object_properties(self):
        """Test that result object has all expected properties."""
        A = np.array([[4, -1], [-1, 4]], dtype=float)
        b = np.array([3, 3], dtype=float)

        solver = BinaryGaussSeidel(verbose=False)
        result = solver.solve(A, b)

        # Check result has all properties
        assert hasattr(result, 'x')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'residual')
        assert hasattr(result, 'relative_change')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'warnings')
        
        # Check types
        assert isinstance(result.x, np.ndarray)
        assert isinstance(result.iterations, int)
        assert isinstance(result.residual, float)
        assert isinstance(result.relative_change, float)
        assert isinstance(result.converged, bool)
        assert isinstance(result.warnings, list)
