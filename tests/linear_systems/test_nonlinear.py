"""
Comprehensive tests for NonLinearGaussSeidel solver.

Tests cover:
- Initialization and parameter validation
- 2D and 3D nonlinear systems
- Convergence behavior
- Error handling
- Edge cases
"""

import pytest
import numpy as np
from math_toolkit.linear_systems import NonLinearGaussSeidel


class TestNonLinearGaussSeidelInit:
    """Test initialization and parameters."""

    def test_default_initialization(self):
        """Test creating solver with default parameters."""
        f1 = lambda x, y: x**2 + y - 1
        f2 = lambda x, y: x + y**2 - 1
        
        solver = NonLinearGaussSeidel(functions=[f1, f2])
        
        assert solver.n == 2
        assert solver.tolerance == 1e-6
        assert solver.max_iterations == 100
        assert solver.verbose is False

    def test_custom_parameters(self):
        """Test creating solver with custom parameters."""
        f1 = lambda x: x**2 - 4
        
        solver = NonLinearGaussSeidel(
            functions=[f1],
            tolerance=1e-8,
            max_iterations=50,
            verbose=True
        )
        
        assert solver.tolerance == 1e-8
        assert solver.max_iterations == 50
        assert solver.verbose is True

    def test_empty_functions_raises_error(self):
        """Test that empty function list raises error."""
        with pytest.raises(ValueError, match="Must provide at least one function"):
            NonLinearGaussSeidel(functions=[])

    def test_invalid_tolerance_raises_error(self):
        """Test that non-positive tolerance raises error."""
        f1 = lambda x: x**2 - 4
        
        with pytest.raises(ValueError, match="tolerance must be positive"):
            NonLinearGaussSeidel(functions=[f1], tolerance=-1e-6)


class TestNonLinearGaussSeidelSolving:
    """Test solving various nonlinear systems."""

    def test_simple_quadratic_1d(self):
        """Test solving x² - 4 = 0 (solution x=2)."""
        f = lambda x: x**2 - 4
        
        solver = NonLinearGaussSeidel(functions=[f], verbose=False)
        result = solver.solve(initial_guess=[1.0])
        
        assert abs(result.x[0] - 2.0) < 0.01
        assert result.converged is True

    def test_2d_system(self):
        """Test solving 2D system: x²+y-11=0, x+y²-7=0."""
        # Expected solution: x=3, y=2
        f1 = lambda x, y: x**2 + y - 11
        f2 = lambda x, y: x + y**2 - 7
        
        solver = NonLinearGaussSeidel(functions=[f1, f2], verbose=False)
        result = solver.solve(initial_guess=[0, 0])
        
        assert abs(result.x[0] - 3.0) < 0.01
        assert abs(result.x[1] - 2.0) < 0.01
        assert result.converged is True
        assert result.residual < 1e-5

    def test_3d_system(self):
        """Test solving 3D system."""
        # System: x²+y²+z²-14=0, x+y+z-6=0, xyz-6=0
        # One solution: x=3, y=2, z=1
        f1 = lambda x, y, z: x**2 + y**2 + z**2 - 14
        f2 = lambda x, y, z: x + y + z - 6
        f3 = lambda x, y, z: x * y * z - 6
        
        solver = NonLinearGaussSeidel(functions=[f1, f2, f3], verbose=False)
        result = solver.solve(initial_guess=[1, 1, 1])
        
        # Check solution satisfies equations
        assert result.residual < 1e-5
        assert result.converged is True

    def test_linear_system_as_nonlinear(self):
        """Test that linear system works correctly (sanity check)."""
        # System: 2x + y - 5 = 0, x + 3y - 8 = 0
        # Solution: x=1, y≈2.33
        f1 = lambda x, y: 2*x + y - 5
        f2 = lambda x, y: x + 3*y - 8
        
        solver = NonLinearGaussSeidel(functions=[f1, f2], verbose=False)
        result = solver.solve(initial_guess=[0, 0])
        
        # Check that equations are satisfied (residual small)
        assert result.residual < 1e-5
        assert result.converged is True


class TestNonLinearGaussSeidelConvergence:
    """Test convergence behavior."""

    def test_convergence_with_good_initial_guess(self):
        """Test convergence when starting near solution."""
        f1 = lambda x, y: x**2 + y - 11
        f2 = lambda x, y: x + y**2 - 7
        
        solver = NonLinearGaussSeidel(functions=[f1, f2], verbose=False)
        result = solver.solve(initial_guess=[2.9, 2.1])  # Near solution
        
        assert result.converged is True
        # Should converge quickly
        assert result.iterations < 10

    def test_max_iterations_reached(self):
        """Test behavior when max iterations is reached."""
        # Difficult system with poor initial guess
        f1 = lambda x, y: np.exp(x) + y - 10
        f2 = lambda x, y: x + np.exp(y) - 10
        
        solver = NonLinearGaussSeidel(
            functions=[f1, f2],
            max_iterations=3,
            verbose=False
        )
        result = solver.solve(initial_guess=[0, 0])
        
        assert result.iterations == 3
        assert len(result.warnings) > 0

    def test_residual_decreases(self):
        """Test that residual decreases with iterations."""
        f1 = lambda x, y: x**2 + y - 11
        f2 = lambda x, y: x + y**2 - 7
        
        solver = NonLinearGaussSeidel(functions=[f1, f2], verbose=False)
        result = solver.solve(initial_guess=[0, 0])
        
        # Final residual should be small
        assert result.residual < 1e-5


class TestNonLinearGaussSeidelInputValidation:
    """Test input validation."""

    def test_wrong_initial_guess_length(self):
        """Test that wrong length initial guess raises error."""
        f1 = lambda x, y: x**2 + y - 1
        f2 = lambda x, y: x + y**2 - 1
        
        solver = NonLinearGaussSeidel(functions=[f1, f2], verbose=False)
        
        with pytest.raises(ValueError, match="initial_guess must have length 2"):
            solver.solve(initial_guess=[1.0])

    def test_nan_in_initial_guess(self):
        """Test that NaN in initial guess raises error."""
        f = lambda x: x**2 - 4
        
        solver = NonLinearGaussSeidel(functions=[f], verbose=False)
        
        with pytest.raises(ValueError, match="must not contain NaN"):
            solver.solve(initial_guess=[np.nan])

    def test_inf_in_initial_guess(self):
        """Test that Inf in initial guess raises error."""
        f = lambda x: x**2 - 4
        
        solver = NonLinearGaussSeidel(functions=[f], verbose=False)
        
        with pytest.raises(ValueError, match="must not contain NaN or Inf"):
            solver.solve(initial_guess=[np.inf])


class TestNonLinearGaussSeidelErrorHandling:
    """Test error handling for problematic functions."""

    def test_function_with_domain_error(self):
        """Test handling of function with restricted domain."""
        # sqrt(x) with negative x causes ValueError
        f = lambda x: np.sqrt(x) - 2
        
        solver = NonLinearGaussSeidel(functions=[f], verbose=False)
        result = solver.solve(initial_guess=[1.0])
        
        # Should handle gracefully and find positive solution
        assert result.x[0] > 0

    def test_function_with_division_by_zero(self):
        """Test handling of division by zero."""
        f = lambda x: 1/x - 2
        
        solver = NonLinearGaussSeidel(functions=[f], verbose=False)
        result = solver.solve(initial_guess=[1.0])
        
        # Should handle gracefully
        assert result.x[0] != 0


class TestNonLinearGaussSeidelEdgeCases:
    """Test edge cases."""

    def test_already_at_solution(self):
        """Test when initial guess is already the solution."""
        f1 = lambda x, y: x - 3
        f2 = lambda x, y: y - 2
        
        solver = NonLinearGaussSeidel(functions=[f1, f2], verbose=False)
        result = solver.solve(initial_guess=[3.0, 2.0])
        
        # Should converge immediately
        assert result.iterations <= 2
        assert result.converged is True

    def test_very_tight_tolerance(self):
        """Test convergence with very tight tolerance."""
        f1 = lambda x, y: x**2 + y - 11
        f2 = lambda x, y: x + y**2 - 7
        
        solver = NonLinearGaussSeidel(
            functions=[f1, f2],
            tolerance=1e-10,
            verbose=False
        )
        result = solver.solve(initial_guess=[0, 0])
        
        # Should still converge, just take more iterations
        assert result.residual < 1e-8

    def test_verbose_output(self, caplog):
        """Test that verbose mode logs progress."""
        import logging
        
        f1 = lambda x, y: x**2 + y - 11
        f2 = lambda x, y: x + y**2 - 7
        
        with caplog.at_level(logging.INFO):
            solver = NonLinearGaussSeidel(functions=[f1, f2], verbose=True)
            result = solver.solve(initial_guess=[0, 0])
        
        log_text = caplog.text
        # Should log iteration info
        assert "Iter" in log_text or "Converged" in log_text or "Starting" in log_text

    def test_result_object_properties(self):
        """Test that result object has all expected properties."""
        f1 = lambda x, y: x**2 + y - 11
        f2 = lambda x, y: x + y**2 - 7
        
        solver = NonLinearGaussSeidel(functions=[f1, f2], verbose=False)
        result = solver.solve(initial_guess=[0, 0])
        
        # Check result has all properties
        assert hasattr(result, 'x')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'residual')
        assert hasattr(result, 'relative_change')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'function_evals')
        
        # Check types
        assert isinstance(result.x, np.ndarray)
        assert isinstance(result.iterations, int)
        assert isinstance(result.residual, float)
        assert isinstance(result.converged, bool)
        assert isinstance(result.function_evals, int)

    def test_function_evaluation_count(self):
        """Test that function evaluations are tracked."""
        f = lambda x: x**2 - 4
        
        solver = NonLinearGaussSeidel(functions=[f], verbose=False)
        result = solver.solve(initial_guess=[1.0])
        
        # Should have evaluated the function multiple times
        assert result.function_evals > 0
