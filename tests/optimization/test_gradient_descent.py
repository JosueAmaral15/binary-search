"""
Comprehensive tests for Binary Rate Optimizer.

Tests cover:
- Basic optimization functionality
- Input validation
- Edge cases
- Convergence behavior
- History tracking

Target: 90%+ code coverage
"""

import pytest
import numpy as np
from math_toolkit.optimization import BinaryRateOptimizer


class TestBinaryRateOptimizerInit:
    """Test initialization and parameter validation."""

    def test_default_initialization(self):
        """Test creating optimizer with default parameters."""
        optimizer = BinaryRateOptimizer()
        assert optimizer.max_iter == 100
        assert optimizer.tol == 1e-6
        assert optimizer.expansion_factor == 2.0
        assert optimizer.binary_search_steps == 10
        assert optimizer.history == {"theta": [], "cost": [], "alpha": []}

    def test_custom_parameters(self):
        """Test creating optimizer with custom parameters."""
        optimizer = BinaryRateOptimizer(
            max_iter=50, tol=1e-8, expansion_factor=3.0, binary_search_steps=15
        )
        assert optimizer.max_iter == 50
        assert optimizer.tol == 1e-8
        assert optimizer.expansion_factor == 3.0
        assert optimizer.binary_search_steps == 15

    def test_invalid_max_iter(self):
        """Test that invalid max_iter raises ValueError."""
        with pytest.raises(ValueError, match="max_iter must be positive"):
            BinaryRateOptimizer(max_iter=0)
        with pytest.raises(ValueError, match="max_iter must be positive"):
            BinaryRateOptimizer(max_iter=-5)

    def test_invalid_tolerance(self):
        """Test that invalid tolerance raises ValueError."""
        with pytest.raises(ValueError, match="tol must be positive"):
            BinaryRateOptimizer(tol=0)
        with pytest.raises(ValueError, match="tol must be positive"):
            BinaryRateOptimizer(tol=-1e-6)

    def test_invalid_expansion_factor(self):
        """Test that invalid expansion_factor raises ValueError."""
        with pytest.raises(ValueError, match="expansion_factor must be > 1.0"):
            BinaryRateOptimizer(expansion_factor=1.0)
        with pytest.raises(ValueError, match="expansion_factor must be > 1.0"):
            BinaryRateOptimizer(expansion_factor=0.5)

    def test_invalid_binary_search_steps(self):
        """Test that invalid binary_search_steps raises ValueError."""
        with pytest.raises(ValueError, match="binary_search_steps must be positive"):
            BinaryRateOptimizer(binary_search_steps=0)
        with pytest.raises(ValueError, match="binary_search_steps must be positive"):
            BinaryRateOptimizer(binary_search_steps=-10)


class TestBinaryRateOptimizerOptimization:
    """Test main optimization functionality."""

    def test_simple_linear_regression(self):
        """Test optimization on simple linear problem: y = 2x."""

        def mse_cost(theta, X, y):
            predictions = X * theta
            return np.mean((predictions - y) ** 2)

        def mse_gradient(theta, X, y):
            predictions = X * theta
            error = predictions - y
            return 2 * np.mean(error * X)

        X = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        initial_theta = np.array([0.0])

        optimizer = BinaryRateOptimizer(max_iter=50, tol=1e-9, verbose=False)
        final_theta = optimizer.optimize(X, y, initial_theta, mse_cost, mse_gradient)

        # Should find theta ≈ 2.0
        assert abs(final_theta[0] - 2.0) < 0.01

    def test_convergence_with_good_initial_guess(self):
        """Test that optimization converges quickly with good initial guess."""

        def cost(theta, X, y):
            return np.mean((X * theta - y) ** 2)

        def grad(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)

        X = np.array([1, 2, 3, 4])
        y = np.array([3, 6, 9, 12])
        initial_theta = np.array([2.9])  # Close to optimal (3.0)

        optimizer = BinaryRateOptimizer(max_iter=10, tol=1e-6)
        final_theta = optimizer.optimize(X, y, initial_theta, cost, grad)

        assert abs(final_theta[0] - 3.0) < 0.01
        # Should converge in few iterations
        assert len(optimizer.history["cost"]) < 10

    def test_convergence_with_bad_initial_guess(self):
        """Test that optimization still works with bad initial guess."""

        def cost(theta, X, y):
            return np.mean((X * theta - y) ** 2)

        def grad(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)

        X = np.array([1, 2, 3, 4])
        y = np.array([3, 6, 9, 12])
        initial_theta = np.array([-100.0])  # Very bad initial guess

        optimizer = BinaryRateOptimizer(max_iter=100, tol=1e-6, verbose=False)
        final_theta = optimizer.optimize(X, y, initial_theta, cost, grad)

        assert abs(final_theta[0] - 3.0) < 0.1

    def test_history_tracking(self):
        """Test that history is properly tracked."""

        def cost(theta, X, y):
            return np.mean((X * theta - y) ** 2)

        def grad(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)

        X = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        initial_theta = np.array([0.0])

        optimizer = BinaryRateOptimizer(max_iter=20, tol=1e-9)
        optimizer.optimize(X, y, initial_theta, cost, grad)

        # Check history structure
        assert len(optimizer.history["theta"]) > 0
        assert len(optimizer.history["cost"]) > 0
        assert len(optimizer.history["alpha"]) > 0

        # All should have same length
        assert len(optimizer.history["theta"]) == len(optimizer.history["cost"])
        assert len(optimizer.history["cost"]) == len(optimizer.history["alpha"])

        # Cost should be decreasing (or equal due to convergence)
        costs = optimizer.history["cost"]
        for i in range(len(costs) - 1):
            assert costs[i] >= costs[i + 1]

    def test_zero_gradient_convergence(self):
        """Test early stopping when gradient becomes zero."""

        def cost(theta, X, y):
            # Perfect fit already
            return 0.0

        def grad(theta, X, y):
            # Gradient is zero
            return np.array([0.0])

        X = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        initial_theta = np.array([1.0])

        optimizer = BinaryRateOptimizer(max_iter=100, tol=1e-6, verbose=False)
        optimizer.optimize(X, y, initial_theta, cost, grad)

        # Should stop immediately due to zero gradient
        assert len(optimizer.history["cost"]) == 1

    def test_multiple_optimizations_clear_history(self):
        """Test that history is cleared between optimizations."""

        def cost(theta, X, y):
            return np.mean((X * theta - y) ** 2)

        def grad(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)

        X = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        initial_theta = np.array([0.0])

        optimizer = BinaryRateOptimizer(max_iter=10, tol=1e-6)

        # First optimization
        optimizer.optimize(X, y, initial_theta, cost, grad)
        first_history_len = len(optimizer.history["cost"])

        # Second optimization
        optimizer.optimize(X, y, initial_theta, cost, grad)
        second_history_len = len(optimizer.history["cost"])

        # History should be from second run only
        assert second_history_len > 0
        # Should be similar length (not cumulative)
        assert abs(second_history_len - first_history_len) < 5


class TestBinaryRateOptimizerInputValidation:
    """Test input validation in optimize method."""

    def test_non_numpy_arrays(self):
        """Test that non-numpy arrays raise ValueError."""

        def cost(theta, X, y):
            return 0

        def grad(theta, X, y):
            return np.array([0.0])

        optimizer = BinaryRateOptimizer()

        with pytest.raises(ValueError, match="X and y must be numpy arrays"):
            optimizer.optimize([1, 2, 3], np.array([1, 2, 3]), np.array([0.0]), cost, grad)

        with pytest.raises(ValueError, match="X and y must be numpy arrays"):
            optimizer.optimize(np.array([1, 2, 3]), [1, 2, 3], np.array([0.0]), cost, grad)

    def test_non_numpy_initial_theta(self):
        """Test that non-numpy initial_theta raises ValueError."""

        def cost(theta, X, y):
            return 0

        def grad(theta, X, y):
            return np.array([0.0])

        optimizer = BinaryRateOptimizer()

        with pytest.raises(ValueError, match="initial_theta must be a numpy array"):
            optimizer.optimize(np.array([1, 2, 3]), np.array([1, 2, 3]), [0.0], cost, grad)

    def test_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise ValueError."""

        def cost(theta, X, y):
            return 0

        def grad(theta, X, y):
            return np.array([0.0])

        optimizer = BinaryRateOptimizer()

        with pytest.raises(ValueError, match="X and y must have same length"):
            optimizer.optimize(
                np.array([1, 2, 3]), np.array([1, 2]), np.array([0.0]), cost, grad
            )

    def test_nan_in_data(self):
        """Test that NaN values raise ValueError."""

        def cost(theta, X, y):
            return 0

        def grad(theta, X, y):
            return np.array([0.0])

        optimizer = BinaryRateOptimizer()

        with pytest.raises(ValueError, match="X and y must not contain NaN or Inf"):
            optimizer.optimize(
                np.array([1, np.nan, 3]),
                np.array([1, 2, 3]),
                np.array([0.0]),
                cost,
                grad,
            )

    def test_inf_in_data(self):
        """Test that Inf values raise ValueError."""

        def cost(theta, X, y):
            return 0

        def grad(theta, X, y):
            return np.array([0.0])

        optimizer = BinaryRateOptimizer()

        with pytest.raises(ValueError, match="X and y must not contain NaN or Inf"):
            optimizer.optimize(
                np.array([1, 2, 3]),
                np.array([1, np.inf, 3]),
                np.array([0.0]),
                cost,
                grad,
            )

    def test_nan_in_initial_theta(self):
        """Test that NaN in initial_theta raises ValueError."""

        def cost(theta, X, y):
            return 0

        def grad(theta, X, y):
            return np.array([0.0])

        optimizer = BinaryRateOptimizer()

        with pytest.raises(ValueError, match="initial_theta must not contain NaN or Inf"):
            optimizer.optimize(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                np.array([np.nan]),
                cost,
                grad,
            )


class TestBinaryRateOptimizerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_data_point(self):
        """Test optimization with single data point."""

        def cost(theta, X, y):
            return (X * theta - y) ** 2

        def grad(theta, X, y):
            return 2 * (X * theta - y) * X

        X = np.array([2.0])
        y = np.array([4.0])
        initial_theta = np.array([0.0])

        optimizer = BinaryRateOptimizer(max_iter=50, tol=1e-9, verbose=False)
        final_theta = optimizer.optimize(X, y, initial_theta, cost, grad)

        assert abs(final_theta[0] - 2.0) < 0.01

    def test_already_converged(self):
        """Test when initial guess is already optimal."""

        def cost(theta, X, y):
            return np.mean((X * theta - y) ** 2)

        def grad(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)

        X = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        initial_theta = np.array([2.0])  # Already optimal

        optimizer = BinaryRateOptimizer(max_iter=50, tol=1e-9, verbose=False)
        final_theta = optimizer.optimize(X, y, initial_theta, cost, grad)

        assert abs(final_theta[0] - 2.0) < 0.01
        # Should converge very quickly
        assert len(optimizer.history["cost"]) <= 3

    def test_verbose_output(self, capsys):
        """Test that verbose mode prints progress."""

        def cost(theta, X, y):
            return np.mean((X * theta - y) ** 2)

        def grad(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)

        X = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        initial_theta = np.array([0.0])

        optimizer = BinaryRateOptimizer(max_iter=5, tol=1e-9, verbose=True)
        optimizer.optimize(X, y, initial_theta, cost, grad)

        captured = capsys.readouterr()
        assert "Starting BR-GD Optimization" in captured.out
        assert "Initial Cost" in captured.out
        assert "Iter" in captured.out

    def test_tolerance_convergence(self):
        """Test convergence based on tolerance."""

        def cost(theta, X, y):
            return np.mean((X * theta - y) ** 2)

        def grad(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)

        X = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        initial_theta = np.array([1.5])  # Close to optimal (2.0)

        # Tight tolerance should require more iterations
        optimizer_tight = BinaryRateOptimizer(max_iter=100, tol=1e-12, verbose=False)
        optimizer_tight.optimize(X, y, initial_theta, cost, grad)

        # Loose tolerance should require fewer iterations
        optimizer_loose = BinaryRateOptimizer(max_iter=100, tol=1e-2)
        optimizer_loose.optimize(X, y, initial_theta, cost, grad)

        # Loose tolerance should converge faster
        assert len(optimizer_loose.history["cost"]) <= len(optimizer_tight.history["cost"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=binary_rate_optimizer", "--cov-report=term-missing"])
