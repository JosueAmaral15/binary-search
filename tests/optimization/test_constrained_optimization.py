"""
Comprehensive tests for constrained optimization (box constraints).

Tests cover both BinaryRateOptimizer and AdamW with box constraints.
"""

import pytest
import numpy as np
from math_toolkit.optimization import BinaryRateOptimizer, AdamW


class TestBinaryRateOptimizerConstraints:
    """Test BinaryRateOptimizer with box constraints."""

    def test_init_with_bounds(self):
        """Test initialization with bounds."""
        bounds = (0, 1)
        optimizer = BinaryRateOptimizer(bounds=bounds, verbose=False)
        assert optimizer.bounds == bounds

    def test_init_invalid_bounds(self):
        """Test that invalid bounds raise errors."""
        # Lower >= upper
        with pytest.raises(ValueError, match="lower bounds must be < upper"):
            BinaryRateOptimizer(bounds=(1, 0), verbose=False)
        
        # Not a tuple
        with pytest.raises(ValueError, match="bounds must be a tuple"):
            BinaryRateOptimizer(bounds=[0, 1], verbose=False)
        
        # Wrong length
        with pytest.raises(ValueError, match="bounds must be a tuple"):
            BinaryRateOptimizer(bounds=(0, 1, 2), verbose=False)

    def test_unconstrained_same_as_no_bounds(self):
        """Test that bounds=None behaves same as no bounds."""
        def cost(theta, X, y):
            return np.mean((X * theta - y) ** 2)
        
        def grad(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)
        
        X = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        initial = np.array([0.0])
        
        # Without bounds
        opt1 = BinaryRateOptimizer(max_iter=50, verbose=False)
        theta1 = opt1.optimize(X, y, initial.copy(), cost, grad)
        
        # With bounds=None
        opt2 = BinaryRateOptimizer(max_iter=50, verbose=False, bounds=None)
        theta2 = opt2.optimize(X, y, initial.copy(), cost, grad)
        
        assert np.allclose(theta1, theta2)

    def test_constrained_to_positive(self):
        """Test constraining parameters to positive values."""
        def cost(theta, X, y):
            return np.mean((X * theta - y) ** 2)
        
        def grad(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)
        
        # Data that would give negative theta without constraints
        X = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([-2, -4, -6, -8], dtype=float)  # Negative pattern
        
        initial = np.array([1.0])  # Start positive
        bounds = (0, np.inf)  # Only positive values
        
        optimizer = BinaryRateOptimizer(max_iter=100, verbose=False, bounds=bounds)
        theta = optimizer.optimize(X, y, initial, cost, grad)
        
        # Should be clamped at 0 (can't go negative)
        assert theta[0] >= 0

    def test_constrained_unit_interval(self):
        """Test constraining parameters to [0, 1]."""
        def cost(theta, X, y):
            return np.mean((X @ theta - y) ** 2)
        
        def grad(theta, X, y):
            return 2 * X.T @ (X @ theta - y) / len(y)
        
        X = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        y = np.array([0.5, 0.8, 1.3], dtype=float)
        initial = np.array([0.5, 0.5])
        bounds = (0, 1)  # Unit interval for both params
        
        optimizer = BinaryRateOptimizer(max_iter=100, verbose=False, bounds=bounds)
        theta = optimizer.optimize(X, y, initial, cost, grad)
        
        # All parameters should be in [0, 1]
        assert np.all(theta >= 0)
        assert np.all(theta <= 1)

    def test_per_parameter_bounds(self):
        """Test different bounds for each parameter."""
        def cost(theta, X, y):
            return np.mean((X @ theta - y) ** 2)
        
        def grad(theta, X, y):
            return 2 * X.T @ (X @ theta - y) / len(y)
        
        X = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        y = np.array([2, 10, 12], dtype=float)
        initial = np.array([1.0, 5.0])
        
        # First param: [0, 5], Second param: [0, 15]
        bounds = (np.array([0, 0]), np.array([5, 15]))
        
        optimizer = BinaryRateOptimizer(max_iter=100, verbose=False, bounds=bounds)
        theta = optimizer.optimize(X, y, initial, cost, grad)
        
        # Check bounds respected
        assert 0 <= theta[0] <= 5
        assert 0 <= theta[1] <= 15

    def test_initial_guess_outside_bounds(self):
        """Test that initial guess outside bounds is projected."""
        def cost(theta, X, y):
            return np.mean((X * theta - y) ** 2)
        
        def grad(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)
        
        X = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        
        # Initial guess way outside bounds
        initial = np.array([10.0])
        bounds = (0, 1)
        
        optimizer = BinaryRateOptimizer(max_iter=50, verbose=False, bounds=bounds)
        theta = optimizer.optimize(X, y, initial, cost, grad)
        
        # Final should be in bounds
        assert 0 <= theta[0] <= 1


class TestAdamWConstraints:
    """Test AdamW with box constraints."""

    def test_init_with_bounds(self):
        """Test initialization with bounds."""
        bounds = (0, 1)
        optimizer = AdamW(bounds=bounds, verbose=False)
        assert optimizer.bounds == bounds

    def test_init_invalid_bounds(self):
        """Test that invalid bounds raise errors."""
        # Lower >= upper
        with pytest.raises(ValueError, match="lower bounds must be < upper"):
            AdamW(bounds=(1, 0), verbose=False)

    def test_unconstrained_same_as_no_bounds(self):
        """Test that bounds=None behaves same as no bounds."""
        def cost(theta, X, y):
            return np.mean((X * theta - y) ** 2)
        
        def grad(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)
        
        X = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        initial = np.array([0.0])
        
        # Without bounds
        opt1 = AdamW(max_iter=50, use_binary_search=False, verbose=False)
        theta1 = opt1.optimize(X, y, initial.copy(), cost, grad)
        
        # With bounds=None
        opt2 = AdamW(max_iter=50, use_binary_search=False, verbose=False, bounds=None)
        theta2 = opt2.optimize(X, y, initial.copy(), cost, grad)
        
        assert np.allclose(theta1, theta2)

    def test_constrained_to_positive(self):
        """Test constraining parameters to positive values with AdamW."""
        def cost(theta, X, y):
            return np.mean((X * theta - y) ** 2)
        
        def grad(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)
        
        X = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([-2, -4, -6, -8], dtype=float)
        
        initial = np.array([1.0])
        bounds = (0, np.inf)
        
        optimizer = AdamW(max_iter=100, use_binary_search=False, verbose=False, bounds=bounds)
        theta = optimizer.optimize(X, y, initial, cost, grad)
        
        assert theta[0] >= 0

    def test_constrained_unit_interval(self):
        """Test constraining parameters to [0, 1] with AdamW."""
        def cost(theta, X, y):
            return np.mean((X @ theta - y) ** 2)
        
        def grad(theta, X, y):
            return 2 * X.T @ (X @ theta - y) / len(y)
        
        X = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        y = np.array([0.5, 0.8, 1.3], dtype=float)
        initial = np.array([0.5, 0.5])
        bounds = (0, 1)
        
        optimizer = AdamW(max_iter=100, use_binary_search=False, verbose=False, bounds=bounds)
        theta = optimizer.optimize(X, y, initial, cost, grad)
        
        assert np.all(theta >= 0)
        assert np.all(theta <= 1)

    def test_per_parameter_bounds(self):
        """Test different bounds for each parameter with AdamW."""
        def cost(theta, X, y):
            return np.mean((X @ theta - y) ** 2)
        
        def grad(theta, X, y):
            return 2 * X.T @ (X @ theta - y) / len(y)
        
        X = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        y = np.array([2, 10, 12], dtype=float)
        initial = np.array([1.0, 5.0])
        
        bounds = (np.array([0, 0]), np.array([5, 15]))
        
        optimizer = AdamW(max_iter=100, use_binary_search=False, verbose=False, bounds=bounds)
        theta = optimizer.optimize(X, y, initial, cost, grad)
        
        assert 0 <= theta[0] <= 5
        assert 0 <= theta[1] <= 15

    def test_constrained_probability_regression(self):
        """Test real use case: constraining regression coefficients to form valid probabilities."""
        # Logistic regression-like problem where we want output in [0, 1]
        def cost(theta, X, y):
            predictions = 1 / (1 + np.exp(-(X @ theta)))  # Sigmoid
            return np.mean((predictions - y) ** 2)
        
        def grad(theta, X, y):
            predictions = 1 / (1 + np.exp(-(X @ theta)))
            error = predictions - y
            sigmoid_grad = predictions * (1 - predictions)
            return 2 * X.T @ (error * sigmoid_grad) / len(y)
        
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.uniform(0, 1, 50)  # Target probabilities
        
        initial = np.zeros(3)
        bounds = (-5, 5)  # Reasonable bounds for logistic coefficients
        
        optimizer = AdamW(max_iter=200, use_binary_search=False, verbose=False, bounds=bounds)
        theta = optimizer.optimize(X, y, initial, cost, grad)
        
        # All coefficients should be in [-5, 5]
        assert np.all(theta >= -5)
        assert np.all(theta <= 5)


class TestProjectionMethod:
    """Test projection methods directly."""

    def test_projection_scalar_bounds(self):
        """Test projection with scalar bounds."""
        optimizer = BinaryRateOptimizer(bounds=(0, 1), verbose=False)
        
        # Test various inputs
        assert optimizer._project_to_bounds(np.array([0.5]))[0] == 0.5
        assert optimizer._project_to_bounds(np.array([-0.5]))[0] == 0.0  # Clipped to lower
        assert optimizer._project_to_bounds(np.array([1.5]))[0] == 1.0   # Clipped to upper
        
        # Multi-dimensional
        theta = np.array([0.5, -0.5, 1.5])
        projected = optimizer._project_to_bounds(theta)
        expected = np.array([0.5, 0.0, 1.0])
        assert np.allclose(projected, expected)

    def test_projection_per_parameter_bounds(self):
        """Test projection with per-parameter bounds."""
        lower = np.array([0, -10, 5])
        upper = np.array([1, 10, 15])
        optimizer = BinaryRateOptimizer(bounds=(lower, upper), verbose=False)
        
        theta = np.array([0.5, 0, 10])
        projected = optimizer._project_to_bounds(theta)
        assert np.allclose(projected, [0.5, 0, 10])  # All within bounds
        
        theta = np.array([-1, -20, 20])
        projected = optimizer._project_to_bounds(theta)
        assert np.allclose(projected, [0, -10, 15])  # All clipped

    def test_projection_no_bounds(self):
        """Test that projection with no bounds returns unchanged."""
        optimizer = BinaryRateOptimizer(verbose=False)  # No bounds
        
        theta = np.array([100, -100, 0])
        projected = optimizer._project_to_bounds(theta)
        assert np.allclose(projected, theta)  # Unchanged
