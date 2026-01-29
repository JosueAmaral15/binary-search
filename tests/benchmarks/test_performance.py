"""
Performance benchmarks for math_toolkit algorithms.

These tests establish baseline performance expectations and catch regressions.
Run with: pytest tests/benchmarks/ -v
"""

import time
import pytest
import numpy as np
from math_toolkit.binary_search import BinarySearch
from math_toolkit.optimization import BinaryRateOptimizer, AdamW
from math_toolkit.linear_systems import BinaryGaussSeidel, NonLinearGaussSeidel


class TestBinarySearchPerformance:
    """Performance benchmarks for BinarySearch algorithms."""

    def test_basic_search_performance(self):
        """Binary search should complete in < 10ms for 1K elements."""
        arr = list(range(1_000))
        target = 750
        
        bs = BinarySearch()
        start = time.perf_counter()
        result = bs.search_for_array(target, arr)
        elapsed = time.perf_counter() - start
        
        assert len(result) == 3  # Returns tuple (index, low, high)
        assert elapsed < 0.010, f"Search took {elapsed:.6f}s, expected < 0.010s"

    def test_large_array_search(self):
        """Binary search should handle 10K elements in < 20ms."""
        arr = list(range(10_000))
        target = 9_500
        
        bs = BinarySearch()
        start = time.perf_counter()
        result = bs.search_for_array(target, arr)
        elapsed = time.perf_counter() - start
        
        assert len(result) == 3  # Returns tuple (index, low, high)
        assert elapsed < 0.020, f"Large search took {elapsed:.6f}s, expected < 0.020s"


class TestBinaryRateOptimizerPerformance:
    """Performance benchmarks for BinaryRateOptimizer."""

    def test_simple_linear_optimization(self):
        """Optimize simple linear regression in < 100ms."""
        def cost_fn(theta, X, y):
            predictions = X * theta
            return np.mean((predictions - y) ** 2)
        
        def gradient_fn(theta, X, y):
            predictions = X * theta
            error = predictions - y
            return 2 * np.mean(error * X)
        
        X = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        initial_theta = np.array([0.0])
        
        optimizer = BinaryRateOptimizer(max_iter=50, verbose=False)
        
        start = time.perf_counter()
        final_theta = optimizer.optimize(X, y, initial_theta, cost_fn, gradient_fn)
        elapsed = time.perf_counter() - start
        
        assert abs(final_theta[0] - 2.0) < 0.1
        assert elapsed < 0.100, f"Optimization took {elapsed:.6f}s, expected < 0.100s"

    def test_high_dimensional_optimization(self):
        """Optimize 20-dimensional problem in < 200ms."""
        def cost_fn(theta, X, y):
            return np.mean((X @ theta - y) ** 2)
        
        def gradient_fn(theta, X, y):
            return 2 * X.T @ (X @ theta - y) / len(y)
        
        np.random.seed(42)
        X = np.random.randn(50, 20)
        true_theta = np.random.randn(20)
        y = X @ true_theta
        initial_theta = np.zeros(20)
        
        optimizer = BinaryRateOptimizer(max_iter=100, verbose=False)
        
        start = time.perf_counter()
        final_theta = optimizer.optimize(X, y, initial_theta, cost_fn, gradient_fn)
        elapsed = time.perf_counter() - start
        
        # Should find something close
        assert cost_fn(final_theta, X, y) < 1.0
        assert elapsed < 0.200, f"High-dim optimization took {elapsed:.6f}s, expected < 0.200s"


class TestAdamWPerformance:
    """Performance benchmarks for AdamW optimizer."""

    def test_simple_optimization_speed(self):
        """AdamW should optimize simple function in < 100ms."""
        def cost_fn(theta, X, y):
            return np.mean((X * theta - y) ** 2)
        
        def gradient_fn(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)
        
        X = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        initial_theta = np.array([0.0])
        
        optimizer = AdamW(max_iter=100, verbose=False)
        
        start = time.perf_counter()
        final_theta = optimizer.optimize(X, y, initial_theta, cost_fn, gradient_fn)
        elapsed = time.perf_counter() - start
        
        assert abs(final_theta[0] - 2.0) < 0.5
        assert elapsed < 0.100, f"AdamW took {elapsed:.6f}s, expected < 0.100s"


class TestBinaryGaussSeidelPerformance:
    """Performance benchmarks for BinaryGaussSeidel linear solver."""

    def test_small_system_3x3(self):
        """Solve 3×3 system in < 10ms."""
        A = np.array([[4, 1, 0],
                      [1, 4, 1],
                      [0, 1, 4]], dtype=float)
        b = np.array([5, 6, 5], dtype=float)
        
        solver = BinaryGaussSeidel(tolerance=1e-6, max_iterations=100, verbose=False)
        
        start = time.perf_counter()
        result = solver.solve(A, b)
        elapsed = time.perf_counter() - start
        
        assert result.converged
        assert elapsed < 0.010, f"3×3 solve took {elapsed:.6f}s, expected < 0.010s"

    def test_medium_system_10x10(self):
        """Solve 10×10 system in < 50ms."""
        np.random.seed(42)
        A = np.diag(np.full(10, 10.0)) + np.random.randn(10, 10) * 0.5
        b = np.random.randn(10)
        
        solver = BinaryGaussSeidel(tolerance=1e-6, max_iterations=200, verbose=False)
        
        start = time.perf_counter()
        result = solver.solve(A, b)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.050, f"10×10 solve took {elapsed:.6f}s, expected < 0.050s"

    def test_large_system_100x100(self):
        """Solve 100×100 system in < 1000ms."""
        np.random.seed(42)
        A = np.diag(np.full(100, 20.0)) + np.random.randn(100, 100) * 0.5
        b = np.random.randn(100)
        
        solver = BinaryGaussSeidel(tolerance=1e-4, max_iterations=500, verbose=False)
        
        start = time.perf_counter()
        result = solver.solve(A, b)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.000, f"100×100 solve took {elapsed:.6f}s, expected < 1.000s"

    def test_polynomial_regression_performance(self):
        """Polynomial regression (degree 3, 100 points) should complete in < 100ms."""
        np.random.seed(42)
        x_data = np.linspace(0, 10, 100)
        y_data = 2*x_data**3 - 3*x_data**2 + 5*x_data + 1
        
        solver = BinaryGaussSeidel(tolerance=1e-6, max_iterations=500, verbose=False)
        
        start = time.perf_counter()
        result = solver.fit_polynomial(x_data, y_data, degree=3)
        elapsed = time.perf_counter() - start
        
        # May not always converge for polynomial problems, but should be fast
        assert elapsed < 0.100, f"Polynomial fit took {elapsed:.6f}s, expected < 0.100s"


class TestNonLinearGaussSeidelPerformance:
    """Performance benchmarks for NonLinearGaussSeidel."""

    def test_simple_2d_system(self):
        """Solve simple 2D nonlinear system in < 50ms."""
        # x + y = 3
        # x*y = 2
        # Solution: (1, 2) or (2, 1)
        f1 = lambda x, y: x + y - 3
        f2 = lambda x, y: x * y - 2
        
        solver = NonLinearGaussSeidel(
            functions=[f1, f2],
            tolerance=1e-6,
            max_iterations=100,
            verbose=False
        )
        
        start = time.perf_counter()
        result = solver.solve(initial_guess=[1.5, 1.5])
        elapsed = time.perf_counter() - start
        
        assert result.converged
        assert elapsed < 0.050, f"2D nonlinear solve took {elapsed:.6f}s, expected < 0.050s"

    def test_3d_nonlinear_system(self):
        """Solve 3D nonlinear system in < 150ms."""
        # System: x^2 + y^2 + z^2 = 14, x + y + z = 6, xy = 2
        f1 = lambda x, y, z: x**2 + y**2 + z**2 - 14
        f2 = lambda x, y, z: x + y + z - 6
        f3 = lambda x, y, z: x * y - 2
        
        solver = NonLinearGaussSeidel(
            functions=[f1, f2, f3],
            tolerance=1e-4,
            max_iterations=200,
            verbose=False
        )
        
        start = time.perf_counter()
        result = solver.solve(initial_guess=[1.0, 2.0, 3.0])
        elapsed = time.perf_counter() - start
        
        # May or may not converge, but should be fast
        assert elapsed < 0.150, f"3D nonlinear solve took {elapsed:.6f}s, expected < 0.150s"


def test_overall_package_import_time():
    """Package import should complete in < 200ms."""
    import sys
    import importlib
    
    # Remove from cache if present
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('math_toolkit')]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    start = time.perf_counter()
    import math_toolkit
    from math_toolkit.binary_search import BinarySearch
    from math_toolkit.optimization import BinaryRateOptimizer, AdamW
    from math_toolkit.linear_systems import BinaryGaussSeidel, NonLinearGaussSeidel
    elapsed = time.perf_counter() - start
    
    assert elapsed < 0.200, f"Package import took {elapsed:.6f}s, expected < 0.200s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

