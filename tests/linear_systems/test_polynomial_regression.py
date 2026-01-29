"""
Comprehensive tests for BinaryGaussSeidel polynomial regression.

Tests the fit_polynomial method with various polynomial degrees,
comparing results with numpy.polyfit for validation.
"""

import pytest
import numpy as np
from math_toolkit.linear_systems import BinaryGaussSeidel


class TestBinaryGaussSeidelPolynomialRegression:
    """Test polynomial regression functionality."""

    def test_linear_fit_degree_1(self):
        """Test fitting a linear polynomial (degree 1)."""
        # Perfect linear data: y = 2x + 3
        x_data = np.linspace(0, 10, 20)
        y_data = 2 * x_data + 3
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.fit_polynomial(x_data, y_data, degree=1)
        
        # Coefficients should be [3, 2] (a0=3, a1=2)
        assert abs(result.x[0] - 3.0) < 0.01, f"Expected a0=3, got {result.x[0]}"
        assert abs(result.x[1] - 2.0) < 0.01, f"Expected a1=2, got {result.x[1]}"
        assert result.converged is True

    def test_quadratic_fit_degree_2(self):
        """Test fitting a quadratic polynomial (degree 2)."""
        # Perfect quadratic data: y = x^2 + 2x + 1
        x_data = np.linspace(-5, 5, 30)
        y_data = x_data**2 + 2*x_data + 1
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.fit_polynomial(x_data, y_data, degree=2)
        
        # Coefficients should be [1, 2, 1] (a0=1, a1=2, a2=1)
        assert abs(result.x[0] - 1.0) < 0.01
        assert abs(result.x[1] - 2.0) < 0.01
        assert abs(result.x[2] - 1.0) < 0.01
        assert result.converged is True

    def test_cubic_fit_degree_3(self):
        """Test fitting a cubic polynomial (degree 3)."""
        # Perfect cubic data: y = x^3 - 2x^2 + 3x - 1
        x_data = np.linspace(-3, 3, 40)
        y_data = x_data**3 - 2*x_data**2 + 3*x_data - 1
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.fit_polynomial(x_data, y_data, degree=3)
        
        # Coefficients should be [-1, 3, -2, 1]
        expected = np.array([-1, 3, -2, 1])
        assert np.allclose(result.x, expected, atol=0.01)
        assert result.converged is True

    def test_quartic_fit_degree_4(self):
        """Test fitting a quartic polynomial (degree 4)."""
        # Quartic: y = 0.5x^4 - x^3 + 2x^2 - 3x + 1
        x_data = np.linspace(-2, 2, 50)
        y_data = 0.5*x_data**4 - x_data**3 + 2*x_data**2 - 3*x_data + 1
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.fit_polynomial(x_data, y_data, degree=4)
        
        # Coefficients should be [1, -3, 2, -1, 0.5]
        expected = np.array([1, -3, 2, -1, 0.5])
        assert np.allclose(result.x, expected, atol=0.01)

    def test_fit_with_noise(self):
        """Test polynomial fit with noisy data."""
        np.random.seed(42)
        x_data = np.linspace(0, 10, 50)
        y_true = 3*x_data**2 + 2*x_data + 1
        y_data = y_true + np.random.normal(0, 5, len(x_data))  # Add noise
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.fit_polynomial(x_data, y_data, degree=2)
        
        # Should be close to true coefficients [1, 2, 3]
        # but not exact due to noise
        assert abs(result.x[0] - 1.0) < 10  # Intercept
        assert abs(result.x[1] - 2.0) < 5   # Linear term
        assert abs(result.x[2] - 3.0) < 2   # Quadratic term
        assert result.converged is True

    def test_compare_with_numpy_polyfit_degree_2(self):
        """Compare results with numpy.polyfit for degree 2."""
        x_data = np.linspace(-5, 5, 30)
        y_data = 2*x_data**2 + 3*x_data + 1
        
        # Our implementation
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.fit_polynomial(x_data, y_data, degree=2)
        
        # NumPy polyfit (returns coefficients in reverse order!)
        numpy_coeffs = np.polyfit(x_data, y_data, deg=2)[::-1]
        
        # Should match closely
        assert np.allclose(result.x, numpy_coeffs, atol=0.1)

    def test_compare_with_numpy_polyfit_degree_3(self):
        """Compare results with numpy.polyfit for degree 3."""
        x_data = np.linspace(-3, 3, 40)
        y_data = x_data**3 + 2*x_data**2 - x_data + 5
        
        # Our implementation
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.fit_polynomial(x_data, y_data, degree=3)
        
        # NumPy polyfit
        numpy_coeffs = np.polyfit(x_data, y_data, deg=3)[::-1]
        
        # Should match closely
        assert np.allclose(result.x, numpy_coeffs, atol=0.1)

    def test_insufficient_data_points(self):
        """Test that insufficient data points raises error."""
        x_data = np.array([1, 2])  # Only 2 points
        y_data = np.array([3, 4])
        
        solver = BinaryGaussSeidel(verbose=False)
        
        # Need at least 3 points for degree 2
        with pytest.raises(ValueError, match="Need at least 3 data points"):
            solver.fit_polynomial(x_data, y_data, degree=2)

    def test_mismatched_data_lengths(self):
        """Test that mismatched x and y lengths raise error."""
        x_data = np.array([1, 2, 3])
        y_data = np.array([4, 5])
        
        solver = BinaryGaussSeidel(verbose=False)
        
        with pytest.raises(ValueError, match="must have same length"):
            solver.fit_polynomial(x_data, y_data, degree=1)

    def test_invalid_degree_zero(self):
        """Test that degree < 1 raises error."""
        x_data = np.array([1, 2, 3])
        y_data = np.array([4, 5, 6])
        
        solver = BinaryGaussSeidel(verbose=False)
        
        with pytest.raises(ValueError, match="degree must be >= 1"):
            solver.fit_polynomial(x_data, y_data, degree=0)

    def test_polynomial_result_metadata(self):
        """Test that result contains polynomial-specific metadata."""
        x_data = np.linspace(0, 5, 20)
        y_data = x_data**2 + 1
        
        solver = BinaryGaussSeidel(verbose=False)
        result = solver.fit_polynomial(x_data, y_data, degree=2)
        
        # Check metadata
        assert hasattr(result, 'polynomial_degree')
        assert hasattr(result, 'num_data_points')
        assert result.polynomial_degree == 2
        assert result.num_data_points == 20

    def test_prediction_with_fitted_polynomial(self):
        """Test using fitted coefficients for prediction."""
        # Fit polynomial
        x_train = np.linspace(0, 10, 30)
        y_train = 2*x_train**2 + 3*x_train + 1
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.fit_polynomial(x_train, y_train, degree=2)
        
        # Make predictions on new data
        x_test = np.array([5.0, 7.5, 10.0])
        y_pred = np.polyval(result.x[::-1], x_test)  # polyval expects reversed order
        
        # True values
        y_true = 2*x_test**2 + 3*x_test + 1
        
        # Predictions should be accurate
        assert np.allclose(y_pred, y_true, atol=0.1)

    def test_high_degree_polynomial(self):
        """Test fitting higher degree polynomial (degree 5)."""
        x_data = np.linspace(-2, 2, 60)
        # y = x^5 - 2x^4 + 3x^3 - x^2 + 2x - 1
        y_data = x_data**5 - 2*x_data**4 + 3*x_data**3 - x_data**2 + 2*x_data - 1
        
        solver = BinaryGaussSeidel(tolerance=1e-6, max_iterations=2000, verbose=False)
        result = solver.fit_polynomial(x_data, y_data, degree=5)
        
        expected = np.array([-1, 2, -1, 3, -2, 1])
        assert np.allclose(result.x, expected, atol=0.1)

    def test_fit_constant_function(self):
        """Test fitting constant data (should work with degree 1)."""
        x_data = np.linspace(0, 10, 20)
        y_data = np.full(20, 5.0)  # Constant y = 5
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.fit_polynomial(x_data, y_data, degree=1)
        
        # Should get a0 ≈ 5, a1 ≈ 0
        assert abs(result.x[0] - 5.0) < 0.1
        assert abs(result.x[1]) < 0.1

    def test_negative_x_values(self):
        """Test fitting with negative x values."""
        x_data = np.linspace(-10, -1, 25)
        y_data = -2*x_data**2 + 3*x_data - 4
        
        solver = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
        result = solver.fit_polynomial(x_data, y_data, degree=2)
        
        expected = np.array([-4, 3, -2])
        assert np.allclose(result.x, expected, atol=0.01)

    def test_large_dataset(self):
        """Test polynomial regression with large noisy dataset."""
        np.random.seed(123)
        x_data = np.linspace(0, 20, 200)
        y_data = 0.1*x_data**3 - 2*x_data**2 + 5*x_data + 10 + np.random.normal(0, 2, 200)
        
        # Large noisy system needs relaxed tolerance
        solver = BinaryGaussSeidel(tolerance=1e-4, max_iterations=2000, verbose=False)
        result = solver.fit_polynomial(x_data, y_data, degree=3)
        
        # With noise, coefficients will vary but should be in reasonable range
        # True coefficients: [10, 5, -2, 0.1]
        # Allow wider bounds for noisy regression
        assert 5 < result.x[0] < 20    # Intercept: roughly 10 ± 10
        assert 0 < result.x[1] < 10    # Linear: roughly 5 ± 5
        assert -5 < result.x[2] < 0    # Quadratic: roughly -2 ± 3
        assert -0.5 < result.x[3] < 1  # Cubic: roughly 0.1 ± 0.6
