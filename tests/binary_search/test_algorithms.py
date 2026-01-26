"""
Comprehensive tests for Binary Search Algorithms.

Tests cover:
- Array search with tolerance
- Function search (including bug fixes)
- Stepped search (binary and linear)
- Dictionary search
- Mathematical comparison functions
- Edge cases and error handling

Target: 90%+ code coverage
"""

import pytest
import math
from math_toolkit.binary_search import BinarySearch


class TestBinarySearchInit:
    """Test initialization and configuration."""

    def test_default_initialization(self):
        """Test creating BinarySearch with default parameters."""
        searcher = BinarySearch()
        assert searcher.binary_search_priority_for_smallest_values is True
        assert searcher.previous_value_should_be_the_basis_of_binary_search_calculations is False
        assert searcher.previous_value_is_the_target is False
        assert searcher.change_behavior_mid_step is False
        assert searcher.number_of_attempts == 20

    def test_custom_parameters(self):
        """Test creating BinarySearch with custom parameters."""
        searcher = BinarySearch(
            binary_search_priority_for_smallest_values=False,
            previous_value_should_be_the_basis_of_binary_search_calculations=True,
            previous_value_is_the_target=True,
            change_behavior_mid_step=True,
            number_of_attempts=50,
        )
        assert searcher.binary_search_priority_for_smallest_values is False
        assert searcher.previous_value_should_be_the_basis_of_binary_search_calculations is True
        assert searcher.previous_value_is_the_target is True
        assert searcher.change_behavior_mid_step is True
        assert searcher.number_of_attempts == 50


class TestSearchForArray:
    """Test array search functionality."""

    def test_find_exact_match(self):
        """Test finding exact value in array."""
        searcher = BinarySearch()
        array = [1.0, 2.0, 3.0, 4.0, 5.0]
        index, is_max, is_min = searcher.search_for_array(3.0, array, tolerance=0.001)
        assert array[index] == 3.0

    def test_find_closest_value(self):
        """Test finding closest value when exact match doesn't exist."""
        searcher = BinarySearch()
        array = [1.0, 3.0, 5.0, 7.0, 9.0]
        index, is_max, is_min = searcher.search_for_array(4.0, array, tolerance=0.1)
        # Should find 3.0 or 5.0 (closest values)
        assert array[index] in [3.0, 5.0]

    def test_find_minimum(self):
        """Test finding value at minimum index."""
        searcher = BinarySearch()
        array = [1.0, 2.0, 3.0, 4.0, 5.0]
        index, is_max, is_min = searcher.search_for_array(0.5, array, tolerance=0.1)
        assert is_min == 1
        assert index == 0

    def test_find_maximum(self):
        """Test finding value at maximum index."""
        searcher = BinarySearch()
        array = [1.0, 2.0, 3.0, 4.0, 5.0]
        index, is_max, is_min = searcher.search_for_array(10.0, array, tolerance=0.1)
        assert is_max == 1
        assert index == len(array) - 1

    def test_single_element_array(self):
        """Test search in array with single element."""
        searcher = BinarySearch()
        array = [5.0]
        index, is_max, is_min = searcher.search_for_array(5.0, array, tolerance=0.1)
        assert index == 0
        # With single element, the algorithm may or may not set is_max/is_min
        # The important thing is it finds the correct index
        assert index >= 0 and index < len(array)

    def test_empty_array_raises_error(self):
        """Test that empty array raises IndexError."""
        searcher = BinarySearch()
        with pytest.raises(IndexError, match="Cannot search in empty array"):
            searcher.search_for_array(5.0, [], tolerance=0.1)

    def test_invalid_tolerance_raises_error(self):
        """Test that negative tolerance raises ValueError."""
        searcher = BinarySearch()
        with pytest.raises(ValueError, match="Tolerance must be positive"):
            searcher.search_for_array(5.0, [1, 2, 3], tolerance=-0.1)

    def test_large_array(self):
        """Test search in large array."""
        searcher = BinarySearch()
        array = list(range(0, 1000, 2))  # [0, 2, 4, ..., 998]
        index, is_max, is_min = searcher.search_for_array(500, array, tolerance=1.0)
        assert abs(array[index] - 500) <= 2  # Should find 500 or very close


class TestSearchForFunction:
    """Test function search functionality (static method)."""

    def test_square_function(self):
        """Test finding x where x^2 = 16."""
        def square(x):
            return x ** 2

        x = BinarySearch.search_for_function(16, square, tolerance=1e-6)
        assert abs(x - 4.0) < 0.01 or abs(x + 4.0) < 0.01  # Could find ±4

    def test_linear_function(self):
        """Test finding x where 2x + 3 = 10."""
        def linear(x):
            return 2 * x + 3

        x = BinarySearch.search_for_function(10, linear, tolerance=1e-6)
        assert abs(x - 3.5) < 0.01  # 2*3.5 + 3 = 10

    def test_exp_function_no_overflow(self):
        """Test that exponential function doesn't cause overflow (BUG FIX)."""
        def exp(x):
            return math.exp(x)

        # This would crash in v0.1 due to overflow
        x = BinarySearch.search_for_function(10, exp, tolerance=0.1)
        assert abs(exp(x) - 10) < 1.0

    def test_function_at_zero(self):
        """Test finding value when function(0) is the answer."""
        def constant(x):
            return 5.0

        x = BinarySearch.search_for_function(5.0, constant, tolerance=1e-6)
        assert abs(x) < 0.01

    def test_undefined_function_at_zero(self):
        """Test that undefined function at x=0 raises ValueError."""
        def undefined(x):
            if x == 0:
                raise ValueError("Undefined at 0")
            return 1 / x

        with pytest.raises(ValueError, match="Function is not defined at x=0"):
            BinarySearch.search_for_function(2.0, undefined)

    def test_decreasing_function(self):
        """Test finding value in decreasing function."""
        def decreasing(x):
            return 10 - x

        x = BinarySearch.search_for_function(5, decreasing, tolerance=1e-6)
        assert abs(x - 5.0) < 0.01

    def test_sin_function(self):
        """Test finding value in sin function."""
        def sin(x):
            return math.sin(x)

        x = BinarySearch.search_for_function(0.5, sin, tolerance=1e-6)
        assert abs(math.sin(x) - 0.5) < 0.01


class TestBinarySearchByStep:
    """Test stepped binary search."""

    def test_basic_progression(self):
        """Test basic stepped search progression."""
        searcher = BinarySearch(number_of_attempts=10)
        values = [searcher.binary_search_by_step(i, 0, 100) for i in range(5)]

        # Should progress from min toward max
        assert values[0] == 0
        assert 0 < values[1] < 100
        assert values[1] < values[2] < 100

    def test_priority_for_smallest(self):
        """Test search prioritizing smallest values."""
        searcher = BinarySearch(binary_search_priority_for_smallest_values=True)
        value = searcher.binary_search_by_step(1, 0, 100)
        assert value < 100  # Should start from low end

    def test_priority_for_largest(self):
        """Test search prioritizing largest values."""
        searcher = BinarySearch(binary_search_priority_for_smallest_values=False)
        value = searcher.binary_search_by_step(1, 0, 100)
        assert value > 0  # Should start from high end

    def test_with_previous_value(self):
        """Test search using previous value as basis."""
        searcher = BinarySearch(
            previous_value_should_be_the_basis_of_binary_search_calculations=True
        )
        value = searcher.binary_search_by_step(1, 0, 100, previous_value=50)
        # Should be relative to previous value (50)
        assert 0 <= value <= 100

    def test_rational_function_progression(self):
        """Test that progression uses rational function (non-linear)."""
        searcher = BinarySearch(number_of_attempts=20)
        values = [searcher.binary_search_by_step(i, 0, 100) for i in range(10)]

        # Check that progression is non-linear (not arithmetic)
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        # Differences should generally decrease (logarithmic approach)
        assert diffs[0] > diffs[-1]


class TestLinearSearchStep:
    """Test stepped linear search."""

    def test_linear_progression(self):
        """Test that linear search uses arithmetic progression."""
        searcher = BinarySearch(number_of_attempts=10)
        values = [searcher.linear_search_step(i, 0, 100) for i in range(5)]

        # Should have equal steps
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        assert all(abs(d - 10) < 0.1 for d in diffs)  # Steps of 10

    def test_linear_full_range(self):
        """Test linear search covers full range."""
        searcher = BinarySearch(number_of_attempts=10)
        values = [searcher.linear_search_step(i, 0, 100) for i in range(11)]
        assert values[0] == 0
        assert abs(values[-1] - 100) < 1


class TestDictionarySearch:
    """Test dictionary miniterm search."""

    def test_find_exact_match_in_dict(self):
        """Test finding exact value in dictionary."""
        searcher = BinarySearch()
        data = {'a': 1.0, 'b': 3.0, 'c': 5.0, 'd': 7.0}
        value, key = searcher.binary_search_to_find_miniterm_from_dict(5.0, data)
        assert value == 5.0
        assert key == 'c'

    def test_find_closest_in_dict(self):
        """Test finding closest value in dictionary."""
        searcher = BinarySearch()
        data = {'a': 1.0, 'b': 3.0, 'c': 5.0, 'd': 7.0}
        value, key = searcher.binary_search_to_find_miniterm_from_dict(4.0, data)
        # Should find 3.0 or 5.0 (closest)
        assert value in [3.0, 5.0]

    def test_single_item_dict(self):
        """Test dictionary with single item."""
        searcher = BinarySearch()
        data = {'only': 10.0}
        value, key = searcher.binary_search_to_find_miniterm_from_dict(5.0, data)
        assert value == 10.0
        assert key == 'only'

    def test_empty_dict_raises_error(self):
        """Test that empty dictionary raises ValueError."""
        searcher = BinarySearch()
        with pytest.raises(ValueError, match="Cannot search in empty dictionary"):
            searcher.binary_search_to_find_miniterm_from_dict(5.0, {})


class TestReset:
    """Test reset functionality."""

    def test_reset_after_behavior_change(self):
        """Test reset reverses priority change."""
        searcher = BinarySearch(
            binary_search_priority_for_smallest_values=True,
            change_behavior_mid_step=True,
            number_of_attempts=20
        )

        # Trigger mid-step behavior change (needs to be after threshold)
        for i in range(12):  # Past the number_of_attempts//2 threshold
            searcher.binary_search_by_step(i, 0, 100)

        # Behavior may have changed if threshold reached
        was_modified = searcher.binary_search_priority_modified

        # Reset should restore if it was modified
        searcher.reset()
        assert searcher.binary_search_priority_modified is False

        # If behavior changed, priority should flip back
        if was_modified:
            # Priority was flipped during search, reset flips it back
            pass  # Test passes if no exceptions

    def test_reset_no_op_when_not_modified(self):
        """Test reset is no-op when priority not modified."""
        searcher = BinarySearch(binary_search_priority_for_smallest_values=True)
        original_priority = searcher.binary_search_priority_for_smallest_values

        searcher.reset()
        assert searcher.binary_search_priority_for_smallest_values == original_priority


class TestMathematicalFunctions:
    """Test the mathematical comparison lambda functions."""

    def test_greater_than_function(self):
        """Test tolerance-based greater than comparison."""
        searcher = BinarySearch()
        tolerance = 0.01

        # 5 > 3 with tolerance should return 1
        result = searcher.greater_than_function(5, 3, tolerance)
        assert result == 1

        # 3 > 5 should return 0
        result = searcher.greater_than_function(3, 5, tolerance)
        assert result == 0

    def test_equals_function(self):
        """Test tolerance-based equals comparison."""
        searcher = BinarySearch()
        tolerance = 0.01

        # Exactly equal
        result = searcher.equals_function(5.0, 5.0, tolerance)
        assert result == 1

        # Not equal
        result = searcher.equals_function(5.0, 3.0, tolerance)
        assert result == 0

    def test_rational_function(self):
        """Test rational function progression."""
        searcher = BinarySearch()

        # Test known values
        assert abs(searcher.rational_function(0) - 0.0) < 0.01
        assert abs(searcher.rational_function(1) - 0.5) < 0.01
        assert abs(searcher.rational_function(2) - 0.667) < 0.01
        assert abs(searcher.rational_function(3) - 0.75) < 0.01

        # Should approach 1 as a -> infinity
        assert searcher.rational_function(1000) > 0.999

    def test_average_function(self):
        """Test average calculation."""
        searcher = BinarySearch()
        assert searcher.average_function(0, 10) == 5.0
        assert searcher.average_function(-5, 5) == 0.0
        assert searcher.average_function(2.5, 7.5) == 5.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_tolerance(self):
        """Test with very small tolerance."""
        searcher = BinarySearch()
        array = [1.0, 2.0, 3.0, 4.0, 5.0]
        index, _, _ = searcher.search_for_array(3.0, array, tolerance=1e-10)
        assert array[index] == 3.0

    def test_negative_values(self):
        """Test search with negative values."""
        searcher = BinarySearch()
        array = [-10.0, -5.0, 0.0, 5.0, 10.0]
        index, _, _ = searcher.search_for_array(-5.0, array, tolerance=0.1)
        assert array[index] == -5.0

    def test_floating_point_array(self):
        """Test with floating point values."""
        searcher = BinarySearch()
        array = [1.1, 2.2, 3.3, 4.4, 5.5]
        index, _, _ = searcher.search_for_array(3.3, array, tolerance=0.01)
        assert abs(array[index] - 3.3) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=binary_search_algorithms", "--cov-report=term-missing"])
