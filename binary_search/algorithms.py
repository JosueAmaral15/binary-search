"""
Binary Search Algorithms - Advanced search implementations with tolerance-based comparisons.

This module provides a comprehensive BinarySearch class that implements various
binary search strategies including array search, function search, stepped search,
and dictionary-based search with mathematical tolerance comparisons.

The unique feature of this implementation is the use of lambda-based tolerance
comparisons instead of traditional if/else statements, providing mathematical
elegance and precision in edge case handling.
"""

from math import ceil, floor, isfinite  # BUG FIX: Added missing ceil, floor imports
from typing import Callable, Dict, Tuple, Any, Optional


class BinarySearch:
    """
    Advanced Binary Search with tolerance-based comparisons and multiple search modes.
    
    This class implements several binary search algorithms with a unique mathematical
    approach using lambda functions for comparisons. Instead of traditional if/else
    statements, it uses tolerance-based mathematical formulas that handle edge cases
    with precision.
    
    Key Features:
        - Array search with tolerance
        - Function search (find x where f(x) = y)
        - Stepped binary/linear search
        - Dictionary-based miniterm search
        - Configurable search direction and behavior
    
    Attributes:
        binary_search_priority_for_smallest_values: Search direction preference
        previous_value_should_be_the_basis_of_binary_search_calculations: Use previous value as reference
        previous_value_is_the_target: Previous value is the search target
        change_behavior_mid_step: Change search direction mid-execution
        number_of_attempts: Maximum search iterations
    
    Mathematical Approach:
        The class uses lambda functions for comparisons:
        - greater_than_function: Tolerance-based > comparison
        - equals_function: Tolerance-based == comparison
        - rational_function: Non-linear progression (a/(a+1))
    
    Example:
        >>> searcher = BinarySearch()
        >>> array = [1, 3, 5, 7, 9, 11, 13, 15]
        >>> index, is_max, is_min = searcher.search_for_array(7, array)
        >>> print(f"Found at index: {index}")
        Found at index: 3
    """

    def __init__(
        self,
        binary_search_priority_for_smallest_values: bool = True,
        previous_value_should_be_the_basis_of_binary_search_calculations: bool = False,
        previous_value_is_the_target: bool = False,
        change_behavior_mid_step: bool = False,
        number_of_attempts: int = 20,
    ) -> None:
        """
        Initialize the Binary Search with configurable behavior.
        
        Args:
            binary_search_priority_for_smallest_values: If True, search prioritizes
                finding smallest values first. If False, prioritizes largest values.
                Default: True
            
            previous_value_should_be_the_basis_of_binary_search_calculations: If True,
                uses previous search result as basis for next search. Useful for
                iterative searches. Default: False
            
            previous_value_is_the_target: If True, treats the previous value as the
                target destination of the search. Default: False
            
            change_behavior_mid_step: If True, reverses search priority halfway through
                the search iterations. Useful for bidirectional searches. Default: False
            
            number_of_attempts: Maximum number of search iterations before stopping.
                More attempts = higher precision but slower. Default: 20
        
        Note:
            The mathematical comparison functions use tolerance-based formulas
            instead of traditional conditionals, providing precise edge case handling.
        """
        # Mathematical comparison functions (tolerance-based)
        # These lambda functions replace traditional if/else with mathematical formulas
        self.greater_than_function = lambda x, r, tolerance: round(
            (1 / (2 * tolerance)) * (abs(x - r) - abs(x - r - tolerance) + tolerance)
        )
        
        self.equals_function = lambda x, r, tolerance: round(
            (1 / (2 * tolerance))
            * (abs(x - r + tolerance) - abs(x - r) + tolerance)
            * ((-1 / (2 * tolerance)) * (abs(x - r) - abs(x - r - tolerance) + tolerance) + 1)
        )
        
        self.average_function = lambda a, b: (a + b) / 2
        
        self.lowest_function = lambda average, initial_lower, result, expected_result, tolerance: (
            self.equals_function(average, initial_lower, tolerance)
            * self.greater_than_function(result, expected_result, tolerance)
        )
        
        self.greatest_function = lambda average, initial_upper, result, expected_result, tolerance: (
            self.equals_function(average, initial_upper, tolerance)
            * self.greater_than_function(expected_result, result, tolerance)
        )
        
        # Non-linear progression function for stepped search
        self.rational_function = lambda a: a / (a + 1)
        
        # Arithmetic progression for linear search
        self.arithmetic_progression_function = lambda a1, n, r: a1 + n * r
        
        # Selector functions with graduation (using ceil/floor - BUG FIX: now imported)
        self.selector_without_graduation_and_with_inclusion = lambda x, a, b, d, m: (
            (a - m) * (floor((1 / 2) * (abs(x - b + 1) - abs(x - b) + 1))) + m + d
        )
        
        self.equals_with_no_graduation = lambda x, b, d: (
            d
            * (-ceil((1 / 2) * (abs(x - b) - abs(x - b - 1) + 1)) + 1)
            * floor((1 / 2) * (abs(x - b + 1) - abs(x - b) + 1))
        )
        
        # Configuration values
        self.binary_search_priority_for_smallest_values = binary_search_priority_for_smallest_values
        self.previous_value_should_be_the_basis_of_binary_search_calculations = (
            previous_value_should_be_the_basis_of_binary_search_calculations
        )
        self.previous_value_is_the_target = previous_value_is_the_target
        self.number_of_attempts = number_of_attempts
        self.binary_search_priority_modified = False
        self.change_behavior_mid_step = change_behavior_mid_step

    def search_for_array(
        self, expected_result: float, array: list, tolerance: float = 0.001
    ) -> Tuple[int, int, int]:
        """
        Binary search in a sorted array with tolerance-based comparison.
        
        Searches for the index of the element closest to expected_result in the array.
        Uses mathematical tolerance comparisons to handle floating-point precision.
        
        Args:
            expected_result: The value to search for
            array: Sorted list of numerical values
            tolerance: Comparison tolerance for floating-point equality. Default: 0.001
        
        Returns:
            Tuple containing:
                - index (int): Index of the closest element
                - is_global_maximum (int): 1 if found at maximum index, 0 otherwise
                - is_global_minimum (int): 1 if found at minimum index, 0 otherwise
        
        Raises:
            IndexError: If array is empty
            ValueError: If tolerance is not positive
        
        Example:
            >>> searcher = BinarySearch()
            >>> array = [1.0, 2.5, 4.0, 5.5, 7.0]
            >>> index, is_max, is_min = searcher.search_for_array(4.2, array)
            >>> print(f"Closest value at index {index}: {array[index]}")
            Closest value at index 2: 4.0
        """
        if not array:
            raise IndexError("Cannot search in empty array")
        if tolerance <= 0:
            raise ValueError(f"Tolerance must be positive, got {tolerance}")
        
        array_length = len(array)
        initial_upper = array_length - 1
        upper_value1 = initial_upper
        initial_lower = 0
        lower_value1 = initial_lower
        average1 = initial_upper
        result1 = array[average1]
        is_global_maximum = 0
        is_global_minimum = 0
        
        continue_execution = self.greater_than_function(upper_value1, lower_value1 + 1, tolerance)
        
        while continue_execution and not is_global_maximum and not is_global_minimum:
            lower_value2 = average1 * self.greater_than_function(
                expected_result, result1, tolerance
            ) + lower_value1 * self.greater_than_function(result1, expected_result, tolerance)
            
            upper_value2 = upper_value1 * self.greater_than_function(
                expected_result, result1, tolerance
            ) + average1 * self.greater_than_function(result1, expected_result, tolerance)
            
            average2 = self.average_function(lower_value2, upper_value2) + average1 * self.equals_function(
                expected_result, result1, tolerance
            )
            
            average1 = average2
            average2 = int(average2)
            result2 = array[average2]
            lower_value1 = lower_value2
            upper_value1 = upper_value2
            result1 = result2
            
            continue_execution = self.greater_than_function(upper_value2, lower_value2 + 1, tolerance)
            is_global_maximum = self.greatest_function(
                average2, initial_upper, result2, expected_result, tolerance
            )
            is_global_minimum = self.lowest_function(
                average2, initial_lower, result2, expected_result, tolerance
            )

        return int(average1), is_global_maximum, is_global_minimum

    @staticmethod
    def search_for_function(
        y: float, function: Callable[[float], float], tolerance: float = 1e-6, max_iter: int = 1000
    ) -> float:
        """
        Find x such that function(x) ≈ y using binary search.
        
        This static method searches for the input value x that produces the target
        output y when passed to the given function. It automatically detects whether
        the function is increasing or decreasing and adapts the search strategy.
        
        BUG FIX: Improved overflow handling and bounds checking to prevent crashes
        with exponential functions like exp().
        
        Algorithm:
            1. Start at x=0 and evaluate f(0)
            2. Detect if function is increasing or decreasing
            3. Expand search interval in the correct direction
            4. Refine using binary search within found bounds
        
        Args:
            y: Target output value
            function: Callable that takes float and returns float
            tolerance: Acceptable error in f(x) - y. Default: 1e-6
            max_iter: Maximum refinement iterations. Default: 1000
        
        Returns:
            The value x such that |function(x) - y| < tolerance
        
        Raises:
            ValueError: If function is not defined at x=0
            OverflowError: If function values exceed float limits
        
        Example:
            >>> def square(x):
            ...     return x ** 2
            >>> x = BinarySearch.search_for_function(16, square)
            >>> print(f"sqrt(16) ≈ {x:.6f}")
            sqrt(16) ≈ 4.000000
        """
        # 1. Starting point
        x0 = 0.0
        try:
            f0 = function(x0)
        except (ValueError, OverflowError) as e:
            raise ValueError(f"Function is not defined at x=0: {e}")

        if abs(f0 - y) < tolerance:
            return x0

        # 2. Detect monotonicity (increasing or decreasing)
        test_step = 1e-3
        try:
            f_test = function(x0 + test_step)
            is_increasing = f_test > f0
        except (ValueError, OverflowError):
            # If can't evaluate, assume decreasing
            is_increasing = False

        # 3. Expand interval in correct direction (BUG FIX: Better overflow handling)
        step = 1.0
        x_best, f_best = x0, f0
        max_expansion_attempts = 50  # Limit expansion to prevent infinite loops

        for attempt in range(max_expansion_attempts):
            x1 = x0 + step if (y > f0) == is_increasing else x0 - step
            
            try:
                f1 = function(x1)
            except (ValueError, OverflowError):
                # Reached domain limit, stop expansion
                break
                
            if not isfinite(f1):
                # Function returned inf/nan, stop expansion
                break
            
            # Update best estimate if closer to target
            if abs(f1 - y) < abs(f_best - y):
                x_best, f_best = x1, f1
            
            # Stop if close enough
            if abs(f1 - y) < tolerance:
                return x1
            
            # BUG FIX: Stop expansion if we're getting further away
            if abs(f1 - y) > abs(f0 - y) * 10:  # Getting much worse
                break
            
            # Continue expanding
            x0, f0 = x1, f1
            step *= 2
            
            # BUG FIX: Limit step size to prevent overflow
            if abs(step) > 1e10:
                break

        # Define interval [a, b] around best value found
        search_width = min(abs(step), 1e10)  # BUG FIX: Cap search width
        a, b = x_best - search_width, x_best + search_width

        # 4. Binary search refinement
        for _ in range(max_iter):
            mid = (a + b) / 2
            
            try:
                fmid = function(mid)
            except (ValueError, OverflowError):
                # If function not defined, shrink interval
                b = mid if mid > a else a
                continue

            if not isfinite(fmid):
                b = mid
                continue

            # Update best estimate
            if abs(fmid - y) < abs(f_best - y):
                x_best, f_best = mid, fmid

            if abs(fmid - y) < tolerance:
                return mid

            # Binary search direction
            if (fmid < y) == is_increasing:
                a = mid
            else:
                b = mid

        return x_best

    def search_for_function_step(
        self,
        y: float,
        function: Callable[[float], float],
        lower_value: float,
        upper_value: float,
        average: float,
        result: float,
    ) -> Tuple[float, float, float, float]:
        """
        Perform one step of binary search for a function.
        
        This method executes a single iteration of binary search, updating the
        search bounds based on the current average and result.
        
        Args:
            y: Target value to find
            function: Function to search
            lower_value: Current lower bound
            upper_value: Current upper bound
            average: Current midpoint
            result: Function value at current midpoint
        
        Returns:
            Tuple of (new_lower, new_upper, new_average, new_result)
        
        Example:
            >>> searcher = BinarySearch()
            >>> def f(x): return x ** 2
            >>> lower, upper, avg, res = searcher.search_for_function_step(
            ...     16, f, 0, 10, 5, 25
            ... )
        """
        average2 = self.average_function(lower_value, upper_value) + average * self.equals_with_no_graduation(
            y, result, 1
        )
        result2 = function(average2)
        
        lower_value2 = (
            average * self.selector_without_graduation_and_with_inclusion(y, 1, result, 0, 0)
            + lower_value * self.selector_without_graduation_and_with_inclusion(result, 1, y, 0, 0)
        )
        
        upper_value2 = (
            upper_value * self.selector_without_graduation_and_with_inclusion(y, 1, result, 0, 0)
            + average * self.selector_without_graduation_and_with_inclusion(result, 1, y, 0, 0)
        )
        
        return lower_value2, upper_value2, average2, result2

    def binary_search_by_step(
        self, steps: int, minimum_limit: float, maximum_limit: float, previous_value: float = 0
    ) -> float:
        """
        Stepped binary search with non-linear progression.
        
        Instead of traditional binary search that always picks the exact midpoint,
        this method uses a rational function (a/(a+1)) to create non-linear
        progression through the search space. This can be useful for certain
        optimization problems where you want to explore more granularly near edges.
        
        Args:
            steps: Current step number in the search
            minimum_limit: Lower bound of search range
            maximum_limit: Upper bound of search range
            previous_value: Reference value from previous search (if configured)
        
        Returns:
            The next search position based on step count and configuration
        
        Note:
            The progression follows: position = min + (max-min) * steps/(steps+1)
            This creates logarithmic approach to the limits rather than linear.
        
        Example:
            >>> searcher = BinarySearch()
            >>> for step in range(5):
            ...     pos = searcher.binary_search_by_step(step, 0, 100)
            ...     print(f"Step {step}: position {pos:.2f}")
            Step 0: position 0.00
            Step 1: position 50.00
            Step 2: position 66.67
            Step 3: position 75.00
            Step 4: position 80.00
        """
        if not self.previous_value_should_be_the_basis_of_binary_search_calculations:
            if self.binary_search_priority_for_smallest_values:
                return minimum_limit + (maximum_limit - minimum_limit) * self.rational_function(steps)
            else:
                return maximum_limit - (maximum_limit - minimum_limit) * self.rational_function(steps)
        else:
            if self.change_behavior_mid_step and not self.binary_search_priority_modified and steps > self.number_of_attempts // 2:
                self.binary_search_priority_for_smallest_values = not self.binary_search_priority_for_smallest_values
                self.binary_search_priority_modified = True

            if self.previous_value_is_the_target:
                if not self.change_behavior_mid_step:
                    if self.binary_search_priority_for_smallest_values:
                        return minimum_limit + (previous_value - minimum_limit) * self.rational_function(steps)
                    else:
                        return maximum_limit - (maximum_limit - previous_value) * self.rational_function(steps)
                else:
                    if self.binary_search_priority_for_smallest_values:
                        return minimum_limit + (previous_value - minimum_limit) * self.rational_function(
                            steps % (self.number_of_attempts // 2 + 1)
                        )
                    else:
                        return maximum_limit - (maximum_limit - previous_value) * self.rational_function(
                            steps % (self.number_of_attempts // 2 + 1)
                        )
            else:
                if not self.change_behavior_mid_step:
                    if self.binary_search_priority_for_smallest_values:
                        return previous_value - (previous_value - minimum_limit) * self.rational_function(steps)
                    else:
                        return previous_value + (maximum_limit - previous_value) * self.rational_function(steps)
                else:
                    if self.binary_search_priority_for_smallest_values:
                        return previous_value - (previous_value - minimum_limit) * self.rational_function(
                            steps % (self.number_of_attempts // 2 + 1)
                        )
                    else:
                        return previous_value + (maximum_limit - previous_value) * self.rational_function(
                            steps % (self.number_of_attempts // 2 + 1)
                        )

    def linear_search_step(
        self, steps: int, minimum_limit: float, maximum_limit: float, previous_value: float = 0
    ) -> float:
        """
        Stepped linear search with arithmetic progression.
        
        Unlike binary_search_by_step which uses non-linear progression, this method
        uses arithmetic progression to move linearly through the search space.
        
        Args:
            steps: Current step number
            minimum_limit: Lower bound
            maximum_limit: Upper bound
            previous_value: Reference value from previous search (if configured)
        
        Returns:
            The next search position using linear progression
        
        Note:
            Progression: position = start + steps * (range / number_of_attempts)
        
        Example:
            >>> searcher = BinarySearch(number_of_attempts=10)
            >>> for step in range(5):
            ...     pos = searcher.linear_search_step(step, 0, 100)
            ...     print(f"Step {step}: position {pos:.2f}")
            Step 0: position 0.00
            Step 1: position 10.00
            Step 2: position 20.00
            Step 3: position 30.00
            Step 4: position 40.00
        """
        if not self.previous_value_should_be_the_basis_of_binary_search_calculations:
            if self.binary_search_priority_for_smallest_values:
                return self.arithmetic_progression_function(
                    minimum_limit, steps, (maximum_limit - minimum_limit) / self.number_of_attempts
                )
            else:
                return self.arithmetic_progression_function(
                    minimum_limit, self.number_of_attempts - steps, (maximum_limit - minimum_limit) / self.number_of_attempts
                )
        else:
            if self.change_behavior_mid_step and steps > self.number_of_attempts // 2 and not self.binary_search_priority_modified:
                self.binary_search_priority_for_smallest_values = not self.binary_search_priority_for_smallest_values
                self.binary_search_priority_modified = True

            if self.previous_value_is_the_target:
                if self.binary_search_priority_for_smallest_values:
                    return self.arithmetic_progression_function(
                        minimum_limit,
                        steps if not self.change_behavior_mid_step else 2 * (steps % (self.number_of_attempts // 2 + 1)),
                        (previous_value - minimum_limit) / self.number_of_attempts,
                    )
                else:
                    return self.arithmetic_progression_function(
                        maximum_limit,
                        steps if not self.change_behavior_mid_step else 2 * (steps % (self.number_of_attempts // 2 + 1)),
                        -(maximum_limit - previous_value) / self.number_of_attempts,
                    )
            else:
                if self.binary_search_priority_for_smallest_values:
                    return self.arithmetic_progression_function(
                        previous_value,
                        steps if not self.change_behavior_mid_step else 2 * (steps % (self.number_of_attempts // 2 + 1)),
                        (minimum_limit - previous_value) / self.number_of_attempts,
                    )
                else:
                    return self.arithmetic_progression_function(
                        previous_value,
                        steps if not self.change_behavior_mid_step else 2 * (steps % (self.number_of_attempts // 2 + 1)),
                        (maximum_limit - previous_value) / self.number_of_attempts,
                    )

    def binary_search_to_find_miniterm_from_dict(
        self, wanted_number: float, array_dict: Dict[Any, float]
    ) -> Tuple[float, Any]:
        """
        Binary search in a dictionary to find the closest miniterm.
        
        Searches through dictionary values to find the value closest to wanted_number,
        and returns both the value and its corresponding key.
        
        Args:
            wanted_number: Target value to search for
            array_dict: Dictionary with values to search through
        
        Returns:
            Tuple of (closest_value, corresponding_key)
        
        Raises:
            ValueError: If array_dict is empty
        
        Example:
            >>> searcher = BinarySearch()
            >>> data = {'a': 1.5, 'b': 3.2, 'c': 5.1, 'd': 7.8}
            >>> value, key = searcher.binary_search_to_find_miniterm_from_dict(4.0, data)
            >>> print(f"Closest: {value} at key '{key}'")
            Closest: 3.2 at key 'b'
        """
        if not array_dict:
            raise ValueError("Cannot search in empty dictionary")
        
        determine_the_most_approximate_value = False
        array_dict_length = len(array_dict) - 1
        factor_binary_search = (
            array_dict_length // 2 if array_dict_length % 2 != 0 else (array_dict_length + 1) // 2
        )
        factor_is_zero = False
        first_iteration = True
        index_middle = array_dict_length
        index_middle_result = index_middle
        lower_limit = 0
        average = lambda a, b: (a + b) / 2
        upper_limit = array_dict_length
        array_list = list(array_dict.values())

        while not determine_the_most_approximate_value:
            if wanted_number == array_list[index_middle]:
                determine_the_most_approximate_value = True

            elif wanted_number > array_list[index_middle]:
                if index_middle < array_dict_length:
                    lower_limit = index_middle_result
                    index_middle_result = average(lower_limit, upper_limit)
                    index_middle = int(index_middle_result)
                else:
                    determine_the_most_approximate_value = True
            else:
                if not first_iteration:
                    upper_limit = index_middle_result
                else:
                    first_iteration = False

                index_middle_result = average(lower_limit, upper_limit)
                index_middle = int(index_middle_result)

            if factor_is_zero:
                determine_the_most_approximate_value = True

            if not determine_the_most_approximate_value and not factor_is_zero:
                factor_binary_search = factor_binary_search / 2
                if int(factor_binary_search) == 0:
                    factor_is_zero = True

        # Find closest value
        if index_middle > 0:
            if abs(wanted_number - array_list[index_middle]) > abs(wanted_number - array_list[index_middle - 1]):
                index_middle -= 1

        # Reverse mapping: value -> key
        groups = {value: key for key, value in array_dict.items()}

        return array_list[index_middle], groups[array_list[index_middle]]

    def reset(self) -> None:
        """
        Reset the search behavior state.
        
        If change_behavior_mid_step is enabled and search priority was modified
        during execution, this method resets it back to the original state.
        
        Call this between searches if you want to reuse the same searcher instance
        with original configuration.
        
        Example:
            >>> searcher = BinarySearch(change_behavior_mid_step=True)
            >>> # ... perform searches ...
            >>> searcher.reset()  # Reset to original priority
        """
        if self.binary_search_priority_modified:
            self.binary_search_priority_modified = False
            self.binary_search_priority_for_smallest_values = not self.binary_search_priority_for_smallest_values
