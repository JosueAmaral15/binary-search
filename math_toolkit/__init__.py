"""
Math Toolkit - Comprehensive Mathematical Optimization and Search Library.

This package provides:
1. Binary Search Algorithms:
   - BinarySearch: Advanced search algorithms with tolerance-based comparisons
   
2. Optimization Algorithms:
   - BinaryRateOptimizer: Gradient descent with binary search learning rate
   - AdamW: Adaptive moment estimation with weight decay
   - ObserverAdamW: Parallel hyperparameter tuning with observer pattern

3. Linear System Solvers:
   - BinaryGaussSeidel: Iterative solver with binary search optimizations

Recommended imports:
    from math_toolkit.binary_search import BinarySearch
    from math_toolkit.optimization import BinaryRateOptimizer, AdamW, ObserverAdamW
    from math_toolkit.linear_systems import BinaryGaussSeidel

For backward compatibility, all classes are also available at package level:
    from math_toolkit import BinarySearch, BinaryRateOptimizer, AdamW
"""

# Import from submodules
from math_toolkit.binary_search import BinarySearch
from math_toolkit.optimization import BinaryRateOptimizer, AdamW, ObserverAdamW
from math_toolkit.linear_systems import BinaryGaussSeidel

# Expose all classes at package level
__all__ = [
    'BinarySearch',
    'BinaryRateOptimizer',
    'AdamW',
    'ObserverAdamW',
    'BinaryGaussSeidel',
]

# Version info
__version__ = '2.0.0'
