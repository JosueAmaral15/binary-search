"""
Binary Search Module - DEPRECATED - Use math_toolkit instead.

This module is maintained for backward compatibility only.
All functionality has been moved to the math_toolkit package.

DEPRECATED IMPORTS (still work):
    from binary_search import BinarySearch
    from binary_search import BinaryRateOptimizer, AdamW
    
NEW RECOMMENDED IMPORTS:
    from math_toolkit.binary_search import BinarySearch
    from math_toolkit.optimization import BinaryRateOptimizer, AdamW
    from math_toolkit.linear_systems import BinaryGaussSeidel

The package has been reorganized for better cohesion:
- math_toolkit.binary_search: Search algorithms
- math_toolkit.optimization: Gradient-based optimizers
- math_toolkit.linear_systems: Linear system solvers
"""

import warnings

# Show deprecation warning on import
warnings.warn(
    "The 'binary_search' module is deprecated and will be removed in version 3.0. "
    "Please update your imports to use 'math_toolkit' instead:\n"
    "  OLD: from binary_search import BinarySearch\n"
    "  NEW: from math_toolkit.binary_search import BinarySearch\n"
    "See documentation for full migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from math_toolkit for backward compatibility
from math_toolkit import *

__all__ = ['BinarySearch', 'BinaryRateOptimizer', 'AdamW', 'ObserverAdamW', 'BinaryGaussSeidel']
