"""
Binary Rate Optimizer - Gradient Descent with Dynamic Learning Rate

A gradient descent optimizer that uses binary search to find the optimal 
learning rate at each iteration, eliminating the need for manual tuning.
"""

from .optimizer import BinaryRateOptimizer

__version__ = "1.0.0"
__all__ = ["BinaryRateOptimizer"]
