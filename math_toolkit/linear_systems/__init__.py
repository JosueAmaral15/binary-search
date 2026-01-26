"""
Linear System Solvers Module.

Provides iterative solvers for linear systems using binary search
optimizations for hyperparameter tuning.

Classes:
    BinaryGaussSeidel: Gauss-Seidel iterative solver with binary search
"""

from math_toolkit.linear_systems.iterative import BinaryGaussSeidel

__all__ = ['BinaryGaussSeidel']
