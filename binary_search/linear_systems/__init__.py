"""
Binary-Enhanced Gauss-Seidel Linear System Solver

This package provides iterative solvers for linear systems of equations (Ax = b)
enhanced with binary search for optimal convergence acceleration.
"""

from .binary_gauss_seidel import BinaryGaussSeidel, solve_linear_system

__all__ = ['BinaryGaussSeidel', 'solve_linear_system']
