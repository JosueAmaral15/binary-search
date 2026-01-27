"""
Linear System Solvers Module.

Provides iterative solvers for linear and nonlinear systems using binary search
optimizations for hyperparameter tuning and root finding.

Classes:
    BinaryGaussSeidel: Gauss-Seidel iterative solver for linear systems (Ax=b)
    NonLinearGaussSeidel: Gauss-Seidel solver for nonlinear systems (F(x)=0)
"""

from math_toolkit.linear_systems.iterative import BinaryGaussSeidel
from math_toolkit.linear_systems.nonlinear import NonLinearGaussSeidel

__all__ = ['BinaryGaussSeidel', 'NonLinearGaussSeidel']
