"""
Linear System Solvers Module.

Provides iterative solvers for linear and nonlinear systems using binary search
optimizations for hyperparameter tuning and root finding.

Classes:
    BinaryGaussSeidel: Gauss-Seidel iterative solver for linear systems (Ax=b)
    NonLinearGaussSeidel: Gauss-Seidel solver for nonlinear systems (F(x)=0)
    HybridNewtonBinary: Hybrid Newton-Raphson with binary search fallback
"""

from math_toolkit.linear_systems.iterative import BinaryGaussSeidel
from math_toolkit.linear_systems.nonlinear import NonLinearGaussSeidel
from math_toolkit.linear_systems.hybrid import HybridNewtonBinary

__all__ = ['BinaryGaussSeidel', 'NonLinearGaussSeidel', 'HybridNewtonBinary']
