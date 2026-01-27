"""
Binary-Enhanced Nonlinear Gauss-Seidel Solver

Iterative solver for nonlinear systems F(x) = 0 using Gauss-Seidel approach
with binary search optimization for each coordinate.

This extends the binary search paradigm to nonlinear equation solving:
- Each iteration solves for one variable at a time (Gauss-Seidel style)
- Binary search finds the root of each single-variable equation
- Automatic convergence detection and robust error handling
"""

import numpy as np
import warnings
from typing import Callable, List, Tuple, Optional, Union


class NonLinearResult:
    """Container for nonlinear solver results with metadata"""
    
    def __init__(self, x: np.ndarray, iterations: int, residual: float,
                 relative_change: float, converged: bool, warnings_list: list,
                 function_evals: int):
        self.x = x
        self.iterations = iterations
        self.residual = residual
        self.relative_change = relative_change
        self.converged = converged
        self.warnings = warnings_list
        self.function_evals = function_evals
    
    def __repr__(self):
        status = "CONVERGED" if self.converged else "MAX ITERATIONS"
        return (f"NonLinearResult(status={status}, iterations={self.iterations}, "
                f"residual={self.residual:.2e}, f_evals={self.function_evals})")


class NonLinearGaussSeidel:
    """
    Binary-Enhanced Nonlinear Gauss-Seidel Solver
    
    Solves systems of nonlinear equations F(x) = 0 where:
    - F = [f₁(x₁, x₂, ..., xₙ), f₂(x₁, x₂, ..., xₙ), ..., fₙ(x₁, x₂, ..., xₙ)]
    - Each fᵢ is a scalar function that should equal zero
    
    **Method:**
    Uses Gauss-Seidel iteration with binary search for coordinate-wise root finding.
    In each iteration, solves fᵢ(x₁⁽ᵏ⁺¹⁾, ..., xᵢ₋₁⁽ᵏ⁺¹⁾, xᵢ, xᵢ₊₁⁽ᵏ⁾, ..., xₙ⁽ᵏ⁾) = 0 for xᵢ.
    
    **PARADIGM:** Binary search automatically finds roots without derivatives.
    
    Parameters
    ----------
    functions : List[Callable]
        List of n functions, each taking n arguments and returning a scalar.
        Example: [lambda x, y: x**2 + y - 11, lambda x, y: x + y**2 - 7]
    tolerance : float, default=1e-6
        Convergence tolerance for both residual and coordinate changes
    max_iterations : int, default=100
        Maximum number of Gauss-Seidel iterations
    binary_search_tol : float, default=1e-7
        Tolerance for binary search root finding
    binary_search_max_iter : int, default=50
        Maximum iterations for binary search refinement
    verbose : bool, default=False
        Print iteration progress
    
    Attributes
    ----------
    n : int
        Number of variables/equations
    function_evals : int
        Total number of function evaluations
    
    Examples
    --------
    >>> # Solve: x² + y - 11 = 0, x + y² - 7 = 0
    >>> # Expected solution: x=3, y=2
    >>> f1 = lambda x, y: x**2 + y - 11
    >>> f2 = lambda x, y: x + y**2 - 7
    >>> solver = NonLinearGaussSeidel(functions=[f1, f2])
    >>> result = solver.solve(initial_guess=[0, 0])
    >>> print(f"Solution: x={result.x[0]:.4f}, y={result.x[1]:.4f}")
    
    >>> # Solve 3D system
    >>> f1 = lambda x, y, z: x**2 + y**2 + z**2 - 14
    >>> f2 = lambda x, y, z: x + y + z - 6
    >>> f3 = lambda x, y, z: x*y*z - 6
    >>> solver = NonLinearGaussSeidel(functions=[f1, f2, f3])
    >>> result = solver.solve(initial_guess=[1, 1, 1])
    """
    
    def __init__(self, 
                 functions: List[Callable],
                 tolerance: float = 1e-6,
                 max_iterations: int = 100,
                 binary_search_tol: float = 1e-7,
                 binary_search_max_iter: int = 50,
                 verbose: bool = False):
        
        # Input validation
        if not functions or len(functions) == 0:
            raise ValueError("Must provide at least one function")
        
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        
        self.functions = functions
        self.n = len(functions)
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.binary_search_tol = binary_search_tol
        self.binary_search_max_iter = binary_search_max_iter
        self.verbose = verbose
        self.function_evals = 0
    
    def _safe_eval(self, func: Callable, *args) -> Tuple[float, bool]:
        """
        Safely evaluate a function, handling exceptions.
        
        Returns
        -------
        value : float
            Function value if successful, np.inf if failed
        success : bool
            True if evaluation succeeded
        """
        try:
            self.function_evals += 1
            val = func(*args)
            if np.isnan(val) or np.isinf(val):
                return np.inf, False
            return float(val), True
        except (ValueError, ZeroDivisionError, OverflowError, RuntimeError):
            return np.inf, False
    
    def _detect_monotonicity(self, func: Callable, x0: float, step: float = 1e-4) -> bool:
        """
        Detect if function is locally increasing at x0.
        
        Returns
        -------
        is_increasing : bool
            True if f(x0 + step) > f(x0)
        """
        f0, success0 = self._safe_eval(func, x0)
        if not success0:
            return True  # Default assumption
        
        f1, success1 = self._safe_eval(func, x0 + step)
        if not success1:
            return True
        
        return f1 > f0
    
    def _expand_interval(self, func: Callable, x0: float, target: float,
                        is_increasing: bool, max_expansions: int = 20) -> Tuple[float, float]:
        """
        Expand interval to bracket the target value.
        
        Returns
        -------
        x_best : float
            Best point found
        f_best : float
            Function value at best point
        """
        step = 1.0
        x_current = x0
        f_current, _ = self._safe_eval(func, x0)
        x_best, f_best = x0, f_current
        
        for _ in range(max_expansions):
            # Determine direction based on monotonicity and target
            if (target > f_current) == is_increasing:
                x_next = x_current + step
            else:
                x_next = x_current - step
            
            f_next, success = self._safe_eval(func, x_next)
            
            if not success:
                break
            
            # Track best point (closest to target)
            if abs(f_next - target) < abs(f_best - target):
                x_best, f_best = x_next, f_next
            
            if abs(f_next - target) < self.binary_search_tol:
                return x_next, f_next
            
            x_current = x_next
            f_current = f_next
            step *= 2
        
        return x_best, f_best
    
    def _binary_search_root(self, func: Callable, x0: float, target: float = 0.0) -> float:
        """
        Find x such that func(x) = target using binary search.
        
        This is the core binary search engine optimized for root finding.
        
        Parameters
        ----------
        func : Callable
            Single-variable function
        x0 : float
            Initial guess
        target : float
            Target value (typically 0 for root finding)
        
        Returns
        -------
        x : float
            Best approximation to root
        """
        # Check if initial guess is already good
        f0, success = self._safe_eval(func, x0)
        
        if not success:
            # Try small perturbation if initial point is invalid
            x0 = 0.01
            f0, success = self._safe_eval(func, x0)
            if not success:
                return x0
        
        if abs(f0 - target) < self.binary_search_tol:
            return x0
        
        # Detect monotonicity
        is_increasing = self._detect_monotonicity(func, x0)
        
        # Expand interval to find good starting point
        x_best, f_best = self._expand_interval(func, x0, target, is_increasing)
        
        if abs(f_best - target) < self.binary_search_tol:
            return x_best
        
        # Set up binary search interval
        search_range = abs(x_best - x0) + 1.0
        a = x_best - search_range
        b = x_best + search_range
        
        # Binary search refinement
        for _ in range(self.binary_search_max_iter):
            mid = (a + b) / 2
            fmid, success = self._safe_eval(func, mid)
            
            if not success:
                # Narrow search to valid region
                b = mid
                continue
            
            # Update best point
            if abs(fmid - target) < abs(f_best - target):
                x_best, f_best = mid, fmid
            
            if abs(fmid - target) < self.binary_search_tol:
                return mid
            
            # Adjust interval based on monotonicity
            if (fmid < target) == is_increasing:
                a = mid
            else:
                b = mid
        
        return x_best
    
    def _compute_residual(self, x: np.ndarray) -> float:
        """
        Compute L2 norm of residual F(x).
        
        Returns
        -------
        residual : float
            ||F(x)||₂
        """
        residuals = []
        for i, func in enumerate(self.functions):
            val, success = self._safe_eval(func, *x)
            if success:
                residuals.append(val)
            else:
                residuals.append(1e10)  # Large value for failed evaluation
        
        return np.linalg.norm(residuals)
    
    def solve(self, initial_guess: Union[List[float], np.ndarray]) -> NonLinearResult:
        """
        Solve the nonlinear system F(x) = 0.
        
        Parameters
        ----------
        initial_guess : array-like
            Initial guess for x (length n)
        
        Returns
        -------
        result : NonLinearResult
            Solution with metadata
        """
        # Input validation
        x = np.array(initial_guess, dtype=float)
        
        if len(x) != self.n:
            raise ValueError(f"initial_guess must have length {self.n}, got {len(x)}")
        
        if not np.all(np.isfinite(x)):
            raise ValueError("initial_guess must not contain NaN or Inf")
        
        self.function_evals = 0
        warnings_list = []
        
        if self.verbose:
            print(f"Starting NonLinear Gauss-Seidel (n={self.n})")
            print(f"Initial residual: {self._compute_residual(x):.2e}")
        
        # Main Gauss-Seidel iteration
        for iteration in range(self.max_iterations):
            x_old = x.copy()
            
            # Update each coordinate
            for i in range(self.n):
                # Create single-variable function for coordinate i
                # Fix all other coordinates at current values
                def coordinate_func(val):
                    temp_x = x.copy()
                    temp_x[i] = val
                    result, _ = self._safe_eval(self.functions[i], *temp_x)
                    return result
                
                # Find root using binary search
                x[i] = self._binary_search_root(coordinate_func, x[i], target=0.0)
            
            # Compute convergence metrics
            relative_change = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-10)
            residual = self._compute_residual(x)
            
            if self.verbose:
                print(f"Iter {iteration+1:3d}: residual={residual:.2e}, "
                      f"rel_change={relative_change:.2e}")
            
            # Check convergence
            if relative_change < self.tolerance and residual < self.tolerance:
                if self.verbose:
                    print(f"✓ Converged in {iteration+1} iterations")
                
                return NonLinearResult(
                    x=x,
                    iterations=iteration + 1,
                    residual=residual,
                    relative_change=relative_change,
                    converged=True,
                    warnings_list=warnings_list,
                    function_evals=self.function_evals
                )
        
        # Max iterations reached
        warnings_list.append("Maximum iterations reached without full convergence")
        
        if self.verbose:
            print(f"⚠ Max iterations ({self.max_iterations}) reached")
        
        return NonLinearResult(
            x=x,
            iterations=self.max_iterations,
            residual=self._compute_residual(x),
            relative_change=relative_change,
            converged=False,
            warnings_list=warnings_list,
            function_evals=self.function_evals
        )


# Example usage and tests
if __name__ == "__main__":
    print("=" * 70)
    print("NonLinear Gauss-Seidel Examples")
    print("=" * 70)
    
    # Example 1: 2D system
    print("\n1. Solving 2D system:")
    print("   x² + y - 11 = 0")
    print("   x + y² - 7 = 0")
    print("   Expected: x≈3, y≈2")
    
    f1 = lambda x, y: x**2 + y - 11
    f2 = lambda x, y: x + y**2 - 7
    
    solver = NonLinearGaussSeidel(functions=[f1, f2], verbose=True)
    result = solver.solve(initial_guess=[0, 0])
    
    print(f"\n✓ Solution: x = {result.x[0]:.6f}, y = {result.x[1]:.6f}")
    print(f"  Residual: {result.residual:.2e}")
    print(f"  Function evaluations: {result.function_evals}")
    
    # Example 2: 3D system
    print("\n" + "=" * 70)
    print("\n2. Solving 3D system:")
    print("   x² + y² + z² - 14 = 0")
    print("   x + y + z - 6 = 0")
    print("   xyz - 6 = 0")
    
    f1 = lambda x, y, z: x**2 + y**2 + z**2 - 14
    f2 = lambda x, y, z: x + y + z - 6
    f3 = lambda x, y, z: x * y * z - 6
    
    solver = NonLinearGaussSeidel(functions=[f1, f2, f3], verbose=False)
    result = solver.solve(initial_guess=[1, 1, 1])
    
    print(f"\n✓ Solution: x={result.x[0]:.4f}, y={result.x[1]:.4f}, z={result.x[2]:.4f}")
    print(f"  Residual: {result.residual:.2e}")
    print(f"  Converged: {result.converged}")
