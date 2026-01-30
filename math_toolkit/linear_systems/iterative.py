"""
Binary-Enhanced Gauss-Seidel Solver

Iterative solver for linear systems (Ax = b) with binary search optimization.
Phase 1: Minimal prototype with core functionality.
"""

import numpy as np
import logging
import warnings
from typing import Optional, Tuple, Dict, Any

# Sparse matrix support (optional dependency)
try:
    import scipy.sparse as sp
    SPARSE_AVAILABLE = True
except ImportError:
    SPARSE_AVAILABLE = False
    sp = None

logger = logging.getLogger(__name__)


class SolverResult:
    """Container for solver results with metadata"""
    
    def __init__(self, x: np.ndarray, iterations: int, residual: float, 
                 relative_change: float, converged: bool, warnings_list: list):
        self.x = x
        self.iterations = iterations
        self.residual = residual
        self.relative_change = relative_change
        self.converged = converged
        self.warnings = warnings_list
    
    def __repr__(self):
        status = "CONVERGED" if self.converged else "MAX ITERATIONS REACHED"
        return (f"SolverResult(status={status}, iterations={self.iterations}, "
                f"residual={self.residual:.2e}, relative_change={self.relative_change:.2e})")


class BinaryGaussSeidel:
    """
    Binary-Enhanced Gauss-Seidel solver for linear systems Ax = b.
    
    **PARADIGM**: Automatic hyperparameter tuning via binary search.
    By default, the solver auto-tunes the relaxation factor (ω) dynamically
    per iteration to optimize convergence speed.
    
    This implements the "Binary-Enhanced" philosophy:
    - Hyperparameters are auto-discovered via binary search
    - No manual tuning required (zero-config)
    - User can override if desired
    
    Parameters
    ----------
    tolerance : float, default=1e-6
        Convergence tolerance for both relative change and residual
    max_iterations : int, default=1000
        Maximum number of iterations before stopping
    check_dominance : bool, default=True
        Check diagonal dominance and warn if not satisfied
    verbose : bool, default=False
        Print iteration progress
    auto_tune_omega : bool, default=True
        Enable binary search auto-tuning of relaxation factor ω.
        **PARADIGM DEFAULT**: ON (algorithms should self-optimize)
    omega_search_iterations : int, default=5
        Binary search depth for omega optimization (reduced for performance)
    
    Attributes
    ----------
    tolerance : float
        Convergence tolerance
    max_iterations : int
        Maximum iterations allowed
    check_dominance : bool
        Whether to validate matrix properties
    verbose : bool
        Verbosity flag
    auto_tune_omega : bool
        Whether to auto-tune omega (paradigm: True by default)
    omega_search_iterations : int
        Binary search iterations for omega
    
    Examples
    --------
    >>> # Default: Binary search auto-tuning ENABLED (paradigm)
    >>> A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=float)
    >>> b = np.array([15, 10, 10], dtype=float)
    >>> solver = BinaryGaussSeidel(tolerance=1e-6)  # Auto-tunes by default
    >>> result = solver.solve(A, b)
    >>> print(f"Solution: {result.x}")
    >>> print(f"Iterations: {result.iterations}")
    
    >>> # Opt-out: Disable auto-tuning if you want classical GS
    >>> solver = BinaryGaussSeidel(auto_tune_omega=False)
    >>> result = solver.solve(A, b)
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 1000,
                 check_dominance: bool = True, verbose: bool = False,
                 auto_tune_omega: bool = True, omega_search_iterations: int = 5,
                 enable_polynomial_regression: bool = True, polynomial_degree: int = 2,
                 enable_size_optimization: bool = True, max_system_size: Optional[int] = None,
                 auto_tune_strategy: str = 'adaptive'):
        """
        Initialize Binary-Enhanced Gauss-Seidel solver.
        
        Parameters
        ----------
        tolerance : float, default=1e-6
            Convergence tolerance
        max_iterations : int, default=1000
            Maximum iterations
        check_dominance : bool, default=True
            Check diagonal dominance
        verbose : bool, default=False
            Print progress
        auto_tune_omega : bool, default=True
            Auto-tune relaxation factor via binary search (PARADIGM: ON by default)
        omega_search_iterations : int, default=5
            Binary search depth (reduced from 8 for performance)
        enable_polynomial_regression : bool, default=True
            Enable polynomial regression support (fit curves to data)
        polynomial_degree : int, default=2
            Default polynomial degree for regression
        enable_size_optimization : bool, default=True
            Enable automatic strategy switching based on system size
        max_system_size : int, optional
            Maximum system size hint (None = auto-detect)
        auto_tune_strategy : str, default='adaptive'
            Tuning strategy: 'adaptive' (auto-choose), 'aggressive' (small systems),
            'conservative' (large systems)
        """
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        if auto_tune_strategy not in ['adaptive', 'aggressive', 'conservative']:
            raise ValueError(f"auto_tune_strategy must be 'adaptive', 'aggressive', or 'conservative', got '{auto_tune_strategy}'")
        if polynomial_degree < 1:
            raise ValueError(f"polynomial_degree must be >= 1, got {polynomial_degree}")
        
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.check_dominance = check_dominance
        self.verbose = verbose
        self.auto_tune_omega = auto_tune_omega  # Paradigm: ON by default
        self.omega_search_iterations = omega_search_iterations
        
        # NEW: Polynomial regression parameters
        self.enable_polynomial_regression = enable_polynomial_regression
        self.polynomial_degree = polynomial_degree
        
        # NEW: Size optimization parameters
        self.enable_size_optimization = enable_size_optimization
        self.max_system_size = max_system_size
        self.auto_tune_strategy = auto_tune_strategy  # Optimized: 5 instead of 8
    
    def _determine_strategy(self, n: int) -> dict:
        """
        Determine optimization strategy based on system size.
        
        Parameters
        ----------
        n : int
            System size (number of variables)
        
        Returns
        -------
        dict
            Strategy configuration with omega_search_iterations and other settings
        """
        if not self.enable_size_optimization:
            return {
                'omega_search_iterations': self.omega_search_iterations,
                'strategy_name': 'default'
            }
        
        # Apply strategy selection
        if self.auto_tune_strategy == 'aggressive':
            # Aggressive: More binary search iterations, better for small systems
            return {
                'omega_search_iterations': 8,  # More thorough search
                'strategy_name': 'aggressive'
            }
        elif self.auto_tune_strategy == 'conservative':
            # Conservative: Fewer iterations, better for large systems
            return {
                'omega_search_iterations': 3,  # Faster, less overhead
                'strategy_name': 'conservative'
            }
        else:  # 'adaptive'
            # Auto-choose based on system size
            if n <= 10:
                # Small systems: Use aggressive
                return {
                    'omega_search_iterations': 8,
                    'strategy_name': 'adaptive→aggressive'
                }
            elif n <= 50:
                # Medium systems: Use default
                return {
                    'omega_search_iterations': self.omega_search_iterations,
                    'strategy_name': 'adaptive→default'
                }
            else:
                # Large systems: Use conservative
                return {
                    'omega_search_iterations': 3,
                    'strategy_name': 'adaptive→conservative'
                }
    
    def solve(self, A=None, b=None, 
              x0: Optional[np.ndarray] = None,
              x_data=None, y_data=None, degree: Optional[int] = None,
              optimization_priority: str = 'size_first',
              enable_features: Optional[list] = None) -> SolverResult:
        """
        Solve linear system Ax=b OR fit polynomial to data.
        
        **Backward Compatible**: Old code `solve(A, b)` still works.
        **New Features**: Polynomial regression, size optimization, priority control.
        
        Parameters
        ----------
        A : np.ndarray, shape (n, n), optional
            Coefficient matrix (for direct linear system)
        b : np.ndarray, shape (n,), optional
            Right-hand side vector (for direct linear system)
        x0 : np.ndarray, shape (n,), optional
            Initial guess. If None, uses zero vector.
        x_data : array-like, optional
            X data points for polynomial regression
        y_data : array-like, optional
            Y data points for polynomial regression
        degree : int, optional
            Polynomial degree (overrides self.polynomial_degree)
        optimization_priority : str, default='size_first'
            Priority order: 'size_first', 'polynomial_first', 'both', 'none'
        enable_features : list, optional
            Feature list to enable (None = use defaults from __init__)
        
        Returns
        -------
        SolverResult
            Object containing solution x and convergence metadata
        
        Raises
        ------
        ValueError
            If inputs are invalid or dimensions don't match
        
        Examples
        --------
        >>> # Mode 1: Direct linear system (BACKWARD COMPATIBLE)
        >>> A = np.array([[4, -1], [-1, 4]], dtype=float)
        >>> b = np.array([5, 5], dtype=float)
        >>> result = solver.solve(A, b)
        
        >>> # Mode 2: Polynomial regression (NEW)
        >>> x_data = np.array([1, 2, 3, 4, 5])
        >>> y_data = np.array([2.1, 4.9, 9.2, 15.8, 24.5])
        >>> result = solver.solve(x_data=x_data, y_data=y_data, degree=2)
        
        >>> # Mode 3: Size optimization with priority (NEW)
        >>> result = solver.solve(A, b, optimization_priority='size_first')
        """
        # Route to appropriate handler
        if x_data is not None and y_data is not None:
            # Polynomial regression mode
            if not self.enable_polynomial_regression:
                raise ValueError("Polynomial regression disabled. Set enable_polynomial_regression=True")
            return self._solve_polynomial_regression(x_data, y_data, degree, x0, optimization_priority)
        elif A is not None and b is not None:
            # Direct linear system mode (backward compatible)
            return self._solve_linear_system(A, b, x0, optimization_priority, enable_features)
        else:
            raise ValueError("Must provide either (A, b) for linear system or (x_data, y_data) for polynomial regression")
    
    def _solve_linear_system(self, A: np.ndarray, b: np.ndarray, 
                            x0: Optional[np.ndarray],
                            optimization_priority: str,
                            enable_features: Optional[list]) -> SolverResult:
        """
        Internal method: Solve linear system with size optimization.
        
        This is the core solver that was previously in solve().
        Now refactored to support new features while maintaining compatibility.
        Automatically detects and optimizes for sparse matrices.
        """
        # Detect sparse matrices and route to optimized solver
        if SPARSE_AVAILABLE and sp.issparse(A):
            logger.info(f"Sparse matrix detected: {type(A).__name__}, using optimized sparse solver")
            return self._solve_sparse(A, b, x0, optimization_priority)
        
        # Convert to dense numpy arrays for dense solver
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        
        # Validate inputs
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square matrix, got shape {A.shape}")
        
        n = A.shape[0]
        
        # NEW: Apply size-based strategy selection
        strategy = self._determine_strategy(n)
        omega_search_iters = strategy['omega_search_iterations']
        
        if self.verbose:
            logger.info(f"System size: {n}×{n}")
            logger.info(f"Strategy: {strategy['strategy_name']}")
            logger.info(f"Omega search iterations: {omega_search_iters}")
        
        if b.shape != (n,):
            raise ValueError(f"b must have shape ({n},), got {b.shape}")
        
        # Check for zero diagonal elements
        if np.any(np.abs(np.diag(A)) < 1e-14):
            raise ValueError("Matrix has zero or near-zero diagonal elements. "
                           "Gauss-Seidel cannot proceed.")
        
        # Initialize solution vector
        if x0 is None:
            x = np.zeros(n, dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (n,):
                raise ValueError(f"x0 must have shape ({n},), got {x0.shape}")
            x = x0.copy()
        
        # Check matrix properties and warn if needed
        warnings_list = []
        if self.check_dominance:
            if not self._is_diagonally_dominant(A):
                msg = ("Matrix is not strictly diagonally dominant. "
                      "Convergence is not guaranteed.")
                warnings.warn(msg, UserWarning)
                warnings_list.append(msg)
        
        # Gauss-Seidel iteration with binary search auto-tuning (PARADIGM: always on by default)
        # NEW: Uses strategy-determined omega_search_iters instead of fixed value
        x_old = x.copy()
        omega_cache = 1.0
        should_use_binary_search = self.auto_tune_omega  # Can be disabled if convergence fast
        
        for iteration in range(self.max_iterations):
            # ADAPTIVE DECISION: Disable binary search if convergence already fast
            if iteration == 3 and self.auto_tune_omega:
                # After 3 iterations, check if convergence is fast
                # If converging in <8 iterations predicted, classical GS is sufficient
                relative_change = self._compute_relative_change(x, x_old)
                if relative_change < 0.1:  # Converging fast
                    should_use_binary_search = False  # Disable binary search (overhead not worth it)
            
            # Binary search for optimal omega (SMART: only when needed)
            if should_use_binary_search:
                # OPTIMIZATION: Only search first 3 iterations, then reuse
                if iteration < 3:
                    omega_cache = self._find_adaptive_omega_for_iteration(
                        A, b, x, x_old, n, iteration, search_iterations=omega_search_iters
                    )
                omega = omega_cache
            else:
                omega = 1.0  # Classical Gauss-Seidel
            
            # Store x before this iteration for SOR
            x_iteration_start = x.copy()
            
            # Update each component using latest values
            for i in range(n):
                # sum1: use new values (already computed in this iteration)
                sum1 = np.dot(A[i, :i], x[:i])
                
                # sum2: use old values (not yet computed in this iteration)
                sum2 = np.dot(A[i, i+1:], x[i+1:])
                
                # Compute Gauss-Seidel update
                x_gs = (b[i] - sum1 - sum2) / A[i, i]
                
                # Apply binary search auto-tuned omega (paradigm: always optimizing)
                x[i] = omega * x_gs + (1 - omega) * x_iteration_start[i]
            
            # Check convergence (BOTH criteria per decision 4.C)
            relative_change = self._compute_relative_change(x, x_old)
            residual = self._compute_residual(A, x, b)
            
            if self.verbose:
                logger.info(f"Iteration {iteration + 1}: "
                      f"rel_change={relative_change:.2e}, residual={residual:.2e}")
            
            # Converged if EITHER criterion is satisfied
            if relative_change < self.tolerance or residual < self.tolerance:
                return SolverResult(
                    x=x,
                    iterations=iteration + 1,
                    residual=residual,
                    relative_change=relative_change,
                    converged=True,
                    warnings_list=warnings_list
                )
            
            # Prepare for next iteration
            x_old = x.copy()
        
        # Max iterations reached without convergence
        final_residual = self._compute_residual(A, x, b)
        final_change = self._compute_relative_change(x, x_old)
        
        return SolverResult(
            x=x,
            iterations=self.max_iterations,
            residual=final_residual,
            relative_change=final_change,
            converged=False,
            warnings_list=warnings_list
        )
    
    def _solve_sparse(self, A, b: np.ndarray, 
                     x0: Optional[np.ndarray],
                     optimization_priority: str) -> SolverResult:
        """
        Optimized solver for sparse matrices using CSR format.
        
        Key optimizations for sparse matrices:
        1. Convert to CSR (Compressed Sparse Row) for fast row access
        2. Avoid dense operations (no full matrix @ vector)
        3. Use sparse.dot() for efficient sparse-dense multiplication
        4. Only iterate over non-zero elements
        
        Parameters
        ----------
        A : scipy.sparse matrix
            Sparse coefficient matrix (any scipy.sparse format)
        b : np.ndarray
            Dense right-hand side vector
        x0 : np.ndarray, optional
            Initial guess
        optimization_priority : str
            Optimization priority (inherited from main solve)
        
        Returns
        -------
        SolverResult
            Solution with convergence metadata
        """
        # Convert to CSR format for efficient row-wise operations
        if not sp.isspmatrix_csr(A):
            logger.info(f"Converting {type(A).__name__} to CSR format for optimization")
            A = A.tocsr()
        
        b = np.asarray(b, dtype=float)
        
        # Validate inputs
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square matrix, got shape {A.shape}")
        
        n = A.shape[0]
        
        if b.shape != (n,):
            raise ValueError(f"b must have shape ({n},), got {b.shape}")
        
        # Extract diagonal for checking and iteration
        diagonal = A.diagonal()
        
        if np.any(np.abs(diagonal) < 1e-14):
            raise ValueError("Matrix has zero or near-zero diagonal elements. "
                           "Gauss-Seidel cannot proceed.")
        
        # Initialize solution vector
        if x0 is None:
            x = np.zeros(n, dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (n,):
                raise ValueError(f"x0 must have shape ({n},), got {x0.shape}")
            x = x0.copy()
        
        # Determine strategy based on matrix size
        strategy = self._determine_strategy(n)
        omega_search_iters = strategy['omega_search_iterations']
        
        # Calculate sparsity for logging
        sparsity = 1.0 - (A.nnz / (n * n))
        
        if self.verbose:
            logger.info(f"Sparse system: {n}×{n}, {A.nnz} non-zeros ({sparsity*100:.1f}% sparse)")
            logger.info(f"Strategy: {strategy['strategy_name']}")
            logger.info(f"Omega search iterations: {omega_search_iters}")
        
        # Check diagonal dominance (sparse-aware)
        warnings_list = []
        if self.check_dominance:
            if not self._is_sparse_diagonally_dominant(A, diagonal):
                msg = ("Matrix is not strictly diagonally dominant. "
                      "Convergence is not guaranteed.")
                warnings.warn(msg, UserWarning)
                warnings_list.append(msg)
        
        # Gauss-Seidel iteration with sparse optimization
        x_old = x.copy()
        omega_cache = 1.0
        should_use_binary_search = self.auto_tune_omega
        
        for iteration in range(self.max_iterations):
            # Adaptive decision to disable binary search if convergence fast
            if iteration == 3 and self.auto_tune_omega:
                relative_change = self._compute_relative_change(x, x_old)
                if relative_change < 0.1:
                    should_use_binary_search = False
            
            # Binary search for optimal omega
            if should_use_binary_search:
                if iteration < 3:
                    omega_cache = self._find_adaptive_omega_sparse(
                        A, b, x, x_old, n, diagonal, search_iterations=omega_search_iters
                    )
                omega = omega_cache
            else:
                omega = 1.0
            
            x_iteration_start = x.copy()
            
            # Update each component using sparse row operations
            for i in range(n):
                # Get row i data efficiently from CSR format
                row_start = A.indptr[i]
                row_end = A.indptr[i + 1]
                row_indices = A.indices[row_start:row_end]
                row_data = A.data[row_start:row_end]
                
                # Compute sum using only non-zero elements
                row_sum = 0.0
                for j_idx, col_j in enumerate(row_indices):
                    if col_j != i:
                        row_sum += row_data[j_idx] * x[col_j]
                
                # Gauss-Seidel update
                x_gs = (b[i] - row_sum) / diagonal[i]
                
                # Apply relaxation
                x[i] = omega * x_gs + (1 - omega) * x_iteration_start[i]
            
            # Check convergence
            relative_change = self._compute_relative_change(x, x_old)
            residual = self._compute_sparse_residual(A, x, b)
            
            if self.verbose:
                logger.info(f"Iteration {iteration + 1}: "
                      f"rel_change={relative_change:.2e}, residual={residual:.2e}")
            
            if relative_change < self.tolerance or residual < self.tolerance:
                return SolverResult(
                    x=x,
                    iterations=iteration + 1,
                    residual=residual,
                    relative_change=relative_change,
                    converged=True,
                    warnings_list=warnings_list
                )
            
            x_old = x.copy()
        
        # Max iterations reached
        final_residual = self._compute_sparse_residual(A, x, b)
        final_change = self._compute_relative_change(x, x_old)
        
        return SolverResult(
            x=x,
            iterations=self.max_iterations,
            residual=final_residual,
            relative_change=final_change,
            converged=False,
            warnings_list=warnings_list
        )
    
    def _is_sparse_diagonally_dominant(self, A, diagonal: np.ndarray) -> bool:
        """
        Check diagonal dominance for sparse matrix.
        
        Optimized for CSR format - iterates only over non-zero elements.
        """
        n = A.shape[0]
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            row_data = A.data[row_start:row_end]
            row_indices = A.indices[row_start:row_end]
            
            # Sum absolute values of off-diagonal elements
            off_diagonal_sum = 0.0
            for j_idx, col_j in enumerate(row_indices):
                if col_j != i:
                    off_diagonal_sum += np.abs(row_data[j_idx])
            
            if np.abs(diagonal[i]) <= off_diagonal_sum:
                return False
        
        return True
    
    def _compute_sparse_residual(self, A, x: np.ndarray, b: np.ndarray) -> float:
        """
        Compute residual ||Ax - b||₂ for sparse matrix.
        
        Uses sparse matrix-vector multiplication.
        """
        return np.linalg.norm(A.dot(x) - b)
    
    def _find_adaptive_omega_sparse(self, A, b: np.ndarray, x_current: np.ndarray,
                                   x_previous: np.ndarray, n: int, diagonal: np.ndarray,
                                   search_iterations: int = 5) -> float:
        """
        Binary search for optimal omega for sparse matrices.
        
        Similar to dense version but uses sparse operations.
        """
        omega_low = 0.5
        omega_high = 1.95
        best_omega = 1.0
        best_residual = float('inf')
        
        for _ in range(search_iterations):
            omega_mid = (omega_low + omega_high) / 2.0
            
            # Test three omega values
            for omega_test in [omega_low, omega_mid, omega_high]:
                x_test = x_current.copy()
                
                # One Gauss-Seidel sweep with omega_test
                for i in range(n):
                    row_start = A.indptr[i]
                    row_end = A.indptr[i + 1]
                    row_indices = A.indices[row_start:row_end]
                    row_data = A.data[row_start:row_end]
                    
                    row_sum = 0.0
                    for j_idx, col_j in enumerate(row_indices):
                        if col_j != i:
                            row_sum += row_data[j_idx] * x_test[col_j]
                    
                    x_gs = (b[i] - row_sum) / diagonal[i]
                    x_test[i] = omega_test * x_gs + (1 - omega_test) * x_current[i]
                
                # Compute residual
                residual = self._compute_sparse_residual(A, x_test, b)
                
                if residual < best_residual:
                    best_residual = residual
                    best_omega = omega_test
            
            # Narrow search range
            if abs(best_omega - omega_low) < 0.01:
                omega_high = omega_mid
            elif abs(best_omega - omega_high) < 0.01:
                omega_low = omega_mid
            else:
                omega_low = omega_mid - 0.2
                omega_high = omega_mid + 0.2
        
        return best_omega
    
    def _is_diagonally_dominant(self, A: np.ndarray) -> bool:
        """
        Check if matrix is strictly diagonally dominant.
        
        A matrix is strictly diagonally dominant if:
        |A[i,i]| > sum(|A[i,j]| for j != i) for all rows i
        """
        n = A.shape[0]
        for i in range(n):
            diagonal = np.abs(A[i, i])
            off_diagonal_sum = np.sum(np.abs(A[i, :])) - diagonal
            if diagonal <= off_diagonal_sum:
                return False
        return True
    
    def _compute_residual(self, A: np.ndarray, x: np.ndarray, 
                         b: np.ndarray) -> float:
        """
        Compute residual ||Ax - b||₂
        """
        return np.linalg.norm(A @ x - b)
    
    def _compute_relative_change(self, x_new: np.ndarray, 
                                 x_old: np.ndarray) -> float:
        """
        Compute relative change ||x_new - x_old|| / ||x_old||
        
        If ||x_old|| is too small, returns ||x_new - x_old|| to avoid division by zero.
        """
        norm_old = np.linalg.norm(x_old)
        if norm_old < 1e-14:
            return np.linalg.norm(x_new - x_old)
        return np.linalg.norm(x_new - x_old) / norm_old
    
    def _find_adaptive_omega_for_iteration(self, A: np.ndarray, b: np.ndarray,
                                          x_current: np.ndarray, x_previous: np.ndarray,
                                          n: int, iteration: int, search_iterations: int = 5) -> float:
        """
        Binary search for optimal omega FOR THIS SPECIFIC ITERATION.
        
        OPTIMIZED for performance:
        - Reduced search iterations (5 instead of 8)
        - Early stopping if marginal improvement
        - Adaptive ranges based on convergence state
        
        Strategy: Find omega that maximizes residual reduction in one GS step.
        
        Parameters
        ----------
        A : np.ndarray
            Coefficient matrix
        b : np.ndarray
            Right-hand side vector
        x_current : np.ndarray
            Current solution estimate
        x_previous : np.ndarray
            Previous iteration solution
        n : int
            System size
        iteration : int
            Current iteration number
        
        Returns
        -------
        float
            Optimal omega for this iteration
        """
        # Compute current residual
        current_residual = np.linalg.norm(A @ x_current - b)
        
        # Adaptive search range based on iteration and convergence state
        if iteration == 0:
            # First iteration: moderate search (not too aggressive)
            omega_min, omega_max = 0.9, 1.4
        elif current_residual < 1e-3:
            # Near convergence: conservative (avoid overshoot)
            omega_min, omega_max = 0.98, 1.05
        elif current_residual > 1.0:
            # Slow convergence: aggressive search
            omega_min, omega_max = 0.8, 1.6
        else:
            # Mid-convergence: moderate search
            omega_min, omega_max = 0.95, 1.2
        
        best_omega = 1.0
        best_improvement = 0.0
        
        # Binary search with early stopping (OPTIMIZED: uses strategy-determined iterations)
        for search_iter in range(search_iterations):
            # Test three candidates
            candidates = [
                omega_min,
                (omega_min + omega_max) / 2.0,
                omega_max
            ]
            
            improvements = []
            
            for omega in candidates:
                # Simulate one GS iteration with this omega
                x_test = x_current.copy()
                x_start = x_current.copy()
                
                for i in range(n):
                    sum1 = np.dot(A[i, :i], x_test[:i])
                    sum2 = np.dot(A[i, i+1:], x_test[i+1:])
                    x_gs = (b[i] - sum1 - sum2) / A[i, i]
                    x_test[i] = omega * x_gs + (1 - omega) * x_start[i]
                
                # Measure improvement: reduction in residual
                new_residual = np.linalg.norm(A @ x_test - b)
                improvement = current_residual - new_residual
                improvements.append(improvement)
                
                # Track best
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_omega = omega
            
            # EARLY STOPPING: if all candidates very similar, stop searching
            improvement_range = max(improvements) - min(improvements)
            if improvement_range < current_residual * 0.01:  # Less than 1% variation
                break
            
            # Refine search range toward best region
            best_idx = np.argmax(improvements)
            
            if best_idx == 0:  # Left is best
                omega_max = candidates[1]
            elif best_idx == 2:  # Right is best
                omega_min = candidates[1]
            else:  # Middle is best
                omega_min = candidates[0] + (candidates[1] - candidates[0]) / 2
                omega_max = candidates[1] + (candidates[2] - candidates[1]) / 2
        
        # Safety: ensure omega stays in reasonable range
        best_omega = np.clip(best_omega, 0.5, 1.95)
        
        return best_omega
    
    def _find_optimal_omega(self, A: np.ndarray, b: np.ndarray, 
                           x_current: np.ndarray, n: int) -> float:
        """
        DEPRECATED: Fixed omega search (doesn't work well).
        Kept for backward compatibility but not used.
        Use _find_adaptive_omega_for_iteration instead.
        """
        """
        Binary search for optimal relaxation factor ω (SOR parameter).
        
        Searches in range [1.0, 1.95] to find ω that minimizes residual
        after one Gauss-Seidel iteration.
        
        Parameters
        ----------
        A : np.ndarray
            Coefficient matrix
        b : np.ndarray
            Right-hand side vector
        x_current : np.ndarray
            Current solution estimate
        n : int
            System size
        
        Returns
        -------
        float
            Optimal omega value
        """
        omega_min = 1.0   # Classical Gauss-Seidel
        omega_max = 1.95  # Near upper bound for SOR stability
        
        best_omega = 1.0
        best_residual = float('inf')
        
        # Binary search for optimal omega
        for _ in range(self.omega_search_iterations):
            omega_mid = (omega_min + omega_max) / 2.0
            
            # Test three omega values: left, mid, right
            omegas_to_test = [omega_min, omega_mid, omega_max]
            residuals = []
            
            for omega in omegas_to_test:
                # Simulate one GS iteration with this omega
                x_test = x_current.copy()
                x_start = x_current.copy()
                
                for i in range(n):
                    sum1 = np.dot(A[i, :i], x_test[:i])
                    sum2 = np.dot(A[i, i+1:], x_test[i+1:])
                    x_gs = (b[i] - sum1 - sum2) / A[i, i]
                    x_test[i] = omega * x_gs + (1 - omega) * x_start[i]
                
                # Compute residual after this iteration
                residual = np.linalg.norm(A @ x_test - b)
                residuals.append(residual)
                
                # Track best omega
                if residual < best_residual:
                    best_residual = residual
                    best_omega = omega
            
            # Binary search refinement: move towards better region
            if residuals[0] < residuals[2]:  # Left is better
                omega_max = omega_mid
            elif residuals[2] < residuals[0]:  # Right is better
                omega_min = omega_mid
            else:  # Middle is best, refine around it
                omega_min = (omega_min + omega_mid) / 2.0
                omega_max = (omega_mid + omega_max) / 2.0
        
        return best_omega
    
    def _solve_polynomial_regression(self, x_data, y_data, degree: Optional[int],
                                     x0: Optional[np.ndarray],
                                     optimization_priority: str) -> SolverResult:
        """
        Solve polynomial regression by constructing normal equations.
        
        Fits polynomial: y = a₀ + a₁x + a₂x² + ... + aₙxⁿ
        
        Parameters
        ----------
        x_data : array-like
            X data points
        y_data : array-like
            Y data points
        degree : int, optional
            Polynomial degree (uses self.polynomial_degree if None)
        x0 : np.ndarray, optional
            Initial guess for coefficients
        optimization_priority : str
            Priority order
        
        Returns
        -------
        SolverResult
            Solution contains polynomial coefficients [a₀, a₁, a₂, ...]
        """
        x_data = np.asarray(x_data, dtype=float).flatten()
        y_data = np.asarray(y_data, dtype=float).flatten()
        
        if len(x_data) != len(y_data):
            raise ValueError(f"x_data and y_data must have same length, got {len(x_data)} and {len(y_data)}")
        
        if degree is None:
            degree = self.polynomial_degree
        
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        
        m = len(x_data)  # number of data points
        n = degree + 1   # number of coefficients
        
        if m < n:
            raise ValueError(f"Need at least {n} data points for degree-{degree} polynomial, got {m}")
        
        # Construct Vandermonde matrix: A[i,j] = x[i]^j
        A_vandermonde = np.zeros((m, n))
        for j in range(n):
            A_vandermonde[:, j] = x_data ** j
        
        # Normal equations: (A^T A) c = A^T y
        # where c are the polynomial coefficients
        ATA = A_vandermonde.T @ A_vandermonde
        ATy = A_vandermonde.T @ y_data
        
        if self.verbose:
            logger.info(f"Polynomial regression: degree={degree}, data_points={m}")
            logger.info(f"Solving normal equations: {n}×{n} system")
        
        # Solve using Gauss-Seidel
        result = self._solve_linear_system(ATA, ATy, x0, optimization_priority, None)
        
        # Add polynomial-specific info to result
        result.polynomial_degree = degree
        result.num_data_points = m
        
        return result
    
    def fit_polynomial(self, x_data, y_data, degree: Optional[int] = None) -> SolverResult:
        """
        High-level polynomial fitting method.
        
        Convenience method that calls solve() with polynomial parameters.
        
        Parameters
        ----------
        x_data : array-like
            X data points
        y_data : array-like
            Y data points
        degree : int, optional
            Polynomial degree (uses self.polynomial_degree if None)
        
        Returns
        -------
        SolverResult
            Solution contains polynomial coefficients [a₀, a₁, a₂, ...]
        
        Examples
        --------
        >>> solver = BinaryGaussSeidel()
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2.1, 4.9, 9.2, 15.8, 24.5])
        >>> result = solver.fit_polynomial(x, y, degree=2)
        >>> coeffs = result.x  # [a₀, a₁, a₂]
        >>> print(f"Polynomial: y = {coeffs[0]:.2f} + {coeffs[1]:.2f}x + {coeffs[2]:.2f}x²")
        """
        return self.solve(x_data=x_data, y_data=y_data, degree=degree)
    


def solve_linear_system(A: np.ndarray, b: np.ndarray, 
                       tolerance: float = 1e-6,
                       max_iterations: int = 1000,
                       x0: Optional[np.ndarray] = None,
                       verbose: bool = False) -> SolverResult:
    """
    Convenience function to solve Ax = b using Binary-Enhanced Gauss-Seidel.
    
    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Coefficient matrix
    b : np.ndarray, shape (n,)
        Right-hand side vector
    tolerance : float, default=1e-6
        Convergence tolerance
    max_iterations : int, default=1000
        Maximum iterations
    x0 : np.ndarray, optional
        Initial guess
    verbose : bool, default=False
        Print progress
    
    Returns
    -------
    SolverResult
        Solution and convergence information
    
    Examples
    --------
    >>> A = [[4, -1, 0], [-1, 4, -1], [0, -1, 3]]
    >>> b = [15, 10, 10]
    >>> result = solve_linear_system(A, b)
    >>> print(result.x)
    """
    solver = BinaryGaussSeidel(tolerance=tolerance, max_iterations=max_iterations,
                              verbose=verbose)
    return solver.solve(A, b, x0=x0)
