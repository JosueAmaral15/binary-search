"""
Gradient Descent Optimizer with Binary Search Learning Rate (BR-GD).

This module provides an optimizer that dynamically finds the optimal learning rate
at each iteration using a binary search strategy, eliminating manual tuning.
"""

from typing import Callable, Dict, List
import numpy as np


class BinaryRateOptimizer:
    """
    Gradient Descent Optimizer with Binary Search Learning Rate (BR-GD).
    
    Instead of a fixed learning rate, this optimizer performs a dynamic 
    line search at every step to find the optimal step size that minimizes 
    the cost function along the gradient direction.
    
    The algorithm uses a two-phase approach:
    1. **Expansion Phase**: Exponentially increase alpha until cost starts increasing
    2. **Binary Search Phase**: Refine the optimal alpha within the found bounds
    
    This eliminates the need for manual learning rate tuning and adapts to
    the local topology of the cost function at each iteration.
    
    Attributes:
        max_iter: Maximum number of gradient descent iterations
        tol: Convergence tolerance (stop if cost change < tol)
        expansion_factor: Multiplier for alpha during expansion phase
        binary_search_steps: Number of binary subdivisions for refinement
        history: Dictionary tracking theta, cost, and alpha over iterations
    
    Example:
        >>> import numpy as np
        >>> def cost(theta, X, y):
        ...     return np.mean((X * theta - y) ** 2)
        >>> def grad(theta, X, y):
        ...     return 2 * np.mean((X * theta - y) * X)
        >>> optimizer = BinaryRateOptimizer(max_iter=50, tol=1e-6)
        >>> X = np.array([1, 2, 3, 4])
        >>> y = np.array([2, 4, 6, 8])
        >>> theta_init = np.array([0.0])
        >>> theta_final = optimizer.optimize(X, y, theta_init, cost, grad)
        >>> print(f"Optimal theta: {theta_final[0]:.6f}")
        Optimal theta: 2.000000
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        expansion_factor: float = 2.0,
        binary_search_steps: int = 10,
    ) -> None:
        """
        Initialize the Binary Rate Optimizer.

        Args:
            max_iter: Maximum number of gradient descent iterations.
                Higher values allow more optimization steps but take longer.
                Default: 100
            
            tol: Tolerance threshold for convergence. Optimization stops when
                the absolute change in cost is less than this value.
                Default: 1e-6
            
            expansion_factor: Multiplier used during the alpha expansion phase.
                Larger values explore faster but might overshoot.
                Default: 2.0 (doubles alpha each expansion step)
            
            binary_search_steps: Number of binary subdivisions to refine alpha.
                More steps give better precision but take longer.
                Default: 10 (provides ~0.1% precision)
        
        Raises:
            ValueError: If any parameter is invalid (negative, zero, etc.)
        """
        # Input validation
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        if expansion_factor <= 1.0:
            raise ValueError(f"expansion_factor must be > 1.0, got {expansion_factor}")
        if binary_search_steps <= 0:
            raise ValueError(f"binary_search_steps must be positive, got {binary_search_steps}")
        
        self.max_iter = max_iter
        self.tol = tol
        self.expansion_factor = expansion_factor
        self.binary_search_steps = binary_search_steps
        
        # History for plotting/debugging purposes
        self.history: Dict[str, List] = {
            "theta": [],
            "cost": [],
            "alpha": []
        }

    def _get_loss_at_step(
        self,
        alpha: float,
        current_theta: np.ndarray,
        gradient: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        cost_func: Callable,
    ) -> float:
        """
        Calculate the projected cost if we take a step of size 'alpha'.
        
        This helper function evaluates: cost(theta - alpha * gradient)
        
        Args:
            alpha: Step size to evaluate
            current_theta: Current parameters
            gradient: Current gradient direction
            X: Input features
            y: Target values
            cost_func: Cost function to evaluate
        
        Returns:
            The cost after taking a step of size alpha
        """
        theta_temp = current_theta - alpha * gradient
        return cost_func(theta_temp, X, y)

    def _find_optimal_learning_rate(
        self,
        current_theta: np.ndarray,
        gradient: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        cost_func: Callable,
    ) -> float:
        """
        Execute the BR-GD strategy: Expansion + Binary Search.
        
        This is the core innovation of the algorithm:
        
        Phase 1 - Expansion:
            Start with a small alpha and exponentially increase it until
            the cost starts increasing (we've passed the minimum).
        
        Phase 2 - Binary Search:
            Refine the search within the bounds [alpha_low, alpha_high]
            using binary subdivision to find the optimal alpha.
        
        Args:
            current_theta: Current parameters
            gradient: Current gradient direction  
            X: Input features
            y: Target values
            cost_func: Cost function to minimize
        
        Returns:
            The optimized learning rate (alpha) for the current step.
            Returns 0.0 if gradient is too small or optimization fails.
        """
        # Current cost (equivalent to alpha = 0)
        base_loss = cost_func(current_theta, X, y)
        
        # --- PHASE 1: Expansion (Find the interval [0, alpha_high]) ---
        alpha_low = 0.0
        alpha_high = 1e-4  # Conservative start
        
        best_alpha = 0.0
        best_loss = base_loss
        
        # Attempt to expand alpha until the error worsens (overshooting the valley)
        expanded = False
        for _ in range(20):  # Limit expansion attempts to prevent infinite loops
            loss_new = self._get_loss_at_step(
                alpha_high, current_theta, gradient, X, y, cost_func
            )
            
            if loss_new < best_loss:
                # We are still descending, keep expanding
                best_loss = loss_new
                best_alpha = alpha_high
                alpha_low = alpha_high
                alpha_high *= self.expansion_factor
            else:
                # Error increased, we passed the minimum
                expanded = True
                break
        
        if not expanded:
            # If we expanded 20 times and error kept dropping, 
            # return the largest found (likely approaching a flat region)
            return alpha_high

        # --- PHASE 2: Binary Refinement ---
        # We know the optimal alpha is within [alpha_low, alpha_high]
        
        for _ in range(self.binary_search_steps):
            alpha_mid = (alpha_low + alpha_high) / 2
            loss_mid = self._get_loss_at_step(
                alpha_mid, current_theta, gradient, X, y, cost_func
            )
            
            if loss_mid < best_loss:
                best_loss = loss_mid
                best_alpha = alpha_mid
            
            # Local slope test to decide which side to cut (Pseudo-Gradient on Alpha)
            # Check a point slightly to the right
            loss_right = self._get_loss_at_step(
                alpha_mid * 1.05, current_theta, gradient, X, y, cost_func
            )
            
            if loss_right < loss_mid:
                # The valley is further to the right
                alpha_low = alpha_mid
            else:
                # The valley is to the left (or we passed it)
                alpha_high = alpha_mid
                
        return best_alpha

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_theta: np.ndarray,
        cost_func: Callable,
        grad_func: Callable,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Execute the main optimization loop.
        
        This method iteratively:
        1. Computes the gradient at the current position
        2. Finds the optimal learning rate using binary search
        3. Updates parameters in the gradient direction
        4. Checks for convergence
        
        Args:
            X: Input features array of shape (n_samples,) or (n_samples, n_features)
            y: Target values array of shape (n_samples,)
            initial_theta: Initial parameters array of shape (n_params,)
            cost_func: Function with signature cost_func(theta, X, y) -> float
            grad_func: Function with signature grad_func(theta, X, y) -> np.ndarray
            verbose: If True, print optimization progress. Default: True

        Returns:
            The final optimized parameters (theta) as np.ndarray.
            
        Raises:
            ValueError: If input shapes are incompatible or data contains NaN/Inf
        
        Example:
            >>> def mse(theta, X, y):
            ...     return np.mean((X @ theta - y) ** 2)
            >>> def mse_grad(theta, X, y):
            ...     return 2 * X.T @ (X @ theta - y) / len(y)
            >>> X = np.array([[1], [2], [3], [4]])
            >>> y = np.array([2, 4, 6, 8])
            >>> theta0 = np.array([0.0])
            >>> opt = BinaryRateOptimizer(max_iter=50)
            >>> theta = opt.optimize(X, y, theta0, mse, mse_grad)
        """
        # Input validation
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")
        if not isinstance(initial_theta, np.ndarray):
            raise ValueError("initial_theta must be a numpy array")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must not contain NaN or Inf")
        if not np.all(np.isfinite(initial_theta)):
            raise ValueError("initial_theta must not contain NaN or Inf")
        
        theta = initial_theta.copy()
        
        # Clear history from previous runs
        self.history = {"theta": [], "cost": [], "alpha": []}
        
        # Save initial state
        initial_cost = cost_func(theta, X, y)
        self.history["theta"].append(theta.copy())
        self.history["cost"].append(initial_cost)
        self.history["alpha"].append(0.0)

        if verbose:
            print(f"--- Starting BR-GD Optimization ---")
            print(f"Initial Cost: {initial_cost:.6f}")

        for i in range(self.max_iter):
            # 1. Calculate Gradient
            grad = grad_func(theta, X, y)
            
            # Safety check for zero gradient (perfect convergence already)
            if np.all(np.abs(grad) < 1e-9):
                if verbose:
                    print("Gradient close to zero. Convergence reached.")
                break

            # 2. Find the 'Magic' Alpha (The Core Innovation)
            optimal_alpha = self._find_optimal_learning_rate(
                theta, grad, X, y, cost_func
            )
            
            # 3. Update Weights
            theta_new = theta - optimal_alpha * grad
            new_cost = cost_func(theta_new, X, y)
            
            # Update history
            self.history["theta"].append(theta_new.copy())
            self.history["cost"].append(new_cost)
            self.history["alpha"].append(optimal_alpha)
            
            if verbose:
                print(f"Iter {i+1:03d}: Alpha={optimal_alpha:.6f} | Cost={new_cost:.8f}")

            # 4. Stopping Criterion (Tolerance)
            if abs(self.history["cost"][-2] - new_cost) < self.tol:
                if verbose:
                    print(f"Convergence reached by tolerance ({self.tol}) at iter {i+1}.")
                break
                
            theta = theta_new

        return theta
