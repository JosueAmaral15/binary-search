"""
Machine Learning Optimizers using Binary Search concepts.

This module provides gradient-based optimizers that leverage binary search
strategies for adaptive learning rate selection and parameter updates.

Classes:
    BinaryRateOptimizer: Gradient Descent with Binary Search Learning Rate (BR-GD)
    AdamW: Adaptive Moment Estimation with Weight Decay and Binary Search
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from math import isfinite


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
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        expansion_factor: float = 2.0,
        binary_search_steps: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the Binary Rate Optimizer.

        Args:
            max_iter: Maximum number of gradient descent iterations
            tol: Tolerance threshold for convergence
            expansion_factor: Multiplier used during alpha expansion phase
            binary_search_steps: Number of binary subdivisions to refine alpha
            verbose: If True, print optimization progress
        """
        self.max_iter = max_iter
        self.tol = tol
        self.expansion_factor = expansion_factor
        self.binary_search_steps = binary_search_steps
        self.verbose = verbose
        
        # History for plotting/debugging
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
        cost_func: Callable
    ) -> float:
        """Calculate projected cost if we take a step of size alpha."""
        theta_temp = current_theta - alpha * gradient
        return cost_func(theta_temp, X, y)

    def _find_optimal_learning_rate(
        self,
        current_theta: np.ndarray,
        gradient: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        cost_func: Callable
    ) -> float:
        """
        Execute BR-GD strategy: Expansion + Binary Search.
        
        Returns:
            float: The optimized learning rate (alpha) for the current step
        """
        base_loss = cost_func(current_theta, X, y)
        
        # PHASE 1: Expansion (find interval [0, alpha_high])
        alpha_low = 0.0
        alpha_high = 1e-4  # Conservative start
        
        best_alpha = 0.0
        best_loss = base_loss
        
        expanded = False
        for _ in range(20):  # Limit expansion attempts
            loss_new = self._get_loss_at_step(alpha_high, current_theta, gradient, X, y, cost_func)
            
            if loss_new < best_loss:
                # Still descending, keep expanding
                best_loss = loss_new
                best_alpha = alpha_high
                alpha_low = alpha_high
                alpha_high *= self.expansion_factor
            else:
                # Error increased, passed the minimum
                expanded = True
                break
        
        if not expanded:
            return alpha_high
        
        # PHASE 2: Binary Refinement
        for _ in range(self.binary_search_steps):
            alpha_mid = (alpha_low + alpha_high) / 2
            loss_mid = self._get_loss_at_step(alpha_mid, current_theta, gradient, X, y, cost_func)
            
            if loss_mid < best_loss:
                best_loss = loss_mid
                best_alpha = alpha_mid
            
            # Local slope test (pseudo-gradient on alpha)
            loss_right = self._get_loss_at_step(alpha_mid * 1.05, current_theta, gradient, X, y, cost_func)
            
            if loss_right < loss_mid:
                alpha_low = alpha_mid  # Valley is to the right
            else:
                alpha_high = alpha_mid  # Valley is to the left
                
        return best_alpha

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_theta: np.ndarray,
        cost_func: Callable,
        grad_func: Callable
    ) -> np.ndarray:
        """
        Execute the main optimization loop.

        Args:
            X: Input features
            y: Target values
            initial_theta: Initial parameter values
            cost_func: Cost function callable(theta, X, y) -> float
            grad_func: Gradient function callable(theta, X, y) -> np.ndarray

        Returns:
            np.ndarray: The final optimized parameters (theta)
        """
        theta = initial_theta.copy()
        
        # Save initial state
        initial_cost = cost_func(theta, X, y)
        self.history["theta"].append(theta.copy())
        self.history["cost"].append(initial_cost)
        self.history["alpha"].append(0.0)

        if self.verbose:
            print(f"--- Starting BR-GD Optimization ---")
            print(f"Initial Cost: {initial_cost:.6f}")

        for i in range(self.max_iter):
            # 1. Calculate Gradient
            grad = grad_func(theta, X, y)
            
            # Safety check for zero gradient
            if np.all(np.abs(grad) < 1e-9):
                if self.verbose:
                    print("Gradient close to zero. Convergence reached.")
                break

            # 2. Find optimal alpha (Binary Search)
            optimal_alpha = self._find_optimal_learning_rate(theta, grad, X, y, cost_func)
            
            # 3. Update weights
            theta_new = theta - optimal_alpha * grad
            new_cost = cost_func(theta_new, X, y)
            
            # Update history
            self.history["theta"].append(theta_new.copy())
            self.history["cost"].append(new_cost)
            self.history["alpha"].append(optimal_alpha)
            
            if self.verbose:
                print(f"Iter {i+1:03d}: Alpha={optimal_alpha:.6f} | Cost={new_cost:.8f}")

            # 4. Stopping criterion (tolerance)
            if abs(self.history["cost"][-2] - new_cost) < self.tol:
                if self.verbose:
                    print(f"Convergence reached by tolerance ({self.tol}) at iter {i+1}.")
                break
                
            theta = theta_new

        return theta


