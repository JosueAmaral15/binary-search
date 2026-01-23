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


class AdamW:
    """
    AdamW Optimizer with Binary Search Learning Rate Adaptation.
    
    Combines the adaptive moment estimation (Adam) algorithm with:
    - Decoupled weight decay (AdamW improvement)
    - Binary search for optimal learning rate per iteration
    - Momentum and adaptive learning rates per parameter
    
    This is a hybrid approach: Adam's adaptive gradients + Binary Search's
    optimal step size selection, providing robust convergence without
    extensive hyperparameter tuning.
    
    Key Features:
    - Adaptive learning rates for each parameter
    - Momentum using exponential moving averages
    - Decoupled weight decay (better than L2 regularization)
    - Binary search refinement for global learning rate
    
    Attributes:
        max_iter: Maximum number of optimization iterations
        beta1: Exponential decay rate for first moment estimates (default: 0.9)
        beta2: Exponential decay rate for second moment estimates (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
        weight_decay: L2 penalty coefficient (default: 0.01)
        tol: Convergence tolerance
        use_binary_search: If True, use binary search for learning rate
        base_lr: Base learning rate (used when binary search disabled)
        
    Example:
        >>> import numpy as np
        >>> def cost(theta, X, y):
        ...     return np.mean((X @ theta - y) ** 2)
        >>> def grad(theta, X, y):
        ...     return 2 * X.T @ (X @ theta - y) / len(y)
        >>> optimizer = AdamW(max_iter=100, beta1=0.9, beta2=0.999)
        >>> X = np.random.randn(100, 5)
        >>> y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1
        >>> theta_init = np.zeros(5)
        >>> theta_final = optimizer.optimize(X, y, theta_init, cost, grad)
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        tol: float = 1e-6,
        use_binary_search: bool = True,
        base_lr: float = 0.001,
        expansion_factor: float = 2.0,
        binary_search_steps: int = 10,
        verbose: bool = True
    ):
        """
        Initialize AdamW optimizer.
        
        Args:
            max_iter: Maximum iterations
            beta1: First moment decay rate (momentum)
            beta2: Second moment decay rate (adaptive learning rate)
            epsilon: Numerical stability constant
            weight_decay: Weight decay coefficient
            tol: Convergence tolerance
            use_binary_search: Enable binary search for learning rate
            base_lr: Base learning rate (fallback if binary search disabled)
            expansion_factor: Alpha expansion multiplier
            binary_search_steps: Number of binary search refinements
            verbose: Print optimization progress
        """
        # Validate parameters
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        if base_lr <= 0:
            raise ValueError(f"base_lr must be positive, got {base_lr}")
            
        self.max_iter = max_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.tol = tol
        self.use_binary_search = use_binary_search
        self.base_lr = base_lr
        self.expansion_factor = expansion_factor
        self.binary_search_steps = binary_search_steps
        self.verbose = verbose
        
        # State variables (initialized in optimize)
        self.m: Optional[np.ndarray] = None  # First moment estimate
        self.v: Optional[np.ndarray] = None  # Second moment estimate
        self.t: int = 0  # Timestep
        
        # History tracking
        self.history: Dict[str, List] = {
            "theta": [],
            "cost": [],
            "lr": [],
            "grad_norm": []
        }
    
    def _initialize_state(self, theta: np.ndarray) -> None:
        """Initialize moment estimates to zeros."""
        self.m = np.zeros_like(theta)
        self.v = np.zeros_like(theta)
        self.t = 0
    
    def _compute_adam_direction(self, gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Adam update direction with bias correction.
        
        Args:
            gradient: Current gradient
            
        Returns:
            Tuple of (bias_corrected_m, bias_corrected_v)
        """
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return m_hat, v_hat
    
    def _get_loss_at_step(
        self,
        lr: float,
        current_theta: np.ndarray,
        m_hat: np.ndarray,
        v_hat: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        cost_func: Callable
    ) -> float:
        """
        Calculate projected cost after Adam update with learning rate lr.
        
        Args:
            lr: Learning rate to test
            current_theta: Current parameters
            m_hat: Bias-corrected first moment
            v_hat: Bias-corrected second moment
            X: Input features
            y: Target values
            cost_func: Cost function
            
        Returns:
            float: Projected cost
        """
        # Adam update with weight decay
        theta_temp = current_theta - lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        theta_temp -= lr * self.weight_decay * current_theta  # Weight decay
        
        return cost_func(theta_temp, X, y)
    
    def _find_optimal_learning_rate(
        self,
        current_theta: np.ndarray,
        m_hat: np.ndarray,
        v_hat: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        cost_func: Callable
    ) -> float:
        """
        Binary search for optimal learning rate given Adam direction.
        
        Args:
            current_theta: Current parameters
            m_hat: Bias-corrected first moment
            v_hat: Bias-corrected second moment
            X: Input features
            y: Target values
            cost_func: Cost function
            
        Returns:
            float: Optimal learning rate
        """
        base_loss = cost_func(current_theta, X, y)
        
        # PHASE 1: Expansion
        lr_low = 0.0
        lr_high = self.base_lr * 0.1  # Start conservatively
        
        best_lr = self.base_lr
        best_loss = base_loss
        
        expanded = False
        for _ in range(15):  # Limit expansion
            loss_new = self._get_loss_at_step(lr_high, current_theta, m_hat, v_hat, X, y, cost_func)
            
            if isfinite(loss_new) and loss_new < best_loss:
                best_loss = loss_new
                best_lr = lr_high
                lr_low = lr_high
                lr_high *= self.expansion_factor
            else:
                expanded = True
                break
        
        if not expanded:
            return best_lr
        
        # PHASE 2: Binary refinement
        for _ in range(self.binary_search_steps):
            lr_mid = (lr_low + lr_high) / 2
            loss_mid = self._get_loss_at_step(lr_mid, current_theta, m_hat, v_hat, X, y, cost_func)
            
            if isfinite(loss_mid) and loss_mid < best_loss:
                best_loss = loss_mid
                best_lr = lr_mid
            
            # Directional test
            loss_right = self._get_loss_at_step(lr_mid * 1.05, current_theta, m_hat, v_hat, X, y, cost_func)
            
            if isfinite(loss_right) and loss_right < loss_mid:
                lr_low = lr_mid
            else:
                lr_high = lr_mid
        
        return best_lr
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_theta: np.ndarray,
        cost_func: Callable,
        grad_func: Callable
    ) -> np.ndarray:
        """
        Execute AdamW optimization with binary search learning rate.
        
        Args:
            X: Input features (shape: [n_samples, n_features])
            y: Target values (shape: [n_samples])
            initial_theta: Initial parameters (shape: [n_features])
            cost_func: Cost function callable(theta, X, y) -> float
            grad_func: Gradient function callable(theta, X, y) -> np.ndarray
            
        Returns:
            np.ndarray: Optimized parameters
        """
        theta = initial_theta.copy()
        self._initialize_state(theta)
        
        # Save initial state
        initial_cost = cost_func(theta, X, y)
        self.history["theta"].append(theta.copy())
        self.history["cost"].append(initial_cost)
        self.history["lr"].append(0.0)
        self.history["grad_norm"].append(0.0)
        
        if self.verbose:
            print("--- Starting AdamW Optimization ---")
            print(f"Initial Cost: {initial_cost:.6f}")
            print(f"Binary Search: {'Enabled' if self.use_binary_search else 'Disabled'}")
        
        for i in range(self.max_iter):
            # 1. Compute gradient
            grad = grad_func(theta, X, y)
            grad_norm = np.linalg.norm(grad)
            
            # Check for convergence
            if grad_norm < 1e-9:
                if self.verbose:
                    print("Gradient norm close to zero. Convergence reached.")
                break
            
            # 2. Compute Adam direction
            m_hat, v_hat = self._compute_adam_direction(grad)
            
            # 3. Find optimal learning rate
            if self.use_binary_search:
                lr = self._find_optimal_learning_rate(theta, m_hat, v_hat, X, y, cost_func)
            else:
                lr = self.base_lr
            
            # 4. Update parameters (Adam + weight decay)
            theta_new = theta - lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            theta_new -= lr * self.weight_decay * theta  # Decoupled weight decay
            
            new_cost = cost_func(theta_new, X, y)
            
            # Update history
            self.history["theta"].append(theta_new.copy())
            self.history["cost"].append(new_cost)
            self.history["lr"].append(lr)
            self.history["grad_norm"].append(grad_norm)
            
            if self.verbose:
                print(f"Iter {i+1:03d}: LR={lr:.6e} | Cost={new_cost:.8f} | Grad={grad_norm:.6e}")
            
            # 5. Check convergence
            if abs(self.history["cost"][-2] - new_cost) < self.tol:
                if self.verbose:
                    print(f"Convergence reached by tolerance ({self.tol}) at iter {i+1}.")
                break
            
            theta = theta_new
        
        return theta
