"""
AdamW Optimizer - Adaptive Moment Estimation with Weight Decay.

This module provides the AdamW optimizer with binary search concepts
for adaptive learning rate selection and hyperparameter tuning.

Classes:
    AdamW: Adaptive Moment Estimation with Weight Decay and Binary Search
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from math import isfinite

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
        verbose: bool = True,
        enable_size_optimization: bool = True,
        max_parameter_count: Optional[int] = None,
        auto_tune_strategy: str = 'adaptive',
        auto_tune_beta1: bool = False,
        auto_tune_beta2: bool = False,
        auto_tune_epsilon: bool = False,
        auto_tune_weight_decay: bool = False,
        beta1_range: Tuple[float, float] = (0.8, 0.99),
        beta2_range: Tuple[float, float] = (0.9, 0.9999),
        epsilon_range: Tuple[float, float] = (1e-10, 1e-6),
        weight_decay_range: Tuple[float, float] = (0.0, 0.1)
    ):
        """
        Initialize AdamW optimizer with hyperparameter auto-tuning.
        
        Args:
            max_iter: Maximum iterations
            beta1: First moment decay rate (momentum) - default if not auto-tuned
            beta2: Second moment decay rate (adaptive learning rate) - default if not auto-tuned
            epsilon: Numerical stability constant - default if not auto-tuned
            weight_decay: Weight decay coefficient - default if not auto-tuned
            tol: Convergence tolerance
            use_binary_search: Enable binary search for learning rate
            base_lr: Base learning rate (fallback if binary search disabled)
            expansion_factor: Alpha expansion multiplier
            binary_search_steps: Number of binary search refinements (default strategy)
            verbose: Print optimization progress
            enable_size_optimization: Enable automatic strategy switching based on parameter count
            max_parameter_count: Maximum parameter count hint (None = auto-detect)
            auto_tune_strategy: 'adaptive' (auto-choose), 'aggressive' (small models), 
                               'conservative' (large models)
            auto_tune_beta1: Enable binary search for beta1 (checkbox) - NEW
            auto_tune_beta2: Enable binary search for beta2 (checkbox) - NEW
            auto_tune_epsilon: Enable binary search for epsilon (checkbox) - NEW
            auto_tune_weight_decay: Enable binary search for weight_decay (checkbox) - NEW
            beta1_range: Search range for beta1 if auto-tuning enabled - NEW
            beta2_range: Search range for beta2 if auto-tuning enabled - NEW
            epsilon_range: Search range for epsilon if auto-tuning enabled - NEW
            weight_decay_range: Search range for weight_decay if auto-tuning enabled - NEW
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
        if auto_tune_strategy not in ['adaptive', 'aggressive', 'conservative']:
            raise ValueError(f"auto_tune_strategy must be 'adaptive', 'aggressive', or 'conservative', got '{auto_tune_strategy}'")
        
        # Validate ranges
        if beta1_range[0] >= beta1_range[1] or not (0 <= beta1_range[0] < 1) or not (0 <= beta1_range[1] < 1):
            raise ValueError(f"beta1_range must be (low, high) with 0 <= low < high < 1, got {beta1_range}")
        if beta2_range[0] >= beta2_range[1] or not (0 <= beta2_range[0] < 1) or not (0 <= beta2_range[1] < 1):
            raise ValueError(f"beta2_range must be (low, high) with 0 <= low < high < 1, got {beta2_range}")
        if epsilon_range[0] >= epsilon_range[1] or epsilon_range[0] <= 0:
            raise ValueError(f"epsilon_range must be (low, high) with 0 < low < high, got {epsilon_range}")
        if weight_decay_range[0] >= weight_decay_range[1] or weight_decay_range[0] < 0:
            raise ValueError(f"weight_decay_range must be (low, high) with 0 <= low < high, got {weight_decay_range}")
            
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
        
        # Size optimization parameters
        self.enable_size_optimization = enable_size_optimization
        self.max_parameter_count = max_parameter_count
        self.auto_tune_strategy = auto_tune_strategy
        
        # NEW: Hyperparameter auto-tuning (checkbox paradigm)
        self.auto_tune_beta1 = auto_tune_beta1
        self.auto_tune_beta2 = auto_tune_beta2
        self.auto_tune_epsilon = auto_tune_epsilon
        self.auto_tune_weight_decay = auto_tune_weight_decay
        self.beta1_range = beta1_range
        self.beta2_range = beta2_range
        self.epsilon_range = epsilon_range
        self.weight_decay_range = weight_decay_range
        
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
    
    def _determine_strategy(self, n_params: int) -> dict:
        """
        Determine optimization strategy based on parameter count.
        
        Similar to BinaryGaussSeidel's size optimization, but tailored for ML models.
        
        Parameters
        ----------
        n_params : int
            Number of model parameters
        
        Returns
        -------
        dict
            Strategy configuration with binary_search_steps and other settings
        """
        if not self.enable_size_optimization:
            return {
                'binary_search_steps': self.binary_search_steps,
                'expansion_factor': self.expansion_factor,
                'strategy_name': 'default'
            }
        
        # Apply strategy selection
        if self.auto_tune_strategy == 'aggressive':
            # Aggressive: More binary search iterations, better for small models
            return {
                'binary_search_steps': 15,  # More thorough search
                'expansion_factor': 1.5,    # Smaller expansion (more careful)
                'strategy_name': 'aggressive'
            }
        elif self.auto_tune_strategy == 'conservative':
            # Conservative: Fewer iterations, better for large models
            return {
                'binary_search_steps': 5,   # Faster, less overhead
                'expansion_factor': 3.0,    # Larger expansion (faster exploration)
                'strategy_name': 'conservative'
            }
        else:  # 'adaptive'
            # Auto-choose based on parameter count
            if n_params <= 20:
                # Small models (few parameters): Use aggressive
                return {
                    'binary_search_steps': 15,
                    'expansion_factor': 1.5,
                    'strategy_name': 'adaptive→aggressive'
                }
            elif n_params <= 100:
                # Medium models: Use default
                return {
                    'binary_search_steps': self.binary_search_steps,
                    'expansion_factor': self.expansion_factor,
                    'strategy_name': 'adaptive→default'
                }
            else:
                # Large models (many parameters): Use conservative
                return {
                    'binary_search_steps': 5,
                    'expansion_factor': 3.0,
                    'strategy_name': 'adaptive→conservative'
                }
    
    def _tune_hyperparameter(
        self,
        param_name: str,
        param_range: Tuple[float, float],
        current_theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        cost_func: Callable,
        grad_func: Callable,
        search_steps: int
    ) -> float:
        """
        Binary search for optimal hyperparameter value.
        
        Tests different hyperparameter values and selects the one that gives
        best cost reduction after a few optimization steps.
        
        Parameters
        ----------
        param_name : str
            Name of hyperparameter ('beta1', 'beta2', 'epsilon', 'weight_decay')
        param_range : Tuple[float, float]
            (min_value, max_value) to search within
        current_theta : np.ndarray
            Starting parameters
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
        cost_func : Callable
            Cost function
        grad_func : Callable
            Gradient function
        search_steps : int
            Number of binary search iterations
        
        Returns
        -------
        float
            Optimal hyperparameter value
        """
        low, high = param_range
        best_value = (low + high) / 2
        best_cost = float('inf')
        
        if self.verbose:
            print(f"\n  Auto-tuning {param_name} in range {param_range}...")
        
        # Binary search for optimal value
        for search_iter in range(search_steps):
            # Test three candidates
            candidates = [
                low,
                (low + high) / 2.0,
                high
            ]
            
            costs = []
            
            for candidate_value in candidates:
                # Temporarily set hyperparameter
                original_value = getattr(self, param_name)
                setattr(self, param_name, candidate_value)
                
                # Reinitialize state for fair comparison
                self._initialize_state(current_theta)
                
                # Run a few iterations to evaluate this hyperparameter
                theta_test = current_theta.copy()
                test_cost = cost_func(theta_test, X, y)
                
                for _ in range(5):  # Quick test: 5 iterations
                    grad = grad_func(theta_test, X, y)
                    m_hat, v_hat = self._compute_adam_direction(grad)
                    
                    # Use base learning rate for testing
                    lr_test = self.base_lr
                    theta_test = theta_test - lr_test * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    theta_test -= lr_test * self.weight_decay * theta_test
                    
                    test_cost = cost_func(theta_test, X, y)
                
                costs.append(test_cost)
                
                # Restore original value
                setattr(self, param_name, original_value)
            
            # Find best candidate
            best_idx = np.argmin(costs)
            best_value = candidates[best_idx]
            best_cost = costs[best_idx]
            
            # Update search range (binary search)
            if best_idx == 0:
                # Best is low, search lower half
                high = (low + high) / 2.0
            elif best_idx == 2:
                # Best is high, search upper half
                low = (low + high) / 2.0
            else:
                # Best is middle, narrow from both sides
                range_size = high - low
                low = best_value - range_size * 0.25
                high = best_value + range_size * 0.25
                # Ensure within original bounds
                low = max(low, param_range[0])
                high = min(high, param_range[1])
        
        if self.verbose:
            print(f"    → Optimal {param_name}: {best_value:.6f} (cost after 5 iters: {best_cost:.6f})")
        
        return best_value
    
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
        
        # Use strategy-determined expansion factor
        expansion_factor = getattr(self, '_current_expansion_factor', self.expansion_factor)
        binary_search_steps = getattr(self, '_current_binary_search_steps', self.binary_search_steps)
        
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
                lr_high *= expansion_factor  # Use strategy-determined factor
            else:
                expanded = True
                break
        
        if not expanded:
            return best_lr
        
        # PHASE 2: Binary refinement (use strategy-determined steps)
        for _ in range(binary_search_steps):
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
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")
        if not isinstance(initial_theta, np.ndarray):
            raise ValueError("initial_theta must be a numpy array")
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must not contain NaN or Inf")
        if not np.all(np.isfinite(initial_theta)):
            raise ValueError("initial_theta must not contain NaN or Inf")
        
        theta = initial_theta.copy()
        self._initialize_state(theta)
        
        # NEW: Determine strategy based on parameter count
        n_params = theta.size
        strategy = self._determine_strategy(n_params)
        binary_search_steps_to_use = strategy['binary_search_steps']
        expansion_factor_to_use = strategy['expansion_factor']
        
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
            print(f"Parameters: {n_params}")
            print(f"Strategy: {strategy['strategy_name']}")
            print(f"Binary search steps: {binary_search_steps_to_use}")
            print(f"Expansion factor: {expansion_factor_to_use}")
        
        # Store strategy config for use in _find_optimal_learning_rate
        self._current_binary_search_steps = binary_search_steps_to_use
        self._current_expansion_factor = expansion_factor_to_use
        
        # NEW: AUTO-TUNE HYPERPARAMETERS (if enabled)
        hyperparams_tuned = []
        
        if self.auto_tune_beta1:
            self.beta1 = self._tune_hyperparameter(
                'beta1', self.beta1_range, theta, X, y, cost_func, grad_func, binary_search_steps_to_use
            )
            hyperparams_tuned.append(f"beta1={self.beta1:.6f}")
        
        if self.auto_tune_beta2:
            self.beta2 = self._tune_hyperparameter(
                'beta2', self.beta2_range, theta, X, y, cost_func, grad_func, binary_search_steps_to_use
            )
            hyperparams_tuned.append(f"beta2={self.beta2:.6f}")
        
        if self.auto_tune_epsilon:
            self.epsilon = self._tune_hyperparameter(
                'epsilon', self.epsilon_range, theta, X, y, cost_func, grad_func, binary_search_steps_to_use
            )
            hyperparams_tuned.append(f"epsilon={self.epsilon:.2e}")
        
        if self.auto_tune_weight_decay:
            self.weight_decay = self._tune_hyperparameter(
                'weight_decay', self.weight_decay_range, theta, X, y, cost_func, grad_func, binary_search_steps_to_use
            )
            hyperparams_tuned.append(f"weight_decay={self.weight_decay:.6f}")
        
        if hyperparams_tuned and self.verbose:
            print(f"\n✅ Hyperparameters auto-tuned: {', '.join(hyperparams_tuned)}\n")
        
        # Reinitialize state after hyperparameter tuning
        self._initialize_state(theta)
        
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
