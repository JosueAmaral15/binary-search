"""
Observer-based parallel hyperparameter tuning for AdamW optimizer.

This module implements a multiprocessing architecture where each hyperparameter
has its own process that proposes values, and an Observer coordinates testing
all combinations to find the optimal hyperparameters.

Architecture:
- HyperparameterProcess: Each hyperparameter runs in separate process
- Observer: Tests all combinations, ranks by cost, signals processes
- Centralized Context: Shared state for communication
- Truth Table: Records all tested combinations and results
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Manager, Queue, Event
import itertools
import time
import csv
from typing import Dict, List, Tuple, Optional, Any

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .adaptive_optimizer import AdamW


class HyperparameterProcess:
    """
    Process that proposes values for a single hyperparameter using binary search.
    
    Each process maintains a search range and proposes 3 values (low, mid, high)
    at each iteration. The observer tests all combinations and signals which
    value performed best, allowing the process to narrow its range.
    """
    
    def __init__(self, name: str, initial_range: Tuple[float, float],
                 shared_context: Dict, feedback_queue: Queue, stop_event: Event):
        self.name = name
        self.low, self.high = initial_range
        self.shared_context = shared_context
        self.feedback_queue = feedback_queue
        self.stop_event = stop_event
        self.iteration = 0
        
    def propose_values(self) -> List[float]:
        """Propose 3 values for testing: low, mid, high."""
        mid = (self.low + self.high) / 2
        
        # For log-scale parameters (like epsilon), use geometric mean
        if self.name == 'epsilon' and self.low > 0 and self.high > 0:
            mid = np.sqrt(self.low * self.high)
        
        return [self.low, mid, self.high]
    
    def narrow_range(self, best_value: float):
        """Narrow search range around best value."""
        range_width = self.high - self.low
        
        # Converged?
        if range_width < 0.01 * (self.high - self.low):
            return
        
        # Narrow to 50% around best value
        margin = range_width * 0.25
        self.low = max(self.low, best_value - margin)
        self.high = min(self.high, best_value + margin)
    
    def run(self):
        """Main process loop: propose values, wait for feedback, narrow range."""
        while not self.stop_event.is_set():
            # Propose 3 values
            proposals = self.propose_values()
            
            # Store in shared context
            key = f"{self.name}_proposals"
            self.shared_context[key] = proposals
            
            # Mark as ready
            self.shared_context[f"{self.name}_ready"] = True
            
            # Wait for feedback
            try:
                feedback = self.feedback_queue.get(timeout=300)  # 5 min timeout
                
                if feedback is None:  # Stop signal
                    break
                
                best_value = feedback['best_value']
                self.narrow_range(best_value)
                self.iteration += 1
                
            except:
                break


class ObserverAdamW(AdamW):
    """
    AdamW optimizer with observer-based parallel hyperparameter tuning.
    
    Uses multiprocessing to tune hyperparameters in parallel. Each hyperparameter
    runs in its own process, proposing values via binary search. The observer
    tests all combinations and coordinates the search.
    
    Parameters
    ----------
    observer_tuning : bool, default=False
        Enable observer-based parallel tuning.
    
    auto_tune_beta1 : bool, default=False
        Enable parallel tuning for beta1 (momentum).
    
    auto_tune_beta2 : bool, default=False
        Enable parallel tuning for beta2 (RMSprop decay).
    
    auto_tune_epsilon : bool, default=False
        Enable parallel tuning for epsilon (numerical stability).
    
    auto_tune_weight_decay : bool, default=False
        Enable parallel tuning for weight_decay (L2 regularization).
    
    max_observer_iterations : int, default=5
        Maximum number of observer search iterations.
    
    convergence_threshold : float, default=0.001
        Cost improvement threshold for convergence.
    
    range_convergence_threshold : float, default=0.01
        Range width threshold for convergence (fraction of initial range).
    
    print_truth_table : bool, default=False
        Print truth table to console during optimization.
    
    save_truth_table_csv : bool, default=False
        Save truth table to CSV file after optimization.
    
    return_truth_table_df : bool, default=False
        Return truth table as DataFrame (not implemented yet).
    
    store_truth_table : bool, default=True
        Store truth table as instance attribute.
    
    truth_table_filename : str, default='truth_table.csv'
        Filename for CSV output if save_truth_table_csv=True.
    
    Other parameters inherited from AdamW.
    
    Examples
    --------
    >>> from binary_search.observer_tuning import ObserverAdamW
    >>> import numpy as np
    >>> 
    >>> # Simple example
    >>> X = np.random.randn(100, 10)
    >>> y = X @ np.ones(10) + 0.1 * np.random.randn(100)
    >>> 
    >>> def cost(theta, X, y):
    ...     return 0.5 * np.mean((X @ theta - y) ** 2)
    >>> 
    >>> def grad(theta, X, y):
    ...     return X.T @ (X @ theta - y) / len(y)
    >>> 
    >>> optimizer = ObserverAdamW(
    ...     observer_tuning=True,
    ...     auto_tune_beta1=True,
    ...     auto_tune_beta2=True,
    ...     max_iter=50,
    ...     max_observer_iterations=3,
    ...     print_truth_table=True,
    ...     verbose=True
    ... )
    >>> 
    >>> theta_init = np.zeros(10)
    >>> theta = optimizer.optimize(X, y, theta_init, cost, grad)
    >>> 
    >>> # Access truth table
    >>> print(optimizer.truth_table)
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        tol: float = 1e-6,
        base_lr: float = 0.001,
        verbose: bool = False,
        # Observer tuning parameters
        observer_tuning: bool = False,
        auto_tune_beta1: bool = False,
        auto_tune_beta2: bool = False,
        auto_tune_epsilon: bool = False,
        auto_tune_weight_decay: bool = False,
        max_observer_iterations: int = 5,
        convergence_threshold: float = 0.001,
        range_convergence_threshold: float = 0.01,
        # Truth table output options (checkboxes)
        print_truth_table: bool = False,
        save_truth_table_csv: bool = False,
        return_truth_table_df: bool = False,
        store_truth_table: bool = True,
        truth_table_filename: str = 'truth_table.csv',
        # Inherited from AdamW
        beta1_range: Tuple[float, float] = (0.8, 0.99),
        beta2_range: Tuple[float, float] = (0.9, 0.9999),
        epsilon_range: Tuple[float, float] = (1e-10, 1e-6),
        weight_decay_range: Tuple[float, float] = (0.0, 0.1),
        **kwargs  # Pass through other AdamW parameters
    ):
        super().__init__(
            max_iter=max_iter,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            tol=tol,
            base_lr=base_lr,
            verbose=verbose,
            **kwargs
        )
        
        self.base_lr = base_lr
        
        # Observer tuning configuration
        self.observer_tuning = observer_tuning
        self.auto_tune_beta1 = auto_tune_beta1
        self.auto_tune_beta2 = auto_tune_beta2
        self.auto_tune_epsilon = auto_tune_epsilon
        self.auto_tune_weight_decay = auto_tune_weight_decay
        
        # Convergence parameters
        self.max_observer_iterations = max_observer_iterations
        self.convergence_threshold = convergence_threshold
        self.range_convergence_threshold = range_convergence_threshold
        
        # Truth table output options
        self.print_truth_table = print_truth_table
        self.save_truth_table_csv = save_truth_table_csv
        self.return_truth_table_df = return_truth_table_df
        self.store_truth_table_flag = store_truth_table
        self.truth_table_filename = truth_table_filename
        
        # Hyperparameter ranges
        self.beta1_range = beta1_range
        self.beta2_range = beta2_range
        self.epsilon_range = epsilon_range
        self.weight_decay_range = weight_decay_range
        
        # Truth table storage
        self.truth_table = [] if store_truth_table else None
        
    def _get_enabled_hyperparameters(self) -> Dict[str, Tuple[float, float]]:
        """Get dictionary of enabled hyperparameters and their ranges."""
        hyperparams = {}
        
        if self.auto_tune_beta1:
            hyperparams['beta1'] = self.beta1_range
        if self.auto_tune_beta2:
            hyperparams['beta2'] = self.beta2_range
        if self.auto_tune_epsilon:
            hyperparams['epsilon'] = self.epsilon_range
        if self.auto_tune_weight_decay:
            hyperparams['weight_decay'] = self.weight_decay_range
            
        return hyperparams
    
    def _test_combination(self, hyperparams: Dict[str, float], X, y, theta, cost_fn, grad_fn) -> float:
        """Test a single hyperparameter combination and return final cost."""
        # Create temporary optimizer with these hyperparameters
        temp_optimizer = AdamW(
            base_lr=self.base_lr,
            beta1=hyperparams.get('beta1', self.beta1),
            beta2=hyperparams.get('beta2', self.beta2),
            epsilon=hyperparams.get('epsilon', self.epsilon),
            weight_decay=hyperparams.get('weight_decay', self.weight_decay),
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=False  # Silent during testing
        )
        
        # Run optimization
        theta_copy = theta.copy()
        temp_optimizer.optimize(X, y, theta_copy, cost_fn, grad_fn)
        
        # Return final cost
        return cost_fn(theta_copy, X, y)
    
    def _observer_tune_hyperparameters(self, X, y, theta, cost_fn, grad_fn):
        """
        Observer-based parallel hyperparameter tuning.
        
        Coordinates multiple processes, each tuning one hyperparameter.
        Tests all combinations of proposed values and signals best to each process.
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("OBSERVER-BASED PARALLEL HYPERPARAMETER TUNING")
            print("=" * 80)
        
        # Get enabled hyperparameters
        hyperparams_config = self._get_enabled_hyperparameters()
        
        if not hyperparams_config:
            if self.verbose:
                print("No hyperparameters enabled for tuning.")
            return
        
        if self.verbose:
            print(f"\nTuning {len(hyperparams_config)} hyperparameters:")
            for name, range_vals in hyperparams_config.items():
                print(f"  - {name}: range {range_vals}")
        
        # Create shared context and queues
        manager = Manager()
        shared_context = manager.dict()
        feedback_queues = {name: Queue() for name in hyperparams_config.keys()}
        stop_event = Event()
        
        # Create processes for each hyperparameter
        processes = []
        for name, initial_range in hyperparams_config.items():
            hp_process = HyperparameterProcess(
                name=name,
                initial_range=initial_range,
                shared_context=shared_context,
                feedback_queue=feedback_queues[name],
                stop_event=stop_event
            )
            process = Process(target=hp_process.run)
            processes.append((name, process, hp_process))
            process.start()
        
        # Observer iteration loop
        best_cost = float('inf')
        best_combination = None
        prev_best_cost = float('inf')
        converged_count = 0
        
        for iteration in range(self.max_observer_iterations):
            if self.verbose:
                print(f"\n{'=' * 80}")
                print(f"Observer Iteration {iteration + 1}/{self.max_observer_iterations}")
                print(f"{'=' * 80}")
            
            # Wait for all processes to propose values
            time.sleep(0.5)
            
            while not all(shared_context.get(f"{name}_ready", False) for name in hyperparams_config.keys()):
                time.sleep(0.1)
            
            # Collect proposals from all processes
            proposals = {}
            for name in hyperparams_config.keys():
                key = f"{name}_proposals"
                proposals[name] = shared_context.get(key, [])
                shared_context[f"{name}_ready"] = False
            
            if self.verbose:
                print("\nProposed values:")
                for name, values in proposals.items():
                    print(f"  {name}: {values}")
            
            # Generate all combinations
            names = list(proposals.keys())
            values_lists = [proposals[name] for name in names]
            combinations = list(itertools.product(*values_lists))
            
            if self.verbose:
                print(f"\nTesting {len(combinations)} combinations...")
            
            # Test all combinations
            results = []
            for combo in combinations:
                hyperparams_dict = dict(zip(names, combo))
                cost = self._test_combination(hyperparams_dict, X, y, theta, cost_fn, grad_fn)
                
                result = {
                    'iteration': iteration + 1,
                    **hyperparams_dict,
                    'cost': cost
                }
                results.append(result)
                
                # Store in truth table
                if self.store_truth_table_flag:
                    self.truth_table.append(result.copy())
                
                if self.verbose:
                    params_str = ", ".join(f"{k}={v:.6f}" for k, v in hyperparams_dict.items())
                    print(f"  [{params_str}] → cost = {cost:.8f}")
            
            # Find best combination in this iteration
            best_result = min(results, key=lambda x: x['cost'])
            iteration_best_cost = best_result['cost']
            
            if iteration_best_cost < best_cost:
                best_cost = iteration_best_cost
                best_combination = {k: v for k, v in best_result.items() if k not in ['iteration', 'cost']}
            
            if self.verbose:
                print(f"\n✓ Best in iteration: cost = {iteration_best_cost:.8f}")
                for name, value in best_combination.items():
                    print(f"    {name} = {value:.6f}")
            
            # Check convergence
            cost_improvement = abs(prev_best_cost - best_cost)
            if cost_improvement < self.convergence_threshold:
                converged_count += 1
                if converged_count >= 2:
                    if self.verbose:
                        print(f"\n✓ Converged: cost stable for {converged_count} iterations")
                    break
            else:
                converged_count = 0
            
            prev_best_cost = best_cost
            
            # Send feedback to each process
            for name in hyperparams_config.keys():
                feedback = {'best_value': best_combination[name]}
                feedback_queues[name].put(feedback)
        
        # Stop all processes
        stop_event.set()
        for name, process, _ in processes:
            feedback_queues[name].put(None)  # Stop signal
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        
        # Apply best hyperparameters
        if best_combination:
            if self.verbose:
                print(f"\n{'=' * 80}")
                print("FINAL BEST HYPERPARAMETERS:")
                print(f"{'=' * 80}")
                print(f"Cost: {best_cost:.8f}")
                for name, value in best_combination.items():
                    print(f"  {name} = {value:.6f}")
                print(f"{'=' * 80}\n")
            
            # Set hyperparameters
            for name, value in best_combination.items():
                setattr(self, name, value)
        
        # Output truth table
        if self.print_truth_table and self.truth_table:
            self._print_truth_table()
        
        if self.save_truth_table_csv and self.truth_table:
            self._save_truth_table_csv()
    
    def _print_truth_table(self):
        """Print truth table to console."""
        if not self.truth_table:
            return
        
        print("\n" + "=" * 80)
        print("TRUTH TABLE")
        print("=" * 80)
        
        # Get column names
        columns = list(self.truth_table[0].keys())
        
        # Print header
        header = " | ".join(f"{col:>12}" for col in columns)
        print(header)
        print("-" * len(header))
        
        # Print rows
        for row in self.truth_table:
            row_str = " | ".join(
                f"{row[col]:>12.6f}" if isinstance(row[col], float) else f"{row[col]:>12}"
                for col in columns
            )
            print(row_str)
        
        print("=" * 80 + "\n")
    
    def _save_truth_table_csv(self):
        """Save truth table to CSV file."""
        if not self.truth_table:
            return
        
        with open(self.truth_table_filename, 'w', newline='') as f:
            if self.truth_table:
                writer = csv.DictWriter(f, fieldnames=self.truth_table[0].keys())
                writer.writeheader()
                writer.writerows(self.truth_table)
        
        if self.verbose:
            print(f"✓ Truth table saved to {self.truth_table_filename}")
    
    def optimize(self, X, y, theta, cost_fn, grad_fn):
        """
        Optimize with observer-based parallel hyperparameter tuning.
        
        If observer_tuning=True, first tunes hyperparameters in parallel using
        multiprocessing, then runs final optimization with best hyperparameters.
        
        Otherwise, uses standard AdamW optimization.
        """
        # Observer-based tuning if enabled
        if self.observer_tuning:
            self._observer_tune_hyperparameters(X, y, theta, cost_fn, grad_fn)
        
        # Run final optimization with tuned (or default) hyperparameters
        return super().optimize(X, y, theta, cost_fn, grad_fn)
