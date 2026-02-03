"""
Combinatorial weight search with binary refinement.

This module implements WeightCombinationSearch, which finds optimal weights
for linear combinations using exhaustive truth table search combined with
binary refinement via Weighted Possibility Number (WPN).

The algorithm tests all 2^N-1 combinations at each refinement level, selects
the best performer, updates weights, and adjusts the search granularity until
convergence or maximum iterations.
"""

import numpy as np
import itertools
import logging
import multiprocessing as mp
from multiprocessing import Pool, Manager, Lock, Event
from typing import Tuple, List, Dict, Optional, Union, TYPE_CHECKING
import warnings

# Optional pandas for DataFrame support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False

# Optional Numba for JIT compilation
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    # Fallback: no-op decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    HAS_NUMBA = False

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# NUMBA-OPTIMIZED CALCULATION FUNCTIONS
# ============================================================================

@njit(cache=True)
def _calculate_result_numba(coefficients, weights, combo, wpn):
    """
    Numba-optimized calculation of result using core formula.
    
    Formula: param[i] * (W[i] if W[i]!=0 else (1 if combo[i] else W[i])) * (WPN if combo[i] else 1)
    
    Args:
        coefficients: Parameter array (must be contiguous)
        weights: Current weight array (must be contiguous)
        combo: Boolean combination array (must be contiguous)
        wpn: Current Weighted Possibility Number
        
    Returns:
        Calculated result value
    """
    result = 0.0
    n = len(coefficients)
    
    for i in range(n):
        # Determine effective weight
        w = weights[i] if weights[i] != 0.0 else (1.0 if combo[i] else weights[i])
        
        # Determine multiplier
        mult = wpn if combo[i] else 1.0
        
        # Accumulate result
        result += coefficients[i] * w * mult
    
    return result


# ============================================================================
# INDEX FILTERING FUNCTIONS (UPDATED FORMULAS v2.0)
# ============================================================================

def _selecionador_absoluto(x, a, b, c, d, j, k, l, m, n, w):
    """
    Core sine-wave selector function for index filtering.
    
    Formula: ((a-m)/2) * (sin(π*n*(((j-w)/(2*c))*(|x-b+c*k| - |x-b-c*(1-k)| + c) 
             + w + π*((l*(2/π) - (1/π))/2))) + 1) + d + m
    
    This function generates smooth transitions from extremes to middle based
    on the step parameter, which reflects proximity to target.
    
    Args:
        x: Input value (typically the step parameter)
        a, b, c, d: Amplitude and position controls
        j, k, l, m, n, w: Wave shaping parameters
        
    Returns:
        Calculated selector value
    """
    import math
    
    # Inner absolute value term
    inner_abs = abs(x - b + c*k) - abs(x - b - c*(1-k)) + c
    
    # Fraction term
    fraction = (j - w) / (2 * c)
    
    # Constant offset term
    l_term = l * (2/math.pi) - (1/math.pi)
    constant_offset = math.pi * (l_term / 2)
    
    # Sine input
    sine_input = math.pi * n * (fraction * inner_abs + w + constant_offset)
    
    # Calculate result
    sine_value = math.sin(sine_input)
    amplitude = (a - m) / 2
    result = amplitude * (sine_value + 1) + d + m
    
    return result


def _positive_numbers(x, a, b, c, d, j, k, l, m, n, w):
    """Alias for _selecionador_absoluto (used in limit calculations)"""
    return _selecionador_absoluto(x, a, b, c, d, j, k, l, m, n, w)


def _calculate_index_filtered(target, result, tq):
    """
    Calculate which coefficient indices to optimize based on proximity to target.
    
    Uses sine-wave-based mathematical formulas to dynamically select indices:
    - Far from target (low %) → extremes (both ends of sorted array)
    - Close to target (high %) → middle (center of sorted array)
    
    Formula version: 2.0 (updated to prevent out-of-range indices)
    
    Args:
        target: Target value to reach
        result: Current result value
        tq: Terms quantity (number of coefficients)
        
    Returns:
        list: Indices to include in optimization (may be empty, may contain negatives)
    """
    import math
    
    # Calculate proximity percentage
    diff_res = abs(target - result)
    percentage = 1 - (diff_res / abs(target)) if target != 0 else 0.0
    
    # Binary selector (protection for extreme divergence)
    selector = math.ceil(0.5 * (abs(percentage) - abs(percentage - 1) + 1))
    step = selector * percentage
    
    # Calculate limits using UPDATED formulas (v2.0)
    c_val = 1 + tq/(tq+1)
    
    # Upper limit: a=tq-1 (prevents out-of-range), m=tq/2
    upper_limit_val = _positive_numbers(step, tq-1, 0, 1, 0, 1, 0, 1, tq/2, 1, 0)
    
    # Lower limit: unchanged
    lower_limit_val = _positive_numbers(step, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)
    
    # Upper subtraction limit: a=tq-1, d=(tq-1)/2
    upper_sub_val = _positive_numbers(step, tq-1, -1, c_val, (tq-1)/2, 1, 0, 1, 0, 1, 0)
    
    # Lower subtraction limit: unchanged
    lower_sub_val = _positive_numbers(step, tq, -1, c_val, -tq/2, 1, 0, 0, 0, 1, 0)
    
    # Generate sequences
    start_add = math.floor(lower_limit_val)
    end_add = math.ceil(upper_limit_val)
    add_numbers = list(range(start_add, end_add + 1))
    
    start_sub = math.ceil(lower_sub_val) + 1
    end_sub = math.floor(upper_sub_val) - 1
    substract_numbers = list(range(start_sub, end_sub + 1)) if start_sub <= end_sub else []
    
    # Remove subtraction from addition
    index_filtered = [x for x in add_numbers if x not in substract_numbers]
    
    return index_filtered


# ============================================================================
# MAIN CLASS
# ============================================================================

class WeightCombinationSearch:
    """
    Find optimal weights using combinatorial search with binary refinement.
    
    Uses truth table to test all 2^N-1 combinations at each refinement level,
    then applies binary search via Weighted Possibility Number (WPN) to 
    converge on best solution.
    
    Algorithm:
    1. Test all 2^N-1 combinations (exclude all-zeros)
    2. Find winner (minimum absolute difference)
    3. Update weights based on winner combo
    4. Adjust WPN based on conditional differences
    5. Repeat until convergence or max iterations
    
    Scalability:
    - Works with ANY number of parameters (2, 3, 6, 10, 20+)
    - Complexity: O(max_iter × 2^N) where N = number of parameters
    - Performance guide:
      * Fast (< 1s): 2-7 parameters (up to 127 combinations/cycle)
      * Good (few seconds): 8-10 parameters (up to 1,023 combinations/cycle)
      * Slow (minutes): 11-15 parameters (up to 32,767 combinations/cycle)
      * Not recommended: 16+ parameters (exponential explosion)
    - For many parameters (>10), consider gradient-based methods instead
      (BinaryRateOptimizer, AdamW)
    
    Example:
        >>> # 3 parameters (fast)
        >>> search = WeightCombinationSearch(tolerance=2, max_iter=50)
        >>> weights = search.find_optimal_weights([15, 47, -12], target=28)
        >>> print(f"Optimal weights: {weights}")
        Optimal weights: [0.5, 0.5, 0.125]
        
        >>> # 6 parameters (still fast)
        >>> weights = search.find_optimal_weights([15, 47, -12, 123, 56, 10], target=28)
        >>> # Works! Returns optimal weights for all 6 parameters
    """
    
    def __init__(self, 
                 tolerance: float = 0,
                 max_iter: int = 32,
                 initial_wpn: float = 1.0,
                 wpn_bounds: Tuple[float, float] = (-np.inf, np.inf),
                 verbose: bool = False,
                 early_stopping: bool = True,
                 adaptive_sampling: bool = True,
                 sampling_threshold: int = 12,
                 sample_size: int = 10000,
                 sampling_strategy: str = 'importance',
                 use_numba: bool = True,
                 parallel: bool = True,
                 n_jobs: int = -1,
                 index_filtering: bool = False):
        """
        Initialize WeightCombinationSearch.
        
        Args:
            tolerance: Stop when |result - target| ≤ tolerance. Default: 0
            max_iter: Maximum number of cycles. Default: 32
            initial_wpn: Starting Weighted Possibility Number. Default: 1.0
            wpn_bounds: (min, max) tuple for WPN bounds. Default: (-inf, inf)
            verbose: Print progress information. Default: False
            early_stopping: Enable intra-cycle early stopping when solution found. Default: True
            adaptive_sampling: Enable intelligent sampling for large N (N > sampling_threshold). Default: True
            sampling_threshold: Switch to sampling mode when N > this value. Default: 12
            sample_size: Maximum number of combinations to test per cycle when sampling. Default: 10000
            sampling_strategy: Strategy for sampling ('importance', 'random', 'progressive'). Default: 'importance'
            use_numba: Enable Numba JIT compilation for faster formula calculation. Default: True
            parallel: Enable parallel processing with multiprocessing. Default: True
            n_jobs: Number of parallel workers (-1 = all cores, 1 = no parallelism). Default: -1
            index_filtering: Enable dynamic index filtering (extremes→middle based on proximity). Default: False
            
        Raises:
            ValueError: If tolerance < 0 or max_iter < 1 or initial_wpn == 0
        """
        if tolerance < 0:
            raise ValueError(f"Tolerance must be non-negative, got {tolerance}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")
        if initial_wpn == 0:
            raise ValueError("initial_wpn cannot be zero")
        if wpn_bounds[0] >= wpn_bounds[1]:
            raise ValueError(f"Invalid WPN bounds: {wpn_bounds}")
        if sampling_threshold < 1:
            raise ValueError(f"sampling_threshold must be at least 1, got {sampling_threshold}")
        if sample_size < 1:
            raise ValueError(f"sample_size must be at least 1, got {sample_size}")
        if sampling_strategy not in ['importance', 'random', 'progressive']:
            raise ValueError(f"sampling_strategy must be 'importance', 'random', or 'progressive', got {sampling_strategy}")
        
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.initial_wpn = initial_wpn
        self.wpn_bounds = wpn_bounds
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.adaptive_sampling = adaptive_sampling
        self.sampling_threshold = sampling_threshold
        self.sample_size = sample_size
        self.sampling_strategy = sampling_strategy
        self.use_numba = use_numba and HAS_NUMBA  # Only enable if Numba available
        self.parallel = parallel
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.index_filtering = index_filtering
        
        # History tracking
        self.history = {
            'cycles': [],
            'wpn_evolution': [],
            'best_delta_evolution': [],
            'early_stops': [],  # Track when early stopping occurred
            'filtering_history': []  # Track filtered indices per cycle
        }
    
    def find_optimal_weights(self,
                            coefficients: Union[List[float], np.ndarray],
                            target: float,
                            tolerance: Optional[float] = None,
                            max_iter: Optional[int] = None,
                            return_truth_table: bool = False,
                            save_csv: bool = False,
                            csv_filename: str = 'truth_table.csv') -> Union[np.ndarray, Tuple]:
        """
        Find optimal weights W such that coefficients · W ≈ target.
        
        Args:
            coefficients: Array [A, B, C, ...] of N parameters
            target: Desired result value
            tolerance: Override constructor tolerance for this search. Default: use self.tolerance
            max_iter: Override constructor max_iter for this search. Default: use self.max_iter
            return_truth_table: Return full truth table. Default: False
            save_csv: Save truth table to CSV file. Default: False
            csv_filename: CSV filename if save_csv=True. Default: 'truth_table.csv'
            
        Returns:
            weights: Array [W1, W2, ..., WN] of optimal weights
            truth_table (optional): DataFrame or list of dicts with all tested combinations
            
        Raises:
            ValueError: If coefficients is empty or contains only zeros
            
        Example:
            >>> # Use constructor defaults
            >>> search = WeightCombinationSearch(tolerance=2, max_iter=50)
            >>> weights = search.find_optimal_weights([15, 47, -12], target=28)
            >>> 
            >>> # Override for specific search
            >>> weights = search.find_optimal_weights([10, 20, 30], target=60, 
            ...                                        tolerance=1, max_iter=100)
        """
        # Use method parameters if provided, otherwise use instance defaults
        tolerance = tolerance if tolerance is not None else self.tolerance
        max_iter = max_iter if max_iter is not None else self.max_iter
        
        # Validate overridden parameters
        if tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")
        # Input validation
        coefficients = np.array(coefficients, dtype=float)
        if len(coefficients) == 0:
            raise ValueError("coefficients cannot be empty")
        if np.allclose(coefficients, 0):
            raise ValueError("coefficients cannot be all zeros")
        
        n_params = len(coefficients)
        
        # Index filtering setup (if enabled)
        if self.index_filtering:
            # Sort coefficients (smallest to largest)
            coeffs_sorted = np.sort(coefficients)
            sort_indices = np.argsort(coefficients)  # Mapping: sorted_idx -> original_idx
            inverse_sort_indices = np.argsort(sort_indices)  # Mapping: original_idx -> sorted_idx
            
            # Use sorted coefficients for optimization
            coefficients_working = coeffs_sorted
        else:
            # Use original order
            coefficients_working = coefficients
            sort_indices = None
            inverse_sort_indices = None
        
        if self.verbose:
            logger.info(f"Starting WeightCombinationSearch:")
            logger.info(f"  Parameters: {n_params}")
            logger.info(f"  Target: {target}")
            logger.info(f"  Tolerance: {tolerance}")
            logger.info(f"  Max iterations: {max_iter}")
            logger.info(f"  Index filtering: {'enabled' if self.index_filtering else 'disabled'}")
            if self.index_filtering:
                logger.info(f"  Sorted coefficients: {coefficients_working}")
        
        # Initialize
        W = np.zeros(n_params)
        WPN = self.initial_wpn
        truth_table = []
        
        # Main search loop
        for cycle in range(max_iter):
            if self.verbose:
                logger.info(f"\nCycle {cycle + 1}/{max_iter} (WPN={WPN:.4f}, W={W})")
            
            # Calculate current result for filtering decision
            if self.index_filtering:
                # Calculate result with all current weights
                all_ones_combo = np.ones(n_params, dtype=bool)
                current_result = self._calculate_result(coefficients_working, W, all_ones_combo, WPN)
                
                # Calculate which indices to optimize this cycle
                index_filtered = _calculate_index_filtered(target, current_result, n_params)
                
                # Filter to valid indices only (0 to n_params-1)
                index_filtered = [idx for idx in index_filtered if 0 <= idx < n_params]
                
                # Store in history
                self.history['filtering_history'].append({
                    'cycle': cycle + 1,
                    'result': current_result,
                    'percentage': 1 - abs(target - current_result)/abs(target) if target != 0 else 0,
                    'index_filtered': index_filtered.copy(),
                    'num_filtered': len(index_filtered)
                })
                
                if self.verbose:
                    percentage = 1 - abs(target - current_result)/abs(target) if target != 0 else 0
                    logger.info(f"  📊 Current result: {current_result:.4f} ({percentage*100:.2f}% accurate)")
                    logger.info(f"  🔍 Index filtering: {len(index_filtered)} out of {n_params} indices")
                    if len(index_filtered) <= 10:
                        logger.info(f"      Filtered indices: {index_filtered}")
                
                # Determine n_params for combination generation
                n_params_filtered = len(index_filtered) if len(index_filtered) > 0 else n_params
            else:
                # No filtering, use all parameters
                index_filtered = None
                n_params_filtered = n_params
            
            # Generate combinations (exhaustive or sampled)
            # If filtering enabled, generate for filtered space
            if self.adaptive_sampling and n_params_filtered > self.sampling_threshold:
                # ADAPTIVE SAMPLING MODE for large N
                combos = self._generate_sampled_combinations(n_params, coefficients_working if not index_filtered else coefficients_working[index_filtered], self.sample_size, self.sampling_strategy, index_filtered)
                if self.verbose:
                    logger.info(f"  🎯 Adaptive sampling: testing {len(combos):,} of {2**n_params_filtered-1:,} combinations")
            else:
                # EXHAUSTIVE MODE for small N
                combos = self._generate_combinations(n_params, index_filtered)
            
            # Track best winner during iteration (Selection Sort pattern)
            best_winner = None
            best_delta = float('inf')
            early_stopped = False
            early_stop_line = None
            
            # For truth table and WPN adjustment, we still need all results
            # But we track the winner incrementally
            cycle_results = []
            
            for line_num, combo in enumerate(combos):
                # Calculate result using core formula
                result = self._calculate_result(coefficients_working, W, combo, WPN)
                
                # Calculate differences
                delta_abs = abs(result - target)
                delta_cond = result - target
                
                combo_data = {
                    'combo': combo,
                    'result': result,
                    'delta_abs': delta_abs,
                    'delta_cond': delta_cond,
                    'line_num': line_num
                }
                
                # Store for WPN adjustment and truth table
                cycle_results.append(combo_data)
                
                # Record in truth table
                truth_table.append({
                    'cycle': cycle + 1,
                    'line': line_num + 1,
                    'combo': tuple(combo),
                    'weights_before': W.copy(),
                    'wpn': WPN,
                    'result': result,
                    'delta_abs': delta_abs,
                    'delta_cond': delta_cond,
                    'is_winner': False
                })
                
                # SELECTION SORT PATTERN: Track best winner during iteration
                if delta_abs < best_delta or (delta_abs == best_delta and result > best_winner['result'] if best_winner else False):
                    # New best found (or tie-break: prefer result > target)
                    best_delta = delta_abs
                    best_winner = combo_data
                
                if self.verbose and line_num < 5:  # Show first few lines
                    logger.info(f"  Line {line_num+1}: combo={combo}, result={result:.4f}, Δ={delta_abs:.4f}")
                    if delta_abs == best_delta:
                        logger.info(f"    ✓ New best!")
                
                # INTRA-CYCLE EARLY STOPPING
                if self.early_stopping and delta_abs <= tolerance:
                    early_stopped = True
                    early_stop_line = line_num + 1
                    
                    # Ensure this is the winner (it has delta <= tolerance)
                    best_winner = combo_data
                    best_delta = delta_abs
                    
                    if self.verbose:
                        logger.info(f"  ⚡ EARLY STOP at line {line_num+1}/{len(combos)}: Δ={delta_abs:.4f} ≤ {tolerance}")
                        logger.info(f"  Skipped {len(combos) - line_num - 1:,} remaining combinations")
                    
                    break  # Exit combo loop immediately
            
            # Winner is already tracked! No need to search again
            winner = best_winner
            
            # Mark winner in truth table
            if winner:
                winner_line_num = winner['line_num']
                winner_index = len(truth_table) - len(cycle_results) + winner_line_num
                truth_table[winner_index]['is_winner'] = True
                
                if self.verbose and not early_stopped:
                    logger.info(f"  Winner: Line {winner_line_num+1}, Δ={winner['delta_abs']:.4f}")
                
                # Track early stop in history
                if early_stopped:
                    self.history['early_stops'].append({
                        'cycle': cycle + 1,
                        'line': early_stop_line,
                        'total_combos': len(combos),
                        'tested_combos': early_stop_line,
                        'skipped_combos': len(combos) - early_stop_line
                    })
            
            # Track history BEFORE any return (bug fix: was after convergence check)
            self.history['cycles'].append(cycle + 1)
            self.history['wpn_evolution'].append(WPN)
            self.history['best_delta_evolution'].append(winner['delta_abs'])
            
            # Check convergence
            if winner['delta_abs'] <= tolerance:
                if self.verbose:
                    logger.info(f"\n✅ Converged! Δ={winner['delta_abs']:.4f} ≤ {tolerance}")
                
                # Update weights one final time
                for i in range(n_params):
                    if winner['combo'][i]:
                        if W[i] == 0:
                            W[i] = 1
                        W[i] *= WPN
                
                # Save truth table if requested
                if save_csv:
                    self._save_truth_table_csv(truth_table, csv_filename)
                
                if return_truth_table:
                    df = self._format_truth_table(truth_table)
                    return W, df
                
                return W
            
            # Update weights based on winner (if not converged)
            for i in range(n_params):
                if winner['combo'][i]:
                    if W[i] == 0:
                        # First time selected, initialize to 1
                        W[i] = 1
                    # Now multiply by WPN
                    W[i] *= WPN
            
            if self.verbose:
                logger.info(f"  Updated weights: W={W}")
            
            # Adjust WPN for next cycle
            old_wpn = WPN
            WPN = self._adjust_wpn(cycle_results, WPN)
            
            if self.verbose:
                logger.info(f"  WPN: {old_wpn:.4f} → {WPN:.4f}")
        
        # Max iterations reached
        if self.verbose:
            logger.warning(f"\n⚠️  Max iterations ({self.max_iter}) reached without convergence")
            logger.info(f"  Best Δ achieved: {winner['delta_abs']:.4f}")
        
        # Save truth table if requested
        if save_csv:
            self._save_truth_table_csv(truth_table, csv_filename)
        
        if return_truth_table:
            df = self._format_truth_table(truth_table)
            return W, df
        
        return W
    
    def _generate_combinations(self, n_params: int, index_filtered: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Generate all 2^N - 1 combinations (exclude all-zeros).
        
        Args:
            n_params: Number of parameters (total if not filtering, filtered count if filtering)
            index_filtered: If provided, indices to optimize (filtering mode)
            
        Returns:
            List of boolean arrays representing combinations
            If filtering: arrays have full n_params length with only filtered indices varying
        """
        if index_filtered is not None:
            # FILTERING MODE: Generate combinations for filtered indices only
            n_filtered = len(index_filtered)
            if n_filtered == 0:
                # No indices to optimize, return empty
                return []
            
            # Generate all combinations for filtered indices
            filtered_combos = list(itertools.product([False, True], repeat=n_filtered))
            
            # Expand to full parameter space
            combos = []
            for filtered_combo in filtered_combos[1:]:  # Exclude all-zeros
                # Create full combo (all False initially)
                full_combo = np.zeros(n_params, dtype=bool)
                # Set filtered indices according to combo
                for i, idx in enumerate(index_filtered):
                    full_combo[idx] = filtered_combo[i]
                combos.append(full_combo)
            
            return combos
        else:
            # NORMAL MODE: Generate all 2^N combinations
            # Generate all 2^N combinations
            all_combos = list(itertools.product([False, True], repeat=n_params))
            
            # Exclude all-zeros (first combination)
            combos = [np.array(combo) for combo in all_combos[1:]]
            
            return combos
    
    def _generate_sampled_combinations(self, 
                                       n_params: int, 
                                       coefficients: np.ndarray,
                                       sample_size: int,
                                       strategy: str,
                                       index_filtered: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Generate sampled combinations for large N using intelligent sampling.
        
        With index filtering: samples from filtered combination space, removing extremes of range.
        
        Args:
            n_params: Number of parameters (filtered count if filtering)
            coefficients: Parameter values (for importance sampling)
            sample_size: Maximum number of combinations to generate
            strategy: Sampling strategy ('importance', 'random', 'progressive')
            index_filtered: If provided, indices to optimize (filtering mode)
            
        Returns:
            List of boolean arrays representing sampled combinations
        """
        if index_filtered is not None:
            n_filtered = len(index_filtered)
            if n_filtered == 0:
                return []
            
            total_combos = 2**n_filtered - 1  # Exclude all-zeros
        else:
            n_filtered = n_params
            total_combos = 2**n_params - 1
        
        actual_sample_size = min(sample_size, total_combos)
        
        # Remove extremes from sampling range (~20% from each end)
        if index_filtered is not None and total_combos > actual_sample_size:
            # Calculate extreme removal
            remove_fraction = 0.20  # Remove 20% from each end
            lower_bound = int(total_combos * remove_fraction)
            upper_bound = int(total_combos * (1 - remove_fraction))
            
            # Example: 16383 combos -> sample from 3277-13106 (middle 60%)
            sampling_range = (lower_bound, upper_bound)
        else:
            sampling_range = None
        
        if strategy == 'importance':
            return self._importance_sampling(n_params, coefficients, actual_sample_size, index_filtered, sampling_range)
        elif strategy == 'random':
            return self._random_sampling(n_params, actual_sample_size, index_filtered, sampling_range)
        elif strategy == 'progressive':
            return self._progressive_sampling(n_params, coefficients, actual_sample_size, index_filtered, sampling_range)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def _importance_sampling(self, 
                            n_params: int, 
                            coefficients: np.ndarray, 
                            sample_size: int,
                            index_filtered: Optional[List[int]] = None,
                            sampling_range: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Importance sampling: favor combinations with high-magnitude parameters.
        
        Strategy:
        - Parameters with larger absolute values are more likely to be selected
        - Ensures diverse coverage of combination space
        - Includes some pure single-parameter combinations
        - With filtering: only varies filtered indices
        
        Args:
            n_params: Number of parameters (total count)
            coefficients: Parameter values
            sample_size: Number of combinations to generate
            index_filtered: Optional list of indices to optimize (filtering mode)
            sampling_range: Optional (lower, upper) bounds for combo indices (removes extremes)
            
        Returns:
            List of sampled combinations
        """
        combos = []
        
        if index_filtered is not None:
            # FILTERING MODE
            n_filtered = len(index_filtered)
            if n_filtered == 0:
                return []
            
            # Calculate importance for filtered coefficients only
            filtered_coeffs = coefficients[index_filtered]
            importance = np.abs(filtered_coeffs)
            if importance.sum() > 0:
                importance = importance / importance.sum()
            else:
                importance = np.ones(n_filtered) / n_filtered
            
            # Always include single-parameter combinations for filtered indices
            for i in range(n_filtered):
                full_combo = np.zeros(n_params, dtype=bool)
                full_combo[index_filtered[i]] = True
                combos.append(full_combo)
            
            # Generate random combinations weighted by importance
            np.random.seed(42)
            remaining_samples = sample_size - n_filtered
            
            for _ in range(remaining_samples):
                filtered_combo = np.random.rand(n_filtered) < (importance * 2).clip(0, 0.8)
                
                if not filtered_combo.any():
                    filtered_combo[np.argmax(importance)] = True
                
                # Expand to full space
                full_combo = np.zeros(n_params, dtype=bool)
                for i, idx in enumerate(index_filtered):
                    full_combo[idx] = filtered_combo[i]
                
                combos.append(full_combo)
        else:
            # NORMAL MODE
            # Calculate importance weights (use absolute values)
            importance = np.abs(coefficients)
            if importance.sum() > 0:
                importance = importance / importance.sum()  # Normalize to probabilities
            else:
                importance = np.ones(n_params) / n_params  # Uniform if all zeros
            
            # Always include single-parameter combinations (pure selections)
            for i in range(n_params):
                combo = np.zeros(n_params, dtype=bool)
                combo[i] = True
                combos.append(combo)
            
            # Generate random combinations weighted by importance
            np.random.seed(42)  # For reproducibility
            remaining_samples = sample_size - n_params
            
            for _ in range(remaining_samples):
                # Each parameter selected independently with its importance probability
                # Modified: increase selection probability to get more "True" values
                combo = np.random.rand(n_params) < (importance * 2).clip(0, 0.8)
                
                # Ensure at least one True (exclude all-zeros)
                if not combo.any():
                    # Select the most important parameter
                    combo[np.argmax(importance)] = True
                
                combos.append(combo)
        
        return combos
    
    def _random_sampling(self, n_params: int, sample_size: int,
                        index_filtered: Optional[List[int]] = None,
                        sampling_range: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Uniform random sampling of combination space.
        
        Args:
            n_params: Number of parameters
            sample_size: Number of combinations to generate
            index_filtered: Optional list of indices to optimize (filtering mode)
            sampling_range: Optional (lower, upper) bounds (not used in random, kept for interface)
            
        Returns:
            List of sampled combinations
        """
        combos = []
        np.random.seed(42)
        
        if index_filtered is not None:
            n_filtered = len(index_filtered)
            if n_filtered == 0:
                return []
            
            for _ in range(sample_size):
                filtered_combo = np.random.rand(n_filtered) < 0.5
                if not filtered_combo.any():
                    filtered_combo[np.random.randint(0, n_filtered)] = True
                
                full_combo = np.zeros(n_params, dtype=bool)
                for i, idx in enumerate(index_filtered):
                    full_combo[idx] = filtered_combo[i]
                combos.append(full_combo)
        else:
            for _ in range(sample_size):
                combo = np.random.rand(n_params) < 0.5
                if not combo.any():
                    combo[np.random.randint(0, n_params)] = True
                combos.append(combo)
        
        return combos
    
    def _progressive_sampling(self, 
                             n_params: int, 
                             coefficients: np.ndarray, 
                             sample_size: int,
                             index_filtered: Optional[List[int]] = None,
                             sampling_range: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Progressive sampling: multi-stage refinement strategy.
        
        Stage 1: Random sample to explore space
        Stage 2: Importance-weighted sample around promising regions
        
        Args:
            n_params: Number of parameters
            coefficients: Parameter values
            sample_size: Number of combinations to generate
            index_filtered: Optional list of indices to optimize (filtering mode)
            sampling_range: Optional (lower, upper) bounds
            
        Returns:
            List of sampled combinations
        """
        combos = []
        
        # Stage 1: 40% random exploration
        stage1_size = int(sample_size * 0.4)
        combos.extend(self._random_sampling(n_params, stage1_size, index_filtered, sampling_range))
        
        # Stage 2: 60% importance-weighted exploitation
        stage2_size = sample_size - stage1_size
        combos.extend(self._importance_sampling(n_params, coefficients, stage2_size, index_filtered, sampling_range))
        
        return combos
    
    def _calculate_result(self, 
                         coefficients: np.ndarray,
                         weights: np.ndarray,
                         combo: np.ndarray,
                         wpn: float) -> float:
        """
        Calculate result using core formula.
        
        Formula: param[i] * (W[i] if W[i]!=0 else (1 if combo[i] else W[i])) * (WPN if combo[i] else 1)
        
        Uses Numba JIT if available for ~10-50x speedup.
        
        Args:
            coefficients: Parameter array
            weights: Current weight array
            combo: Boolean combination array
            wpn: Current Weighted Possibility Number
            
        Returns:
            Calculated result
        """
        if self.use_numba:
            # Use Numba JIT-compiled version (fast path)
            # Ensure arrays are contiguous for Numba
            coefficients = np.ascontiguousarray(coefficients)
            weights = np.ascontiguousarray(weights)
            combo = np.ascontiguousarray(combo, dtype=np.bool_)
            return _calculate_result_numba(coefficients, weights, combo, wpn)
        else:
            # Pure Python fallback
            result = 0.0
            
            for i in range(len(coefficients)):
                # Weight multiplier
                if weights[i] != 0:
                    weight_mult = weights[i]
                else:
                    weight_mult = 1 if combo[i] else weights[i]  # 1 if selected, 0 if not
                
                # WPN multiplier
                wpn_mult = wpn if combo[i] else 1
                
                # Calculate contribution
                result += coefficients[i] * weight_mult * wpn_mult
            
            return result
    
    def _find_winner(self, cycle_results: List[Dict]) -> Dict:
        """
        Find winner with tie breaking (result > target preferred).
        
        Args:
            cycle_results: List of result dicts from current cycle
            
        Returns:
            Winner dict with minimum absolute difference
        """
        # Find minimum absolute difference
        min_delta = min(r['delta_abs'] for r in cycle_results)
        
        # Get all winners with this delta
        winners = [r for r in cycle_results if r['delta_abs'] == min_delta]
        
        if len(winners) == 1:
            return winners[0]
        
        # Tie breaking: choose result > target (higher result)
        winner = max(winners, key=lambda x: x['result'])
        
        return winner
    
    def _adjust_wpn(self, cycle_results: List[Dict], current_wpn: float) -> float:
        """
        Adjust WPN based on conditional differences.
        
        Rules:
        - All Δ_cond < 0 (all too low) → WPN *= 2 (increase)
        - All Δ_cond > 0 (all too high) OR mixed → WPN /= 2 (decrease)
        
        Args:
            cycle_results: List of result dicts from current cycle
            current_wpn: Current WPN value
            
        Returns:
            New WPN value (within bounds)
        """
        all_negative = all(r['delta_cond'] < 0 for r in cycle_results)
        all_positive = all(r['delta_cond'] > 0 for r in cycle_results)
        
        if all_negative:
            # All results too low, increase WPN
            new_wpn = current_wpn * 2
        else:
            # All positive or mixed, decrease WPN
            new_wpn = current_wpn / 2
        
        # Apply bounds
        new_wpn = np.clip(new_wpn, self.wpn_bounds[0], self.wpn_bounds[1])
        
        return new_wpn
    
    def _format_truth_table(self, truth_table: List[Dict]) -> Union[List[Dict], 'pd.DataFrame']:
        """
        Format truth table as DataFrame (if pandas available) or return as list.
        
        Args:
            truth_table: List of truth table entries
            
        Returns:
            DataFrame if pandas available, otherwise list of dicts
        """
        if HAS_PANDAS:
            df = pd.DataFrame(truth_table)
            # Format combo as string for better readability
            df['combo'] = df['combo'].apply(lambda x: str(x))
            return df
        else:
            return truth_table
    
    def _save_truth_table_csv(self, truth_table: List[Dict], filename: str):
        """
        Save truth table to CSV file.
        
        Args:
            truth_table: List of truth table entries
            filename: Output CSV filename
        """
        if HAS_PANDAS:
            df = self._format_truth_table(truth_table)
            df.to_csv(filename, index=False)
            if self.verbose:
                logger.info(f"Truth table saved to {filename}")
        else:
            import csv
            
            if len(truth_table) == 0:
                return
            
            keys = truth_table[0].keys()
            
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                
                for row in truth_table:
                    # Convert numpy arrays to tuples for CSV writing
                    row_copy = row.copy()
                    if isinstance(row_copy['weights_before'], np.ndarray):
                        row_copy['weights_before'] = tuple(row_copy['weights_before'])
                    writer.writerow(row_copy)
            
            if self.verbose:
                logger.info(f"Truth table saved to {filename}")
    
    def set_tolerance(self, tolerance: float) -> None:
        """
        Update the default tolerance value.
        
        Args:
            tolerance: New tolerance value (must be non-negative)
            
        Raises:
            ValueError: If tolerance < 0
            
        Example:
            >>> search = WeightCombinationSearch(tolerance=2)
            >>> search.set_tolerance(1.0)  # Change default to 1.0
            >>> weights = search.find_optimal_weights([15, 47, -12], target=28)  # Uses 1.0
        """
        if tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")
        self.tolerance = tolerance
        if self.verbose:
            logger.info(f"Tolerance updated to {tolerance}")
    
    def set_max_iter(self, max_iter: int) -> None:
        """
        Update the default maximum iterations value.
        
        Args:
            max_iter: New maximum iterations (must be at least 1)
            
        Raises:
            ValueError: If max_iter < 1
            
        Example:
            >>> search = WeightCombinationSearch(max_iter=50)
            >>> search.set_max_iter(100)  # Change default to 100
            >>> weights = search.find_optimal_weights([15, 47, -12], target=28)  # Uses 100
        """
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")
        self.max_iter = max_iter
        if self.verbose:
            logger.info(f"Max iterations updated to {max_iter}")
