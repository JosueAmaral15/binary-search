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
from typing import Tuple, List, Dict, Optional, Union, TYPE_CHECKING
import warnings

# Optional pandas for DataFrame support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False
    warnings.warn("pandas not installed. Truth table will be returned as dict.", ImportWarning)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


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
                 early_stopping: bool = True):
        """
        Initialize WeightCombinationSearch.
        
        Args:
            tolerance: Stop when |result - target| ≤ tolerance. Default: 0
            max_iter: Maximum number of cycles. Default: 32
            initial_wpn: Starting Weighted Possibility Number. Default: 1.0
            wpn_bounds: (min, max) tuple for WPN bounds. Default: (-inf, inf)
            verbose: Print progress information. Default: False
            early_stopping: Enable intra-cycle early stopping when solution found. Default: True
            
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
        
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.initial_wpn = initial_wpn
        self.wpn_bounds = wpn_bounds
        self.verbose = verbose
        self.early_stopping = early_stopping
        
        # History tracking
        self.history = {
            'cycles': [],
            'wpn_evolution': [],
            'best_delta_evolution': [],
            'early_stops': []  # Track when early stopping occurred
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
        
        if self.verbose:
            logger.info(f"Starting WeightCombinationSearch:")
            logger.info(f"  Parameters: {n_params}")
            logger.info(f"  Target: {target}")
            logger.info(f"  Tolerance: {tolerance}")
            logger.info(f"  Max iterations: {max_iter}")
        
        # Initialize
        W = np.zeros(n_params)
        WPN = self.initial_wpn
        truth_table = []
        
        # Main search loop
        for cycle in range(max_iter):
            if self.verbose:
                logger.info(f"\nCycle {cycle + 1}/{max_iter} (WPN={WPN:.4f}, W={W})")
            
            # Generate all 2^N - 1 combinations (exclude all-zeros)
            cycle_results = []
            combos = self._generate_combinations(n_params)
            
            # Track if we found early stopping solution
            early_stopped = False
            early_stop_line = None
            
            for line_num, combo in enumerate(combos):
                # Calculate result using core formula
                result = self._calculate_result(coefficients, W, combo, WPN)
                
                # Calculate differences
                delta_abs = abs(result - target)
                delta_cond = result - target
                
                combo_data = {
                    'combo': combo,
                    'result': result,
                    'delta_abs': delta_abs,
                    'delta_cond': delta_cond
                }
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
                
                if self.verbose and line_num < 5:  # Show first few lines
                    logger.info(f"  Line {line_num+1}: combo={combo}, result={result:.4f}, Δ={delta_abs:.4f}")
                
                # INTRA-CYCLE EARLY STOPPING
                if self.early_stopping and delta_abs <= tolerance:
                    early_stopped = True
                    early_stop_line = line_num + 1
                    
                    if self.verbose:
                        logger.info(f"  ⚡ EARLY STOP at line {line_num+1}/{len(combos)}: Δ={delta_abs:.4f} ≤ {tolerance}")
                        logger.info(f"  Skipped {len(combos) - line_num - 1:,} remaining combinations")
                    
                    # This combo is the winner
                    winner = combo_data
                    break  # Exit combo loop immediately
            
            # If early stopping occurred, no need to find winner again
            if not early_stopped:
                # Find winner (minimum absolute difference, tie break: result > target)
                winner = self._find_winner(cycle_results)
                
                # Find winner index in cycle_results
                winner_line_num = None
                for idx, result_dict in enumerate(cycle_results):
                    if (result_dict['result'] == winner['result'] and 
                        result_dict['delta_abs'] == winner['delta_abs'] and
                        np.array_equal(result_dict['combo'], winner['combo'])):
                        winner_line_num = idx
                        break
                
                winner_index = len(truth_table) - len(cycle_results) + winner_line_num
                truth_table[winner_index]['is_winner'] = True
                
                if self.verbose:
                    logger.info(f"  Winner: Line {winner_line_num+1}, Δ={winner['delta_abs']:.4f}")
            else:
                # Early stopped: winner already set, just mark it in truth table
                winner_index = len(truth_table) - len(cycle_results) + (early_stop_line - 1)
                truth_table[winner_index]['is_winner'] = True
                
                # Track early stop in history
                self.history['early_stops'].append({
                    'cycle': cycle + 1,
                    'line': early_stop_line,
                    'total_combos': len(combos),
                    'tested_combos': early_stop_line,
                    'skipped_combos': len(combos) - early_stop_line
                })
            
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
            
            # Update weights based on winner
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
            
            # Track history
            self.history['cycles'].append(cycle + 1)
            self.history['wpn_evolution'].append(WPN)
            self.history['best_delta_evolution'].append(winner['delta_abs'])
        
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
    
    def _generate_combinations(self, n_params: int) -> List[np.ndarray]:
        """
        Generate all 2^N - 1 combinations (exclude all-zeros).
        
        Args:
            n_params: Number of parameters
            
        Returns:
            List of boolean arrays representing combinations
        """
        # Generate all 2^N combinations
        all_combos = list(itertools.product([False, True], repeat=n_params))
        
        # Exclude all-zeros (first combination)
        combos = [np.array(combo) for combo in all_combos[1:]]
        
        return combos
    
    def _calculate_result(self, 
                         coefficients: np.ndarray,
                         weights: np.ndarray,
                         combo: np.ndarray,
                         wpn: float) -> float:
        """
        Calculate result using core formula.
        
        Formula: param[i] * (W[i] if W[i]!=0 else (1 if combo[i] else W[i])) * (WPN if combo[i] else 1)
        
        Args:
            coefficients: Parameter array
            weights: Current weight array
            combo: Boolean combination array
            wpn: Current Weighted Possibility Number
            
        Returns:
            Calculated result
        """
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
