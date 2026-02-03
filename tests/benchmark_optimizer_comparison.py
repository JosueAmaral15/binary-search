"""
Comprehensive Comparison: WeightCombinationSearch vs BinaryRateOptimizer vs AdamW

Tests all three optimization algorithms on various scenarios and generates
a detailed comparison table with conclusions.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Callable
import pandas as pd

# Import the three optimizers
from math_toolkit.binary_search.combinatorial import WeightCombinationSearch
from math_toolkit.optimization.gradient_descent import BinaryRateOptimizer
from math_toolkit.optimization.adaptive_optimizer import AdamW


class OptimizerBenchmark:
    """Benchmark suite for comparing optimization algorithms."""
    
    def __init__(self):
        self.results = []
        
    def setup_test_scenario(self, scenario_name: str, n_params: int, difficulty: str) -> Tuple:
        """
        Create test scenarios with varying difficulty.
        
        Args:
            scenario_name: Name of the test scenario
            n_params: Number of parameters
            difficulty: 'easy', 'medium', 'hard'
            
        Returns:
            coefficients, target, X, y, cost_func, grad_func
        """
        np.random.seed(42)
        
        if difficulty == 'easy':
            # Linear relationship, well-conditioned
            coeffs = np.random.uniform(1, 10, n_params)
            target = np.sum(coeffs) * 0.7
            X = np.eye(n_params)  # Identity matrix
            y = coeffs * 0.7
            
        elif difficulty == 'medium':
            # Slightly noisy, mixed signs
            coeffs = np.random.uniform(-10, 10, n_params)
            target = np.sum(np.abs(coeffs)) * 0.5
            X = np.random.randn(50, n_params) * 2
            y = X @ coeffs + np.random.randn(50) * 0.1
            
        elif difficulty == 'hard':
            # Highly non-linear, ill-conditioned
            coeffs = np.random.uniform(-100, 100, n_params)
            coeffs[0] = 1000  # Ill-conditioning
            target = np.sum(coeffs ** 2) ** 0.5
            X = np.random.randn(100, n_params) * 10
            X[:, 0] *= 100  # Scale mismatch
            y = X @ coeffs + np.random.randn(100) * 5
        
        # Cost and gradient functions (Mean Squared Error)
        def cost_func(theta, X_data, y_data):
            return np.mean((X_data @ theta - y_data) ** 2)
        
        def grad_func(theta, X_data, y_data):
            return 2 * X_data.T @ (X_data @ theta - y_data) / len(y_data)
        
        return coeffs, target, X, y, cost_func, grad_func
    
    def test_weight_combination_search(
        self,
        coeffs: np.ndarray,
        target: float,
        max_iter: int = 50,
        with_filtering: bool = False
    ) -> Dict:
        """Test WeightCombinationSearch (WCS)."""
        start_time = time.time()
        
        try:
            wcs = WeightCombinationSearch(
                tolerance=abs(target) * 0.01,  # 1% tolerance
                max_iter=max_iter,
                index_filtering=with_filtering,
                verbose=False
            )
            
            weights = wcs.find_optimal_weights(coeffs, target)
            result = np.sum(coeffs * weights)
            elapsed = time.time() - start_time
            
            error = abs(result - target)
            rel_error = error / abs(target) if target != 0 else error
            
            return {
                'success': True,
                'result': result,
                'error': error,
                'rel_error': rel_error,
                'iterations': len(wcs.history['cycles']),  # Fixed: 'cost' -> 'cycles'
                'time': elapsed,
                'weights': weights,
                'converged': rel_error < 0.01
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
    
    def test_binary_rate_optimizer(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cost_func: Callable,
        grad_func: Callable,
        max_iter: int = 50
    ) -> Dict:
        """Test BinaryRateOptimizer (BRO)."""
        start_time = time.time()
        
        try:
            bro = BinaryRateOptimizer(
                max_iter=max_iter,
                tol=1e-6,
                verbose=False
            )
            
            theta_init = np.zeros(X.shape[1])
            theta_final = bro.optimize(X, y, theta_init, cost_func, grad_func)
            
            elapsed = time.time() - start_time
            final_cost = cost_func(theta_final, X, y)
            
            return {
                'success': True,
                'final_cost': final_cost,
                'iterations': len(bro.history['cost']),
                'time': elapsed,
                'theta': theta_final,
                'converged': len(bro.history['cost']) < max_iter
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
    
    def test_adamw(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cost_func: Callable,
        grad_func: Callable,
        max_iter: int = 50,
        use_binary_search: bool = True
    ) -> Dict:
        """Test AdamW optimizer."""
        start_time = time.time()
        
        try:
            adamw = AdamW(
                max_iter=max_iter,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                weight_decay=0.01,
                use_binary_search=use_binary_search,
                base_lr=0.001,
                verbose=False
            )
            
            theta_init = np.zeros(X.shape[1])
            theta_final = adamw.optimize(X, y, theta_init, cost_func, grad_func)
            
            elapsed = time.time() - start_time
            final_cost = cost_func(theta_final, X, y)
            
            return {
                'success': True,
                'final_cost': final_cost,
                'iterations': len(adamw.history['cost']),
                'time': elapsed,
                'theta': theta_final,
                'converged': len(adamw.history['cost']) < max_iter
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
    
    def run_scenario(self, scenario_name: str, n_params: int, difficulty: str):
        """Run all optimizers on a single scenario."""
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name} | N={n_params} | Difficulty={difficulty}")
        print(f"{'='*80}")
        
        # Setup
        coeffs, target, X, y, cost_func, grad_func = self.setup_test_scenario(
            scenario_name, n_params, difficulty
        )
        
        print(f"Target: {target:.4f}")
        print(f"Coefficients shape: {coeffs.shape}")
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Test 1: WeightCombinationSearch (without filtering)
        print("\n[1/4] Testing WeightCombinationSearch (no filtering)...")
        wcs_result = self.test_weight_combination_search(coeffs, target, max_iter=50, with_filtering=False)
        
        # Test 2: WeightCombinationSearch (with filtering)
        print("[2/4] Testing WeightCombinationSearch (with filtering)...")
        wcs_filtered_result = self.test_weight_combination_search(coeffs, target, max_iter=50, with_filtering=True)
        
        # Test 3: BinaryRateOptimizer
        print("[3/4] Testing BinaryRateOptimizer...")
        bro_result = self.test_binary_rate_optimizer(X, y, cost_func, grad_func, max_iter=50)
        
        # Test 4: AdamW
        print("[4/4] Testing AdamW...")
        adamw_result = self.test_adamw(X, y, cost_func, grad_func, max_iter=50, use_binary_search=True)
        
        # Store results
        self.results.append({
            'scenario': scenario_name,
            'n_params': n_params,
            'difficulty': difficulty,
            'target': target,
            'wcs': wcs_result,
            'wcs_filtered': wcs_filtered_result,
            'bro': bro_result,
            'adamw': adamw_result
        })
        
        # Print summary
        print(f"\n{'Results Summary':^80}")
        print(f"{'-'*80}")
        
        if wcs_result['success']:
            print(f"WCS (no filter):  Error={wcs_result['error']:.6f} | "
                  f"Time={wcs_result['time']:.4f}s | Iter={wcs_result['iterations']}")
        
        if wcs_filtered_result['success']:
            print(f"WCS (filtered):   Error={wcs_filtered_result['error']:.6f} | "
                  f"Time={wcs_filtered_result['time']:.4f}s | Iter={wcs_filtered_result['iterations']}")
        
        if bro_result['success']:
            print(f"BinaryRateOpt:    Cost={bro_result['final_cost']:.6f} | "
                  f"Time={bro_result['time']:.4f}s | Iter={bro_result['iterations']}")
        
        if adamw_result['success']:
            print(f"AdamW:            Cost={adamw_result['final_cost']:.6f} | "
                  f"Time={adamw_result['time']:.4f}s | Iter={adamw_result['iterations']}")
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comprehensive comparison table."""
        rows = []
        
        for res in self.results:
            scenario = res['scenario']
            n = res['n_params']
            diff = res['difficulty']
            
            # Extract metrics
            wcs = res['wcs']
            wcs_f = res['wcs_filtered']
            bro = res['bro']
            adamw = res['adamw']
            
            row = {
                'Scenario': scenario,
                'N_Params': n,
                'Difficulty': diff,
                
                # WCS (no filter)
                'WCS_Error': wcs.get('error', np.nan) if wcs['success'] else np.nan,
                'WCS_Time': wcs['time'],
                'WCS_Iter': wcs.get('iterations', 0),
                'WCS_Converged': wcs.get('converged', False) if wcs['success'] else False,
                
                # WCS (filtered)
                'WCS_F_Error': wcs_f.get('error', np.nan) if wcs_f['success'] else np.nan,
                'WCS_F_Time': wcs_f['time'],
                'WCS_F_Iter': wcs_f.get('iterations', 0),
                'WCS_F_Converged': wcs_f.get('converged', False) if wcs_f['success'] else False,
                
                # BRO
                'BRO_Cost': bro.get('final_cost', np.nan) if bro['success'] else np.nan,
                'BRO_Time': bro['time'],
                'BRO_Iter': bro.get('iterations', 0),
                'BRO_Converged': bro.get('converged', False) if bro['success'] else False,
                
                # AdamW
                'AdamW_Cost': adamw.get('final_cost', np.nan) if adamw['success'] else np.nan,
                'AdamW_Time': adamw['time'],
                'AdamW_Iter': adamw.get('iterations', 0),
                'AdamW_Converged': adamw.get('converged', False) if adamw['success'] else False,
            }
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def print_conclusions(self, df: pd.DataFrame):
        """Print detailed conclusions from benchmark results."""
        print("\n" + "="*100)
        print("CONCLUSIONS & ANALYSIS".center(100))
        print("="*100)
        
        print("\n📊 ALGORITHM COMPARISON SUMMARY\n")
        
        # Speed comparison
        print("⏱️  SPEED (Average Time per Scenario)")
        print("-" * 100)
        print(f"  WeightCombinationSearch (no filter): {df['WCS_Time'].mean():.4f}s")
        print(f"  WeightCombinationSearch (filtered):  {df['WCS_F_Time'].mean():.4f}s "
              f"({'faster' if df['WCS_F_Time'].mean() < df['WCS_Time'].mean() else 'slower'} "
              f"by {abs(1 - df['WCS_F_Time'].mean()/df['WCS_Time'].mean())*100:.1f}%)")
        print(f"  BinaryRateOptimizer:                {df['BRO_Time'].mean():.4f}s")
        print(f"  AdamW:                              {df['AdamW_Time'].mean():.4f}s")
        
        fastest = df[['WCS_Time', 'WCS_F_Time', 'BRO_Time', 'AdamW_Time']].mean().idxmin()
        print(f"\n  🏆 Fastest Overall: {fastest.replace('_', ' ')}")
        
        # Convergence comparison
        print("\n✅ CONVERGENCE (Percentage of Scenarios that Converged)")
        print("-" * 100)
        print(f"  WeightCombinationSearch (no filter): {df['WCS_Converged'].mean()*100:.1f}%")
        print(f"  WeightCombinationSearch (filtered):  {df['WCS_F_Converged'].mean()*100:.1f}%")
        print(f"  BinaryRateOptimizer:                {df['BRO_Converged'].mean()*100:.1f}%")
        print(f"  AdamW:                              {df['AdamW_Converged'].mean()*100:.1f}%")
        
        # Iteration efficiency
        print("\n🔄 ITERATION EFFICIENCY (Average Iterations to Completion)")
        print("-" * 100)
        print(f"  WeightCombinationSearch (no filter): {df['WCS_Iter'].mean():.1f}")
        print(f"  WeightCombinationSearch (filtered):  {df['WCS_F_Iter'].mean():.1f}")
        print(f"  BinaryRateOptimizer:                {df['BRO_Iter'].mean():.1f}")
        print(f"  AdamW:                              {df['AdamW_Iter'].mean():.1f}")
        
        # Accuracy (where applicable)
        print("\n🎯 ACCURACY (For WCS: Average Relative Error)")
        print("-" * 100)
        wcs_rel_error = (df['WCS_Error'] / df['Scenario'].map(lambda x: self.results[df[df['Scenario']==x].index[0]]['target'])).mean()
        wcs_f_rel_error = (df['WCS_F_Error'] / df['Scenario'].map(lambda x: self.results[df[df['Scenario']==x].index[0]]['target'])).mean()
        print(f"  WeightCombinationSearch (no filter): {wcs_rel_error*100:.2f}% relative error")
        print(f"  WeightCombinationSearch (filtered):  {wcs_f_rel_error*100:.2f}% relative error")
        print(f"  BinaryRateOptimizer:                Cost={df['BRO_Cost'].mean():.6f} (MSE)")
        print(f"  AdamW:                              Cost={df['AdamW_Cost'].mean():.6f} (MSE)")
        
        # Use case recommendations
        print("\n💡 RECOMMENDATIONS BY USE CASE")
        print("-" * 100)
        
        print("\n  1️⃣  DISCRETE WEIGHT OPTIMIZATION (Binary weights: 0 or 1)")
        print("      → Use: WeightCombinationSearch")
        print("      → Reason: Specifically designed for combinatorial optimization")
        print("      → Filtering: Enable for N > 15 parameters")
        
        print("\n  2️⃣  CONTINUOUS PARAMETER OPTIMIZATION (e.g., Neural Network training)")
        print("      → Use: AdamW")
        print("      → Reason: Adaptive learning rates + momentum + weight decay")
        print("      → Best for: High-dimensional, noisy gradients")
        
        print("\n  3️⃣  LEARNING RATE SEARCH (Finding optimal step size)")
        print("      → Use: BinaryRateOptimizer")
        print("      → Reason: Dynamic line search eliminates manual LR tuning")
        print("      → Best for: Small to medium problems (N < 100)")
        
        print("\n  4️⃣  SMALL PROBLEMS (N < 10 parameters)")
        print("      → Use: WeightCombinationSearch (no filtering) or BinaryRateOptimizer")
        print("      → Reason: Low overhead, fast exhaustive search")
        
        print("\n  5️⃣  LARGE PROBLEMS (N > 50 parameters)")
        print("      → Use: AdamW or WeightCombinationSearch with filtering")
        print("      → Reason: Scales well with dimensionality")
        
        print("\n📝 KEY INSIGHTS")
        print("-" * 100)
        
        print("\n  ✓ WeightCombinationSearch:")
        print("    • Unique strength: Combinatorial optimization (binary weights)")
        print("    • Filtering provides dramatic speedup for N > 15")
        print("    • Deterministic results (no randomness)")
        
        print("\n  ✓ BinaryRateOptimizer:")
        print("    • Unique strength: Adaptive learning rate per iteration")
        print("    • No hyperparameter tuning needed")
        print("    • Best for small-medium smooth problems")
        
        print("\n  ✓ AdamW:")
        print("    • Unique strength: Adaptive per-parameter learning rates")
        print("    • Industry standard for deep learning")
        print("    • Best for high-dimensional, noisy gradients")
        
        print("\n" + "="*100)


def main():
    """Run comprehensive benchmark."""
    print("="*100)
    print("OPTIMIZER COMPARISON BENCHMARK".center(100))
    print("WeightCombinationSearch vs BinaryRateOptimizer vs AdamW".center(100))
    print("="*100)
    
    benchmark = OptimizerBenchmark()
    
    # Test scenarios
    scenarios = [
        ("Small Easy", 5, 'easy'),
        ("Small Medium", 5, 'medium'),
        ("Small Hard", 5, 'hard'),
        ("Medium Easy", 10, 'easy'),
        ("Medium Medium", 10, 'medium'),
        ("Medium Hard", 10, 'hard'),
        ("Large Easy", 20, 'easy'),
        ("Large Medium", 20, 'medium'),
    ]
    
    # Run all scenarios
    for scenario_name, n_params, difficulty in scenarios:
        benchmark.run_scenario(scenario_name, n_params, difficulty)
    
    # Generate comparison table
    print("\n" + "="*100)
    print("DETAILED COMPARISON TABLE".center(100))
    print("="*100)
    
    df = benchmark.generate_comparison_table()
    
    # Print formatted table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    
    print("\n" + df.to_string(index=False))
    
    # Save to CSV
    csv_path = "optimizer_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Print conclusions
    benchmark.print_conclusions(df)
    
    print("\n" + "="*100)
    print("BENCHMARK COMPLETE".center(100))
    print("="*100)


if __name__ == "__main__":
    main()
