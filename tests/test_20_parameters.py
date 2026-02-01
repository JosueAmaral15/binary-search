#!/usr/bin/env python3
"""
20 Parameters Test - WeightCombinationSearch vs AdamW
Single focused test to show the TRUTH about performance at 20 parameters
"""

import sys
import time
import statistics
sys.path.insert(0, '../binary_search')

from math_toolkit.binary_search import WeightCombinationSearch
from math_toolkit.optimization.adaptive_optimizer import AdamW
import numpy as np


def test_weight_combination_search(coefficients, target, tolerance, max_iter, run_number):
    """Test WeightCombinationSearch once and return detailed results"""
    search = WeightCombinationSearch(tolerance=tolerance, max_iter=max_iter, verbose=False)
    
    print(f"   Run {run_number}: Starting...", end='', flush=True)
    start = time.perf_counter()
    weights = search.find_optimal_weights(coefficients, target)
    end = time.perf_counter()
    
    # Calculate result
    result = sum(c * w for c, w in zip(coefficients, weights))
    error = abs(result - target)
    converged = error <= tolerance
    elapsed = end - start
    
    print(f" {elapsed:.3f}s (error: {error:.4f}, {'✓ CONVERGED' if converged else '✗ FAILED'})")
    
    return {
        'time': elapsed,
        'error': error,
        'converged': converged,
        'weights': weights,
        'result': result
    }


def test_adamw(coefficients, target, tolerance, max_iter, run_number):
    """Test AdamW once and return detailed results"""
    n_params = len(coefficients)
    initial_weights = np.ones(n_params) / n_params
    
    X_dummy = np.array([[1.0]])
    y_dummy = np.array([target])
    
    optimizer = AdamW(
        max_iter=max_iter,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.01,
        base_lr=0.001,
        verbose=False
    )
    
    def cost_fn(theta, X, y):
        result = sum(c * w for c, w in zip(coefficients, theta))
        return (result - target) ** 2
    
    def gradient_fn(theta, X, y):
        result = sum(c * w for c, w in zip(coefficients, theta))
        diff = result - target
        return np.array([2 * diff * c for c in coefficients])
    
    print(f"   Run {run_number}: Starting...", end='', flush=True)
    start = time.perf_counter()
    final_weights = optimizer.optimize(
        X=X_dummy,
        y=y_dummy,
        initial_theta=initial_weights,
        cost_func=cost_fn,
        grad_func=gradient_fn
    )
    end = time.perf_counter()
    
    final_result = sum(c * w for c, w in zip(coefficients, final_weights))
    error = abs(final_result - target)
    converged = error <= tolerance
    elapsed = end - start
    
    print(f" {elapsed:.3f}s (error: {error:.4f}, {'✓ CONVERGED' if converged else '✗ FAILED'})")
    
    return {
        'time': elapsed,
        'error': error,
        'converged': converged,
        'weights': final_weights,
        'result': final_result
    }


if __name__ == '__main__':
    print("="*80)
    print("20 PARAMETERS TEST - WeightCombinationSearch vs AdamW")
    print("="*80)
    print("\nObjective: Show the TRUTH about performance at 20 parameters")
    print("Method: 5 runs each, report ALL results honestly\n")
    
    # Test configuration
    np.random.seed(42)
    coefficients = list(range(1, 21))  # [1, 2, 3, ..., 20]
    target = 200
    tolerance = 5.0
    max_iter = 50
    num_runs = 5
    
    print(f"Configuration:")
    print(f"  Coefficients: {coefficients}")
    print(f"  Target: {target}")
    print(f"  Tolerance: {tolerance}")
    print(f"  Max Iterations: {max_iter}")
    print(f"  Number of Runs: {num_runs}")
    print()
    
    # Test WeightCombinationSearch
    print("─"*80)
    print("1️⃣  TESTING: WeightCombinationSearch (20 parameters)")
    print("─"*80)
    
    wcs_results = []
    for i in range(1, num_runs + 1):
        result = test_weight_combination_search(coefficients, target, tolerance, max_iter, i)
        wcs_results.append(result)
    
    print()
    
    # Test AdamW
    print("─"*80)
    print("2️⃣  TESTING: AdamW (20 parameters)")
    print("─"*80)
    
    adamw_results = []
    for i in range(1, num_runs + 1):
        result = test_adamw(coefficients, target, tolerance, max_iter, i)
        adamw_results.append(result)
    
    print()
    
    # Analyze results
    print("="*80)
    print("RESULTS - THE TRUTH")
    print("="*80)
    
    # WeightCombinationSearch statistics
    wcs_times = [r['time'] for r in wcs_results]
    wcs_errors = [r['error'] for r in wcs_results]
    wcs_converged = [r['converged'] for r in wcs_results]
    
    print(f"\n📊 WeightCombinationSearch (20 parameters, {num_runs} runs)")
    print(f"   Time (avg):     {statistics.mean(wcs_times):.3f} seconds")
    print(f"   Time (min):     {min(wcs_times):.3f} seconds")
    print(f"   Time (max):     {max(wcs_times):.3f} seconds")
    print(f"   Time (std):     {statistics.stdev(wcs_times) if len(wcs_times) > 1 else 0:.3f} seconds")
    print(f"   Error (avg):    {statistics.mean(wcs_errors):.4f}")
    print(f"   Error (min):    {min(wcs_errors):.4f}")
    print(f"   Error (max):    {max(wcs_errors):.4f}")
    print(f"   Converged:      {sum(wcs_converged)}/{num_runs} ({sum(wcs_converged)/num_runs*100:.1f}%)")
    
    # Individual run details
    print(f"\n   Individual runs:")
    for i, r in enumerate(wcs_results, 1):
        status = "✓ CONVERGED" if r['converged'] else "✗ FAILED"
        print(f"      Run {i}: {r['time']:.3f}s, error={r['error']:.4f}, result={r['result']:.2f} {status}")
    
    # AdamW statistics
    adamw_times = [r['time'] for r in adamw_results]
    adamw_errors = [r['error'] for r in adamw_results]
    adamw_converged = [r['converged'] for r in adamw_results]
    
    print(f"\n📊 AdamW (20 parameters, {num_runs} runs)")
    print(f"   Time (avg):     {statistics.mean(adamw_times):.3f} seconds")
    print(f"   Time (min):     {min(adamw_times):.3f} seconds")
    print(f"   Time (max):     {max(adamw_times):.3f} seconds")
    print(f"   Time (std):     {statistics.stdev(adamw_times) if len(adamw_times) > 1 else 0:.3f} seconds")
    print(f"   Error (avg):    {statistics.mean(adamw_errors):.4f}")
    print(f"   Error (min):    {min(adamw_errors):.4f}")
    print(f"   Error (max):    {max(adamw_errors):.4f}")
    print(f"   Converged:      {sum(adamw_converged)}/{num_runs} ({sum(adamw_converged)/num_runs*100:.1f}%)")
    
    # Individual run details
    print(f"\n   Individual runs:")
    for i, r in enumerate(adamw_results, 1):
        status = "✓ CONVERGED" if r['converged'] else "✗ FAILED"
        print(f"      Run {i}: {r['time']:.3f}s, error={r['error']:.4f}, result={r['result']:.2f} {status}")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON - THE TRUTH")
    print(f"{'='*80}")
    
    avg_wcs_time = statistics.mean(wcs_times)
    avg_adamw_time = statistics.mean(adamw_times)
    
    if avg_adamw_time < avg_wcs_time:
        ratio = avg_wcs_time / avg_adamw_time
        print(f"\n⚡ SPEED WINNER: AdamW")
        print(f"   AdamW is {ratio:.2f}x FASTER than WeightCombinationSearch")
        print(f"   WCS: {avg_wcs_time:.3f}s vs AdamW: {avg_adamw_time:.3f}s")
    else:
        ratio = avg_adamw_time / avg_wcs_time
        print(f"\n⚡ SPEED WINNER: WeightCombinationSearch")
        print(f"   WCS is {ratio:.2f}x FASTER than AdamW")
        print(f"   WCS: {avg_wcs_time:.3f}s vs AdamW: {avg_adamw_time:.3f}s")
    
    avg_wcs_error = statistics.mean(wcs_errors)
    avg_adamw_error = statistics.mean(adamw_errors)
    
    if avg_wcs_error < avg_adamw_error:
        print(f"\n🎯 ACCURACY WINNER: WeightCombinationSearch")
        print(f"   WCS error: {avg_wcs_error:.4f} vs AdamW error: {avg_adamw_error:.4f}")
    elif avg_adamw_error < avg_wcs_error:
        print(f"\n🎯 ACCURACY WINNER: AdamW")
        print(f"   AdamW error: {avg_adamw_error:.4f} vs WCS error: {avg_wcs_error:.4f}")
    else:
        print(f"\n🎯 ACCURACY: TIE")
        print(f"   Both: {avg_wcs_error:.4f}")
    
    wcs_conv_rate = sum(wcs_converged) / num_runs
    adamw_conv_rate = sum(adamw_converged) / num_runs
    
    if wcs_conv_rate > adamw_conv_rate:
        print(f"\n✅ CONVERGENCE WINNER: WeightCombinationSearch")
        print(f"   WCS: {wcs_conv_rate*100:.1f}% vs AdamW: {adamw_conv_rate*100:.1f}%")
    elif adamw_conv_rate > wcs_conv_rate:
        print(f"\n✅ CONVERGENCE WINNER: AdamW")
        print(f"   AdamW: {adamw_conv_rate*100:.1f}% vs WCS: {wcs_conv_rate*100:.1f}%")
    else:
        print(f"\n✅ CONVERGENCE: TIE")
        print(f"   Both: {wcs_conv_rate*100:.1f}%")
    
    print(f"\n{'='*80}")
    print("FINAL VERDICT FOR 20 PARAMETERS")
    print(f"{'='*80}")
    
    if avg_adamw_time < avg_wcs_time:
        ratio = avg_wcs_time / avg_adamw_time
        print(f"\nAt 20 parameters, AdamW is the CLEAR WINNER:")
        print(f"  • {ratio:.2f}x faster ({avg_adamw_time:.3f}s vs {avg_wcs_time:.3f}s)")
        print(f"  • Convergence: {adamw_conv_rate*100:.1f}% vs {wcs_conv_rate*100:.1f}%")
        print(f"  • Error: {avg_adamw_error:.4f} vs {avg_wcs_error:.4f}")
        print(f"\nTRUTH: WeightCombinationSearch is NOT practical for 20 parameters.")
    else:
        ratio = avg_adamw_time / avg_wcs_time
        print(f"\nAt 20 parameters, WeightCombinationSearch is the CLEAR WINNER:")
        print(f"  • {ratio:.2f}x faster ({avg_wcs_time:.3f}s vs {avg_adamw_time:.3f}s)")
        print(f"  • Convergence: {wcs_conv_rate*100:.1f}% vs {adamw_conv_rate*100:.1f}%")
        print(f"  • Error: {avg_wcs_error:.4f} vs {avg_adamw_error:.4f}")
    
    print(f"\n{'='*80}")
    print("✅ TEST COMPLETE - ALL RESULTS REPORTED HONESTLY")
    print(f"{'='*80}")
