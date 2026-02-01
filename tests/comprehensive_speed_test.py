#!/usr/bin/env python3
"""
Comprehensive Speed Test - WeightCombinationSearch vs AdamW
Unbiased testing with multiple scenarios, parameter counts, and iterations
"""

import sys
import time
import statistics
sys.path.insert(0, '../binary_search')

from math_toolkit.binary_search import WeightCombinationSearch
from math_toolkit.optimization.adaptive_optimizer import AdamW
import numpy as np


def test_weight_combination_search(coefficients, target, tolerance, max_iter, runs=10):
    """Test WeightCombinationSearch multiple times and return statistics"""
    times = []
    errors = []
    converged_count = 0
    
    for _ in range(runs):
        search = WeightCombinationSearch(tolerance=tolerance, max_iter=max_iter)
        
        start = time.perf_counter()
        weights = search.find_optimal_weights(coefficients, target)
        end = time.perf_counter()
        
        times.append(end - start)
        
        # Calculate result
        result = sum(c * w for c, w in zip(coefficients, weights))
        error = abs(result - target)
        errors.append(error)
        
        if error <= tolerance:
            converged_count += 1
    
    return {
        'times': times,
        'avg_time': statistics.mean(times),
        'min_time': min(times),
        'max_time': max(times),
        'std_time': statistics.stdev(times) if len(times) > 1 else 0,
        'errors': errors,
        'avg_error': statistics.mean(errors),
        'converged_rate': converged_count / runs
    }


def test_adamw(coefficients, target, tolerance, max_iter, runs=10):
    """Test AdamW multiple times and return statistics"""
    times = []
    errors = []
    converged_count = 0
    
    for _ in range(runs):
        # Initialize AdamW with correct parameters
        n_params = len(coefficients)
        initial_weights = np.ones(n_params) / n_params
        
        # Create dummy X, y for AdamW (it requires them even though we use custom cost_fn)
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
        
        # Define cost function: minimize (coefficients · weights - target)²
        def cost_fn(theta, X, y):
            result = sum(c * w for c, w in zip(coefficients, theta))
            return (result - target) ** 2
        
        # Define gradient: 2 * (result - target) * coefficient[i]
        def gradient_fn(theta, X, y):
            result = sum(c * w for c, w in zip(coefficients, theta))
            diff = result - target
            return np.array([2 * diff * c for c in coefficients])
        
        start = time.perf_counter()
        final_weights = optimizer.optimize(
            X=X_dummy,
            y=y_dummy,
            initial_theta=initial_weights,
            cost_func=cost_fn,
            grad_func=gradient_fn
        )
        end = time.perf_counter()
        
        times.append(end - start)
        
        # Check convergence
        final_result = sum(c * w for c, w in zip(coefficients, final_weights))
        error = abs(final_result - target)
        errors.append(error)
        
        if error <= tolerance:
            converged_count += 1
    
    return {
        'times': times,
        'avg_time': statistics.mean(times),
        'min_time': min(times),
        'max_time': max(times),
        'std_time': statistics.stdev(times) if len(times) > 1 else 0,
        'errors': errors,
        'avg_error': statistics.mean(errors),
        'converged_rate': converged_count / runs
    }


def run_comprehensive_test(name, coefficients, target, tolerance=2.0, max_iter=50, runs=20):
    """Run comprehensive test comparing both methods"""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")
    print(f"Coefficients: {coefficients}")
    print(f"Target: {target}")
    print(f"Tolerance: {tolerance}")
    print(f"Max Iterations: {max_iter}")
    print(f"Number of runs: {runs}")
    print()
    
    # Test WeightCombinationSearch
    print("Testing WeightCombinationSearch...")
    wcs_results = test_weight_combination_search(coefficients, target, tolerance, max_iter, runs)
    
    # Test AdamW
    print("Testing AdamW...")
    adamw_results = test_adamw(coefficients, target, tolerance, max_iter, runs)
    
    # Print results
    print(f"\n{'─'*80}")
    print("RESULTS")
    print(f"{'─'*80}")
    
    print(f"\n📊 WeightCombinationSearch ({runs} runs)")
    print(f"   Time (avg):     {wcs_results['avg_time']*1000:.3f} ms")
    print(f"   Time (min):     {wcs_results['min_time']*1000:.3f} ms")
    print(f"   Time (max):     {wcs_results['max_time']*1000:.3f} ms")
    print(f"   Time (std):     {wcs_results['std_time']*1000:.3f} ms")
    print(f"   Error (avg):    {wcs_results['avg_error']:.4f}")
    print(f"   Converged:      {wcs_results['converged_rate']*100:.1f}% ({int(wcs_results['converged_rate']*runs)}/{runs})")
    
    print(f"\n📊 AdamW ({runs} runs)")
    print(f"   Time (avg):     {adamw_results['avg_time']*1000:.3f} ms")
    print(f"   Time (min):     {adamw_results['min_time']*1000:.3f} ms")
    print(f"   Time (max):     {adamw_results['max_time']*1000:.3f} ms")
    print(f"   Time (std):     {adamw_results['std_time']*1000:.3f} ms")
    print(f"   Error (avg):    {adamw_results['avg_error']:.4f}")
    print(f"   Converged:      {adamw_results['converged_rate']*100:.1f}% ({int(adamw_results['converged_rate']*runs)}/{runs})")
    
    # Comparison
    print(f"\n{'─'*80}")
    print("COMPARISON")
    print(f"{'─'*80}")
    
    speed_ratio = adamw_results['avg_time'] / wcs_results['avg_time']
    if speed_ratio > 1:
        print(f"⚡ Speed:          WeightCombinationSearch is {speed_ratio:.2f}x FASTER")
    else:
        print(f"⚡ Speed:          AdamW is {1/speed_ratio:.2f}x FASTER")
    
    error_diff = adamw_results['avg_error'] - wcs_results['avg_error']
    if error_diff > 0:
        print(f"🎯 Accuracy:       WeightCombinationSearch is MORE ACCURATE (Δerror: {error_diff:.4f})")
    else:
        print(f"🎯 Accuracy:       AdamW is MORE ACCURATE (Δerror: {-error_diff:.4f})")
    
    conv_diff = wcs_results['converged_rate'] - adamw_results['converged_rate']
    if conv_diff > 0:
        print(f"✅ Convergence:    WeightCombinationSearch {conv_diff*100:+.1f}% better")
    else:
        print(f"✅ Convergence:    AdamW {-conv_diff*100:+.1f}% better")
    
    return {
        'name': name,
        'n_params': len(coefficients),
        'wcs': wcs_results,
        'adamw': adamw_results
    }


if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE SPEED TEST - WeightCombinationSearch vs AdamW")
    print("="*80)
    print("Objective: Unbiased comparison with multiple runs and statistical analysis")
    print()
    
    all_results = []
    
    # Test 1: 3 Parameters
    all_results.append(run_comprehensive_test(
        "3 Parameters - Simple",
        [15, 47, -12],
        28,
        tolerance=2.0,
        max_iter=50,
        runs=20
    ))
    
    # Test 2: 5 Parameters
    all_results.append(run_comprehensive_test(
        "5 Parameters - Linear",
        [5, 10, 15, 20, 25],
        100,
        tolerance=2.0,
        max_iter=50,
        runs=20
    ))
    
    # Test 3: 7 Parameters
    all_results.append(run_comprehensive_test(
        "7 Parameters - Mixed",
        [8, 12, 18, 24, 30, 36, 42],
        150,
        tolerance=5.0,
        max_iter=50,
        runs=20
    ))
    
    # Test 4: 10 Parameters
    all_results.append(run_comprehensive_test(
        "10 Parameters - Large",
        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        200,
        tolerance=5.0,
        max_iter=50,
        runs=20
    ))
    
    # Test 5: 3 Parameters - Different scenario
    all_results.append(run_comprehensive_test(
        "3 Parameters - Large Numbers",
        [100, 250, 500],
        1000,
        tolerance=10.0,
        max_iter=50,
        runs=20
    ))
    
    # Test 6: 5 Parameters - Small numbers
    all_results.append(run_comprehensive_test(
        "5 Parameters - Small Numbers",
        [1, 2, 3, 4, 5],
        20,
        tolerance=1.0,
        max_iter=50,
        runs=20
    ))
    
    # Test 7: 8 Parameters
    all_results.append(run_comprehensive_test(
        "8 Parameters",
        [3, 7, 11, 15, 19, 23, 27, 31],
        120,
        tolerance=3.0,
        max_iter=50,
        runs=20
    ))
    
    # Test 8: 12 Parameters
    all_results.append(run_comprehensive_test(
        "12 Parameters",
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        150,
        tolerance=5.0,
        max_iter=50,
        runs=10  # Fewer runs for larger tests
    ))
    
    # Test 9: 15 Parameters
    all_results.append(run_comprehensive_test(
        "15 Parameters",
        list(range(1, 16)),
        100,
        tolerance=5.0,
        max_iter=50,
        runs=5  # Even fewer runs
    ))
    
    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - ALL TESTS")
    print(f"{'='*80}\n")
    
    print(f"{'Test Name':<30} {'Params':<8} {'WCS Time (ms)':<15} {'AdamW Time (ms)':<15} {'Winner':<10}")
    print(f"{'-'*80}")
    
    for result in all_results:
        wcs_time = result['wcs']['avg_time'] * 1000
        adamw_time = result['adamw']['avg_time'] * 1000
        winner = 'WCS' if wcs_time < adamw_time else 'AdamW'
        
        print(f"{result['name']:<30} {result['n_params']:<8} {wcs_time:<15.3f} {adamw_time:<15.3f} {winner:<10}")
    
    print(f"\n{'='*80}")
    print("ACCURACY SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Test Name':<30} {'Params':<8} {'WCS Error':<15} {'AdamW Error':<15} {'WCS Conv %':<12} {'AdamW Conv %':<12}")
    print(f"{'-'*80}")
    
    for result in all_results:
        wcs_err = result['wcs']['avg_error']
        adamw_err = result['adamw']['avg_error']
        wcs_conv = result['wcs']['converged_rate'] * 100
        adamw_conv = result['adamw']['converged_rate'] * 100
        
        print(f"{result['name']:<30} {result['n_params']:<8} {wcs_err:<15.4f} {adamw_err:<15.4f} {wcs_conv:<12.1f} {adamw_conv:<12.1f}")
    
    print(f"\n{'='*80}")
    print("✅ TESTING COMPLETE")
    print(f"{'='*80}")
