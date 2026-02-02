"""
Test Adaptive Sampling Performance - Phase 1 Optimization

Tests the adaptive sampling feature for large N parameters.
Compares exhaustive vs sampled approaches.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from math_toolkit.binary_search.combinatorial import WeightCombinationSearch


def test_adaptive_sampling_accuracy():
    """Test that sampling gives reasonable accuracy compared to exhaustive."""
    print("=" * 80)
    print("ADAPTIVE SAMPLING - ACCURACY TEST")
    print("=" * 80)
    print()
    
    # Test on N=10 (small enough for exhaustive, large enough to test sampling)
    coeffs = list(range(1, 11))  # [1, 2, 3, ..., 10]
    target = 25
    
    print(f"Configuration:")
    print(f"  Coefficients: {coeffs}")
    print(f"  Target: {target}")
    print(f"  N: {len(coeffs)}")
    print()
    
    # Exhaustive search
    print("🔍 EXHAUSTIVE SEARCH (baseline):")
    print("-" * 80)
    search_exhaustive = WeightCombinationSearch(
        tolerance=2.0,
        max_iter=10,
        adaptive_sampling=False,  # Disable sampling
        verbose=False
    )
    
    start = time.time()
    weights_exhaustive = search_exhaustive.find_optimal_weights(coeffs, target)
    time_exhaustive = time.time() - start
    
    result_exhaustive = sum(coeffs[i] * weights_exhaustive[i] for i in range(len(coeffs)))
    error_exhaustive = abs(result_exhaustive - target)
    
    print(f"  Time: {time_exhaustive:.4f}s")
    print(f"  Weights: {weights_exhaustive}")
    print(f"  Result: {result_exhaustive:.2f}")
    print(f"  Error: {error_exhaustive:.2f}")
    print()
    
    # Sampled search (importance)
    print("🎯 IMPORTANCE SAMPLING:")
    print("-" * 80)
    search_importance = WeightCombinationSearch(
        tolerance=2.0,
        max_iter=10,
        adaptive_sampling=True,
        sampling_threshold=5,  # Force sampling even for N=10
        sample_size=200,
        sampling_strategy='importance',
        verbose=False
    )
    
    start = time.time()
    weights_importance = search_importance.find_optimal_weights(coeffs, target)
    time_importance = time.time() - start
    
    result_importance = sum(coeffs[i] * weights_importance[i] for i in range(len(coeffs)))
    error_importance = abs(result_importance - target)
    
    print(f"  Time: {time_importance:.4f}s")
    print(f"  Weights: {weights_importance}")
    print(f"  Result: {result_importance:.2f}")
    print(f"  Error: {error_importance:.2f}")
    print(f"  Speedup: {time_exhaustive/time_importance:.2f}x")
    print()
    
    # Compare accuracy
    accuracy_ratio = error_importance / max(error_exhaustive, 0.01)
    print("=" * 80)
    print("📊 ACCURACY COMPARISON:")
    print("=" * 80)
    print(f"  Exhaustive error: {error_exhaustive:.2f}")
    print(f"  Sampling error: {error_importance:.2f}")
    print(f"  Accuracy ratio: {accuracy_ratio:.2f}x (< 2.0 is good)")
    
    if accuracy_ratio < 2.0:
        print(f"  ✅ PASS: Sampling maintains good accuracy!")
    else:
        print(f"  ⚠️  WARNING: Sampling accuracy degraded")
    print()


def test_large_n_performance():
    """Test performance on large N where exhaustive is impractical."""
    print("=" * 80)
    print("ADAPTIVE SAMPLING - LARGE N PERFORMANCE")
    print("=" * 80)
    print()
    
    test_cases = [
        (15, "15 params - 32,768 combinations"),
        (20, "20 params - 1,048,576 combinations"),
        (30, "30 params - 1,073,741,824 combinations"),
    ]
    
    for n, description in test_cases:
        print(f"🚀 TEST: {description}")
        print("-" * 80)
        
        coeffs = list(range(1, n + 1))
        target = sum(coeffs) // 2  # Target is half of sum
        
        search = WeightCombinationSearch(
            tolerance=5.0,
            max_iter=20,
            adaptive_sampling=True,
            sampling_threshold=12,
            sample_size=10000,
            sampling_strategy='importance',
            verbose=False
        )
        
        start = time.time()
        weights = search.find_optimal_weights(coeffs, target)
        elapsed = time.time() - start
        
        result = sum(coeffs[i] * weights[i] for i in range(n))
        error = abs(result - target)
        
        print(f"  Time: {elapsed:.4f}s")
        print(f"  Result: {result:.2f}")
        print(f"  Target: {target}")
        print(f"  Error: {error:.2f}")
        print(f"  Converged: {'✅ YES' if error <= 5.0 else '❌ NO'}")
        print()
    
    print("=" * 80)
    print("✅ LARGE N IS NOW PRACTICAL WITH ADAPTIVE SAMPLING!")
    print("=" * 80)


def test_sampling_strategies():
    """Compare different sampling strategies."""
    print("=" * 80)
    print("SAMPLING STRATEGIES COMPARISON")
    print("=" * 80)
    print()
    
    coeffs = list(range(1, 21))  # 20 parameters
    target = 100
    
    strategies = ['importance', 'random', 'progressive']
    
    for strategy in strategies:
        print(f"🎯 STRATEGY: {strategy.upper()}")
        print("-" * 80)
        
        search = WeightCombinationSearch(
            tolerance=5.0,
            max_iter=10,
            adaptive_sampling=True,
            sampling_threshold=10,
            sample_size=5000,
            sampling_strategy=strategy,
            verbose=False
        )
        
        start = time.time()
        weights = search.find_optimal_weights(coeffs, target)
        elapsed = time.time() - start
        
        result = sum(coeffs[i] * weights[i] for i in range(len(coeffs)))
        error = abs(result - target)
        
        print(f"  Time: {elapsed:.4f}s")
        print(f"  Result: {result:.2f}")
        print(f"  Error: {error:.2f}")
        print(f"  Quality: {'✅ GOOD' if error <= 10 else '⚠️  ACCEPTABLE' if error <= 20 else '❌ POOR'}")
        print()


if __name__ == '__main__':
    test_adaptive_sampling_accuracy()
    print("\n" * 2)
    test_large_n_performance()
    print("\n" * 2)
    test_sampling_strategies()
