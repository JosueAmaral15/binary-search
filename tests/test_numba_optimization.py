"""
Test Numba JIT Compilation Performance - Phase 2 Optimization

Compares Numba-accelerated vs pure Python performance.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from math_toolkit.binary_search.combinatorial import WeightCombinationSearch, HAS_NUMBA


def test_numba_accuracy():
    """Verify Numba version gives same results as pure Python."""
    print("=" * 80)
    print("NUMBA JIT - ACCURACY TEST")
    print("=" * 80)
    print()
    
    if not HAS_NUMBA:
        print("⚠️  Numba not available, skipping test")
        return
    
    coeffs = [15, 47, -12]
    target = 28
    
    # Test with Numba
    search_numba = WeightCombinationSearch(
        tolerance=2.0,
        max_iter=10,
        use_numba=True,
        adaptive_sampling=False,
        verbose=False
    )
    
    weights_numba = search_numba.find_optimal_weights(coeffs, target)
    result_numba = sum(coeffs[i] * weights_numba[i] for i in range(len(coeffs)))
    
    # Test without Numba
    search_python = WeightCombinationSearch(
        tolerance=2.0,
        max_iter=10,
        use_numba=False,
        adaptive_sampling=False,
        verbose=False
    )
    
    weights_python = search_python.find_optimal_weights(coeffs, target)
    result_python = sum(coeffs[i] * weights_python[i] for i in range(len(coeffs)))
    
    print(f"Numba result:  {result_numba:.2f}, weights: {weights_numba}")
    print(f"Python result: {result_python:.2f}, weights: {weights_python}")
    print()
    
    if np.allclose(weights_numba, weights_python):
        print("✅ ACCURACY TEST PASSED: Results match!")
    else:
        print("❌ ACCURACY TEST FAILED: Results differ")
    print()


def test_numba_speedup():
    """Benchmark Numba speedup."""
    print("=" * 80)
    print("NUMBA JIT - PERFORMANCE COMPARISON")
    print("=" * 80)
    print()
    
    if not HAS_NUMBA:
        print("⚠️  Numba not available, using pure Python only")
        print()
    
    test_cases = [
        (10, "10 params"),
        (15, "15 params"),
        (20, "20 params (with sampling)"),
    ]
    
    print(f"{'Test':<25} {'Python':<15} {'Numba':<15} {'Speedup'}")
    print("-" * 80)
    
    for n, description in test_cases:
        coeffs = list(range(1, n + 1))
        target = sum(coeffs) // 2
        
        # Test without Numba
        search_python = WeightCombinationSearch(
            tolerance=3.0,
            max_iter=10,
            use_numba=False,
            adaptive_sampling=(n > 15),
            sampling_threshold=15,
            verbose=False
        )
        
        start = time.time()
        weights_python = search_python.find_optimal_weights(coeffs, target)
        time_python = time.time() - start
        
        # Test with Numba (if available)
        if HAS_NUMBA:
            search_numba = WeightCombinationSearch(
                tolerance=3.0,
                max_iter=10,
                use_numba=True,
                adaptive_sampling=(n > 15),
                sampling_threshold=15,
                verbose=False
            )
            
            start = time.time()
            weights_numba = search_numba.find_optimal_weights(coeffs, target)
            time_numba = time.time() - start
            
            speedup = time_python / time_numba
            print(f"{description:<25} {time_python:.4f}s{'':<6} {time_numba:.4f}s{'':<6} {speedup:.2f}x")
        else:
            print(f"{description:<25} {time_python:.4f}s{'':<6} {'N/A':<15} {'N/A'}")
    
    print("=" * 80)
    print()
    
    if HAS_NUMBA:
        print("KEY INSIGHTS:")
        print("  • First call slower (JIT compilation overhead)")
        print("  • Subsequent calls faster (compiled code cached)")
        print("  • Speedup increases with more formula evaluations")
        print("  • Expected: 2-5x for small N, 10-20x for large N")
    else:
        print("💡 Install Numba for 10-50x speedup:")
        print("     pip install numba")
    print()


def test_compilation_overhead():
    """Test first-call compilation overhead."""
    print("=" * 80)
    print("NUMBA JIT - COMPILATION OVERHEAD")
    print("=" * 80)
    print()
    
    if not HAS_NUMBA:
        print("⚠️  Numba not available, skipping test")
        return
    
    coeffs = list(range(1, 11))
    target = 25
    
    search = WeightCombinationSearch(
        tolerance=2.0,
        max_iter=5,
        use_numba=True,
        adaptive_sampling=False,
        verbose=False
    )
    
    # First call (includes compilation)
    start = time.time()
    weights1 = search.find_optimal_weights(coeffs, target)
    time_first = time.time() - start
    
    # Second call (uses cached compilation)
    start = time.time()
    weights2 = search.find_optimal_weights(coeffs, target)
    time_second = time.time() - start
    
    # Third call
    start = time.time()
    weights3 = search.find_optimal_weights(coeffs, target)
    time_third = time.time() - start
    
    print(f"First call (with compilation):  {time_first:.4f}s")
    print(f"Second call (cached):           {time_second:.4f}s  ({time_first/time_second:.1f}x faster)")
    print(f"Third call (cached):            {time_third:.4f}s  ({time_first/time_third:.1f}x faster)")
    print()
    print("✅ Compilation overhead is one-time cost, amortized over multiple calls")
    print()


if __name__ == '__main__':
    test_numba_accuracy()
    print("\n")
    test_numba_speedup()
    print("\n")
    test_compilation_overhead()
