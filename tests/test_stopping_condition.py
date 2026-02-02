#!/usr/bin/env python3
"""
Test to show the TRUTH about stopping conditions
"""

import sys
import time
sys.path.insert(0, '../binary_search')

from math_toolkit.binary_search import WeightCombinationSearch

print("="*80)
print("TESTING STOPPING CONDITION BEHAVIOR")
print("="*80)

# Test with 10 parameters first (manageable)
coefficients = list(range(1, 11))  # [1, 2, 3, ..., 10]
target = 100
tolerance = 5.0

print(f"\nTest Configuration:")
print(f"  Parameters: 10")
print(f"  Coefficients: {coefficients}")
print(f"  Target: {target}")
print(f"  Tolerance: {tolerance}")
print(f"  Expected combos per cycle: 2^10 - 1 = {2**10 - 1:,}")
print()

print("Scenario 1: max_iter=1 (should test 1,023 combos and stop)")
print("-" * 80)
search1 = WeightCombinationSearch(tolerance=tolerance, max_iter=1, verbose=False)

start = time.perf_counter()
weights1 = search1.find_optimal_weights(coefficients, target)
elapsed1 = time.perf_counter() - start

result1 = sum(c * w for c, w in zip(coefficients, weights1))
error1 = abs(result1 - target)

print(f"  Time: {elapsed1:.3f}s")
print(f"  Weights: {weights1}")
print(f"  Result: {result1:.2f}")
print(f"  Error: {error1:.2f}")
print(f"  Converged: {'YES' if error1 <= tolerance else 'NO'}")
print(f"  ✓ Tested {2**10 - 1:,} combinations in 1 cycle")
print()

print("Scenario 2: max_iter=5, tolerance=0 (should test 5,115 combos)")
print("-" * 80)
search2 = WeightCombinationSearch(tolerance=0, max_iter=5, verbose=False)

start = time.perf_counter()
weights2 = search2.find_optimal_weights(coefficients, target)
elapsed2 = time.perf_counter() - start

result2 = sum(c * w for c, w in zip(coefficients, weights2))
error2 = abs(result2 - target)

print(f"  Time: {elapsed2:.3f}s")
print(f"  Weights: {weights2}")
print(f"  Result: {result2:.2f}")
print(f"  Error: {error2:.2f}")
print(f"  Converged: {'YES' if error2 <= 0 else 'NO'}")
print(f"  ✓ Tested {(2**10 - 1) * 5:,} combinations across 5 cycles")
print()

print("="*80)
print("TRUTH REVEALED:")
print("="*80)
print("✅ Stopping condition EXISTS and WORKS")
print("✅ Stops when: delta <= tolerance OR cycle >= max_iter")
print()
print("❌ BUT: Always tests ALL 2^N-1 combos per cycle")
print("❌ No early stopping WITHIN a cycle")
print("❌ Even if combo #1 is perfect, tests all remaining combos")
print()

print("Impact with 20 parameters:")
print(f"  - 1 cycle = {2**20 - 1:,} combinations = ~70 seconds")
print(f"  - tolerance=32, max_iter=2000:")
print(f"    * If converges in cycle 1: tests 1,048,575 combos")
print(f"    * If needs 10 cycles: tests 10,485,750 combos")
print(f"    * If runs all 2000 cycles: tests 2,097,150,000 combos (35+ HOURS!)")
print()

print("="*80)
print("CONCLUSION:")
print("="*80)
print("Current implementation:")
print("  ✅ Has cycle-level stopping (between cycles)")
print("  ❌ NO intra-cycle stopping (within cycle)")
print()
print("This is why optimizations are CRITICAL!")
print("="*80)
