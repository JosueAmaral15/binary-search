"""
Quick test: Index filtering basic functionality
"""

import numpy as np
from math_toolkit.binary_search.combinatorial import WeightCombinationSearch

# Test 1: Small problem with filtering enabled
print("=" * 80)
print("TEST 1: Index Filtering Enabled (10 coefficients)")
print("=" * 80)

search = WeightCombinationSearch(
    tolerance=2,
    max_iter=5,
    index_filtering=True,
    verbose=True
)

coeffs = [1, 7, 4, 9, 3, 4, -2, -7, 11, 20]
target = 28

weights = search.find_optimal_weights(coeffs, target)

print(f"\nFinal weights: {weights}")
print(f"\nFiltering history:")
for entry in search.history['filtering_history']:
    print(f"  Cycle {entry['cycle']}: {entry['num_filtered']} indices, {entry['percentage']*100:.1f}% accurate")

print("\n" + "=" * 80)
print("TEST 2: Index Filtering Disabled (comparison)")
print("=" * 80)

search2 = WeightCombinationSearch(
    tolerance=2,
    max_iter=5,
    index_filtering=False,
    verbose=True
)

weights2 = search2.find_optimal_weights(coeffs, target)

print(f"\nFinal weights: {weights2}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"With filtering:    {weights}")
print(f"Without filtering: {weights2}")
print(f"Same result: {np.allclose(weights, weights2)}")
