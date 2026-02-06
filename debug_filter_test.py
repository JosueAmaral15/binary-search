"""
Debug test for WeightCombinationSearch filtering
According to protocol: Add debug prints to understand actual behavior

Test case:
- coefficients: [-7, -12, 5, -1, 10, 5, 23, 6, 9, 14]
- target: 28
- Check what filtering actually does
"""

import numpy as np
import sys
sys.path.insert(0, '/home/josue/Documents/Informática/Programação/Linguagens de programação/Python - My files/Utils/Math/binary_search')

from math_toolkit.binary_search.combinatorial import WeightCombinationSearch

# Test parameters
coefficients = [-7, -12, 5, -1, 10, 5, 23, 6, 9, 14]
target = 28

print("=" * 80)
print("DEBUG TEST: WeightCombinationSearch with Index Filtering")
print("=" * 80)
print(f"\nCoefficients: {coefficients}")
print(f"Target: {target}")
print(f"Number of parameters: {len(coefficients)}")

# Test with filtering enabled
print("\n" + "=" * 80)
print("RUNNING WITH INDEX FILTERING ENABLED")
print("=" * 80)

wcs = WeightCombinationSearch(
    index_filtering=True,
    max_iter=20,
    tolerance=0.01,
    verbose=True  # Enable verbose mode
)

weights = wcs.find_optimal_weights(coefficients, target)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"\nFinal weights: {weights}")

# Calculate final result
final_result = np.sum(np.array(coefficients) * weights)
print(f"Final result: {final_result:.4f}")
print(f"Target: {target}")
print(f"Error: {abs(final_result - target):.4f}")
print(f"Error percentage: {abs(final_result - target) / abs(target) * 100:.2f}%")

# Show which coefficients were selected
selected_indices = [i for i, w in enumerate(weights) if abs(w) > 1e-10]
selected_coeffs = [coefficients[i] for i in selected_indices]
print(f"\nSelected indices: {selected_indices}")
print(f"Selected coefficients: {selected_coeffs}")
print(f"Sum of selected: {sum(selected_coeffs):.4f}")
    
# Check filtering history
if hasattr(wcs, 'history') and 'filtering_history' in wcs.history and wcs.history['filtering_history']:
    print(f"\n{'=' * 80}")
    print("FILTERING HISTORY (What indices were used each cycle)")
    print("=" * 80)
    
    filtering_history = wcs.history['filtering_history']
    for entry in filtering_history:
        cycle = entry['cycle']
        print(f"\nCycle {cycle}:")
        print(f"  Result at start: {entry['result']:.4f}")
        print(f"  Target: {target}")
        print(f"  Percentage: {entry['percentage']:.4f} ({entry['percentage']*100:.2f}%)")
        print(f"  Filtered indices (sorted space): {entry['index_filtered']}")
        print(f"  Number of indices: {entry['num_filtered']}/{len(coefficients)}")
        
        # Show which coefficients are being used (map from sorted space to original)
        sorted_indices_arr = np.argsort(coefficients)
        original_indices = [sorted_indices_arr[i] for i in entry['index_filtered'] if i < len(coefficients)]
        filtered_coeffs = [coefficients[i] for i in original_indices]
        print(f"  Original indices: {original_indices}")
        print(f"  Filtered coefficients: {filtered_coeffs}")
else:
    print("\n⚠️ No filtering history found!")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Sort coefficients to understand the filtering
sorted_indices = np.argsort(coefficients)
sorted_coeffs = [coefficients[i] for i in sorted_indices]

print(f"\nSorted coefficients: {sorted_coeffs}")
print(f"Sorted indices: {list(sorted_indices)}")
print("\nIndex mapping:")
for i, (idx, coef) in enumerate(zip(sorted_indices, sorted_coeffs)):
    print(f"  Position {i} → Original index {idx} → Value {coef}")

print("\n" + "=" * 80)
print("UNDERSTANDING THE FILTER")
print("=" * 80)
print("""
The filtering should work like this:
1. Sort coefficients: smallest to largest
2. Calculate percentage based on how close result is to target
3. Use formula to select indices from sorted list
4. When close to target (high %), use middle indices
5. When far from target (low %), use extreme indices (both ends)
6. Map back to original coefficient positions
""")

print("\n" + "=" * 80)
print("END OF DEBUG TEST")
print("=" * 80)
