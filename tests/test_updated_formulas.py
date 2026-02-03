"""
UPDATED FORMULAS TEST
New formulas to fix out-of-range indices:
- upper_limit(x) = positive_numbers(x, tq-1, 0, 1, 0, 1, 0, 1, tq/2, 1, 0)  [changed from tq to tq-1]
- upper_substractionlimit(x) = positive_numbers(x, tq-1, -1, 1+tq/(tq+1), (tq-1)/2, 1, 0, 1, 0, 1, 0)  [changed]

Test: result=-25
"""

import math

def selecionador_absoluto(x, a, b, c, d, j, k, l, m, n, w):
    """
    Formula: ((a-m)/2) * (sin(π*n*(((j-w)/(2*c))*(abs(x-b+c*k)-abs(x-b-c*(1-k))+c)+w+π*((l*(2/π)-(1/π))/2)))+1) + d + m
    """
    inner_abs = abs(x - b + c*k) - abs(x - b - c*(1-k)) + c
    fraction = (j - w) / (2 * c)
    l_term = l * (2/math.pi) - (1/math.pi)
    constant_offset = math.pi * (l_term / 2)
    sine_input = math.pi * n * (fraction * inner_abs + w + constant_offset)
    sine_value = math.sin(sine_input)
    amplitude = (a - m) / 2
    result = amplitude * (sine_value + 1) + d + m
    return result

def positive_numbers(x, a, b, c, d, j, k, l, m, n, w):
    return selecionador_absoluto(x, a, b, c, d, j, k, l, m, n, w)


print("=" * 100)
print("UPDATED FORMULAS TEST: result=-25")
print("=" * 100)
print()

# Given data
coeffs = [1, 7, 4, 9, 3, 4, -2, -7, 11, 20]
target = 28
result = -25

print("SETUP")
print("-" * 100)
print(f"  Coefficients: {coeffs}")
coeffs_sorted = sorted(coeffs)
print(f"  Sorted:       {coeffs_sorted}")
print(f"  Valid indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]")
print()

tq = len(coeffs_sorted)
print(f"  tq = {tq}")
print()

# Calculate percentage
diff_res = abs(target - result)
percentage = 1 - (diff_res / abs(target))

print("PERCENTAGE & STEP")
print("-" * 100)
print(f"  diff_res = |{target} - ({result})| = {diff_res}")
print(f"  percentage = 1 - ({diff_res}/{abs(target)}) = {percentage:.10f}")
print(f"  Accuracy: {percentage*100:.2f}%")
print()

selector = math.ceil(0.5 * (abs(percentage) - abs(percentage - 1) + 1))
step = selector * percentage

print(f"  selector_greater({percentage:.10f}) = {selector}")
print(f"  step = {selector} * {percentage:.10f} = {step:.10f}")
print()

# OLD FORMULAS (for comparison)
print("=" * 100)
print("OLD FORMULAS (for comparison)")
print("=" * 100)

print("  OLD: upper_limit(x) = positive_numbers(x, tq, 0, 1, 0, 1, 0, 1, (tq+1)/2, 1, 0)")
old_upper = positive_numbers(step, tq, 0, 1, 0, 1, 0, 1, (tq+1)/2, 1, 0)
print(f"       upper_limit({step:.10f}) = {old_upper:.10f}")
print()

print("  OLD: lower_limit(x) = positive_numbers(x, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)")
old_lower = positive_numbers(step, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)
print(f"       lower_limit({step:.10f}) = {old_lower:.10f}")
print()

old_c = 1 + tq/(tq+1)
print(f"  OLD: upper_substractionlimit(x) = positive_numbers(x, tq, -1, 1+tq/(tq+1), tq/2, 1, 0, 1, 0, 1, 0)")
old_upper_sub = positive_numbers(step, tq, -1, old_c, tq/2, 1, 0, 1, 0, 1, 0)
print(f"       upper_substractionlimit({step:.10f}) = {old_upper_sub:.10f}")
print()

print(f"  OLD: lower_substractionlimit(x) = positive_numbers(x, tq, -1, 1+tq/(tq+1), -tq/2, 1, 0, 0, 0, 1, 0)")
old_lower_sub = positive_numbers(step, tq, -1, old_c, -tq/2, 1, 0, 0, 0, 1, 0)
print(f"       lower_substractionlimit({step:.10f}) = {old_lower_sub:.10f}")
print()

old_start_add = math.floor(old_lower)
old_end_add = math.ceil(old_upper)
old_add = list(range(old_start_add, old_end_add + 1))

old_start_sub = math.ceil(old_lower_sub) + 1
old_end_sub = math.floor(old_upper_sub) - 1
old_sub = list(range(old_start_sub, old_end_sub + 1)) if old_start_sub <= old_end_sub else []

old_filtered = [x for x in old_add if x not in old_sub]

print(f"  OLD Results:")
print(f"    add_numbers = {old_add}")
print(f"    substract_numbers = {old_sub}")
print(f"    index_filtered = {old_filtered}")
print(f"    ⚠️  Contains index {max(old_filtered)} (OUT OF RANGE for 0-{tq-1})")
print()

# NEW FORMULAS
print("=" * 100)
print("NEW FORMULAS (UPDATED)")
print("=" * 100)

print("  NEW: upper_limit(x) = positive_numbers(x, tq-1, 0, 1, 0, 1, 0, 1, tq/2, 1, 0)")
print(f"       Parameters: a={tq-1}, m={tq/2}")
new_upper = positive_numbers(step, tq-1, 0, 1, 0, 1, 0, 1, tq/2, 1, 0)
print(f"       upper_limit({step:.10f}) = {new_upper:.10f}")
print()

print("  NEW: lower_limit(x) = positive_numbers(x, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)")
print(f"       (unchanged)")
new_lower = positive_numbers(step, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)
print(f"       lower_limit({step:.10f}) = {new_lower:.10f}")
print()

new_c = 1 + tq/(tq+1)
print(f"  NEW: upper_substractionlimit(x) = positive_numbers(x, tq-1, -1, 1+tq/(tq+1), (tq-1)/2, 1, 0, 1, 0, 1, 0)")
print(f"       Parameters: a={tq-1}, d={(tq-1)/2}")
new_upper_sub = positive_numbers(step, tq-1, -1, new_c, (tq-1)/2, 1, 0, 1, 0, 1, 0)
print(f"       upper_substractionlimit({step:.10f}) = {new_upper_sub:.10f}")
print()

print(f"  NEW: lower_substractionlimit(x) = positive_numbers(x, tq, -1, 1+tq/(tq+1), -tq/2, 1, 0, 0, 0, 1, 0)")
print(f"       (unchanged)")
new_lower_sub = positive_numbers(step, tq, -1, new_c, -tq/2, 1, 0, 0, 0, 1, 0)
print(f"       lower_substractionlimit({step:.10f}) = {new_lower_sub:.10f}")
print()

# Generate sequences with new formulas
new_start_add = math.floor(new_lower)
new_end_add = math.ceil(new_upper)
new_add = list(range(new_start_add, new_end_add + 1))

new_start_sub = math.ceil(new_lower_sub) + 1
new_end_sub = math.floor(new_upper_sub) - 1
new_sub = list(range(new_start_sub, new_end_sub + 1)) if new_start_sub <= new_end_sub else []

new_filtered = [x for x in new_add if x not in new_sub]

print("=" * 100)
print("SEQUENCES WITH NEW FORMULAS")
print("=" * 100)

print(f"  add_numbers:")
print(f"    start = floor({new_lower:.10f}) = {new_start_add}")
print(f"    end   = ceil({new_upper:.10f}) = {new_end_add}")
print(f"    add_numbers = {new_add}")
print()

print(f"  substract_numbers:")
print(f"    start = ceil({new_lower_sub:.10f}) + 1 = {new_start_sub}")
print(f"    end   = floor({new_upper_sub:.10f}) - 1 = {new_end_sub}")
print(f"    substract_numbers = {new_sub}")
print()

print(f"  index_filtered = Remove(add_numbers, substract_numbers)")
print(f"                 = {new_filtered}")
print()

# Analysis
print("=" * 100)
print("FINAL ANALYSIS")
print("=" * 100)

print(f"  Sorted coefficients: {coeffs_sorted}")
print(f"  Valid indices: 0 to {tq-1}")
print()

print(f"  NEW index_filtered: {new_filtered}")
print()

all_valid = all(0 <= idx < tq for idx in new_filtered)
print(f"  ✓ All indices in valid range [0, {tq-1}]? {all_valid}")
print()

if all_valid:
    print("  Selected coefficients:")
    for idx in new_filtered:
        print(f"    sorted_coeffs[{idx}] = {coeffs_sorted[idx]}")
    print()
    
    # Check if extremes
    min_idx = min(new_filtered)
    max_idx = max(new_filtered)
    avg_idx = sum(new_filtered) / len(new_filtered)
    
    print(f"  Range: [{min_idx}, {max_idx}]")
    print(f"  Average: {avg_idx:.2f} (midpoint is {(tq-1)/2:.1f})")
    
    if min_idx <= 1 and max_idx >= tq-2:
        print(f"  Location: ⚡⚡ BOTH EXTREMES ✓")
    elif avg_idx < tq * 0.3:
        print(f"  Location: ⚡ LOWER EXTREME")
    elif avg_idx > tq * 0.7:
        print(f"  Location: ⚡ UPPER EXTREME")
    else:
        print(f"  Location: 🎯 MIDDLE")
else:
    print("  ❌ ERROR: Some indices out of range!")

print()

# Comparison
print("=" * 100)
print("COMPARISON: OLD vs NEW")
print("=" * 100)
print(f"  OLD index_filtered: {old_filtered}")
print(f"  NEW index_filtered: {new_filtered}")
print()
print(f"  Difference:")
print(f"    OLD has {len(old_filtered)} indices, max={max(old_filtered)} (OUT OF RANGE)")
print(f"    NEW has {len(new_filtered)} indices, max={max(new_filtered)} (VALID)")
print()

print("=" * 100)
print("COMPLETE ✓")
print("=" * 100)
