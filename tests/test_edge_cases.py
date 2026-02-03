"""
Test two more edge cases:
1. result = 27 (very close to target=28, should be ~96% accurate)
2. result = 4294967296 (extremely large, very far from target)
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

def calculate_index_filtered(target, result, tq):
    """Calculate index_filtered using UPDATED formulas"""
    # Calculate percentage
    diff_res = abs(target - result)
    percentage = 1 - (diff_res / abs(target))
    
    # Calculate step
    selector = math.ceil(0.5 * (abs(percentage) - abs(percentage - 1) + 1))
    step = selector * percentage
    
    # UPDATED FORMULAS (fixed out-of-range issue)
    # upper_limit: changed from tq to tq-1, m from (tq+1)/2 to tq/2
    upper_limit_val = positive_numbers(step, tq-1, 0, 1, 0, 1, 0, 1, tq/2, 1, 0)
    
    # lower_limit: unchanged
    lower_limit_val = positive_numbers(step, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)
    
    # upper_substractionlimit: changed from tq to tq-1, d from tq/2 to (tq-1)/2
    c_val = 1 + tq/(tq+1)
    upper_sub_val = positive_numbers(step, tq-1, -1, c_val, (tq-1)/2, 1, 0, 1, 0, 1, 0)
    
    # lower_substractionlimit: unchanged
    lower_sub_val = positive_numbers(step, tq, -1, c_val, -tq/2, 1, 0, 0, 0, 1, 0)
    
    # Generate sequences
    start_add = math.floor(lower_limit_val)
    end_add = math.ceil(upper_limit_val)
    add_numbers = list(range(start_add, end_add + 1))
    
    start_sub = math.ceil(lower_sub_val) + 1
    end_sub = math.floor(upper_sub_val) - 1
    substract_numbers = list(range(start_sub, end_sub + 1)) if start_sub <= end_sub else []
    
    # Remove
    index_filtered = [x for x in add_numbers if x not in substract_numbers]
    
    return {
        'percentage': percentage,
        'selector': selector,
        'step': step,
        'upper_limit': upper_limit_val,
        'lower_limit': lower_limit_val,
        'upper_sub': upper_sub_val,
        'lower_sub': lower_sub_val,
        'add_numbers': add_numbers,
        'substract_numbers': substract_numbers,
        'index_filtered': index_filtered
    }

def print_test_result(test_name, target, result, coeffs_sorted):
    print("=" * 100)
    print(f"TEST: {test_name}")
    print("=" * 100)
    print()
    
    tq = len(coeffs_sorted)
    
    print(f"  Target: {target}")
    print(f"  Result: {result}")
    print(f"  tq: {tq}")
    print()
    
    data = calculate_index_filtered(target, result, tq)
    
    print(f"  Difference: |{target} - {result}| = {abs(target - result)}")
    print(f"  Percentage: {data['percentage']:.10f} ({data['percentage']*100:.2f}%)")
    print(f"  Selector: {data['selector']}")
    print(f"  Step: {data['step']:.10f}")
    print()
    
    print(f"  Limits:")
    print(f"    upper_limit = {data['upper_limit']:.6f}")
    print(f"    lower_limit = {data['lower_limit']:.6f}")
    print(f"    upper_substractionlimit = {data['upper_sub']:.6f}")
    print(f"    lower_substractionlimit = {data['lower_sub']:.6f}")
    print()
    
    print(f"  Sequences:")
    if len(data['add_numbers']) <= 20:
        print(f"    add_numbers = {data['add_numbers']}")
    else:
        print(f"    add_numbers = [{data['add_numbers'][0]}, {data['add_numbers'][1]}, ..., {data['add_numbers'][-2]}, {data['add_numbers'][-1]}] (length={len(data['add_numbers'])})")
    
    if len(data['substract_numbers']) <= 20:
        print(f"    substract_numbers = {data['substract_numbers']}")
    else:
        print(f"    substract_numbers = [{data['substract_numbers'][0]}, {data['substract_numbers'][1]}, ..., {data['substract_numbers'][-2]}, {data['substract_numbers'][-1]}] (length={len(data['substract_numbers'])})")
    
    print()
    
    if len(data['index_filtered']) <= 20:
        print(f"  index_filtered = {data['index_filtered']}")
    else:
        print(f"  index_filtered = [{data['index_filtered'][0]}, {data['index_filtered'][1]}, ..., {data['index_filtered'][-2]}, {data['index_filtered'][-1]}]")
    
    print(f"  Length: {len(data['index_filtered'])} indices")
    print()
    
    # Validation
    all_valid = all(0 <= idx < tq for idx in data['index_filtered'])
    print(f"  ✓ All indices in valid range [0, {tq-1}]? {all_valid}")
    
    if not all_valid:
        out_of_range = [idx for idx in data['index_filtered'] if idx < 0 or idx >= tq]
        print(f"  ❌ Out of range indices: {out_of_range}")
    print()
    
    # Selected coefficients
    if all_valid and len(data['index_filtered']) > 0:
        print(f"  Selected coefficients:")
        for idx in data['index_filtered'][:5]:  # Show first 5
            print(f"    sorted_coeffs[{idx}] = {coeffs_sorted[idx]}")
        if len(data['index_filtered']) > 5:
            print(f"    ... ({len(data['index_filtered']) - 5} more)")
        print()
        
        # Position analysis
        min_idx = min(data['index_filtered'])
        max_idx = max(data['index_filtered'])
        avg_idx = sum(data['index_filtered']) / len(data['index_filtered'])
        
        print(f"  Position Analysis:")
        print(f"    Range: [{min_idx}, {max_idx}]")
        print(f"    Average: {avg_idx:.2f} (midpoint is {(tq-1)/2:.1f})")
        
        # Determine location
        if min_idx <= 1 and max_idx >= tq-2:
            location = "⚡⚡ BOTH EXTREMES"
        elif max_idx < tq * 0.3:
            location = "⚡ LOWER EXTREME"
        elif min_idx > tq * 0.7:
            location = "⚡ UPPER EXTREME"
        elif avg_idx < tq * 0.4 or avg_idx > tq * 0.6:
            location = "🔶 MOVING INWARD"
        else:
            location = "🎯 CENTER/MIDDLE"
        
        print(f"    Location: {location}")
    
    print()

# Setup
coeffs = [1, 7, 4, 9, 3, 4, -2, -7, 11, 20]
coeffs_sorted = sorted(coeffs)
target = 28

print("=" * 100)
print("EDGE CASE TESTING")
print("=" * 100)
print()
print(f"Coefficients: {coeffs}")
print(f"Sorted: {coeffs_sorted}")
print(f"Target: {target}")
print()

# Test 1: result = 27 (very close, ~96%)
print_test_result("result = 27 (very close to target)", target, 27, coeffs_sorted)

# Test 2: result = 4294967296 (extremely large)
print_test_result("result = 4294967296 (extremely far)", target, 4294967296, coeffs_sorted)

# Summary
print("=" * 100)
print("SUMMARY OF ALL TESTS")
print("=" * 100)
print()

test_cases = [
    (25, "89.29%", "Middle [4,5,6]"),
    (-25, "-89.29%", "Both extremes [0,1,8,9]"),
    (27, "~96%", "See above"),
    (4294967296, "Very negative", "See above")
]

print("  All tested scenarios:")
for res, acc, loc in test_cases:
    print(f"    result={res:>12} | accuracy={acc:>12} | location: {loc}")

print()
print("=" * 100)
print("COMPLETE")
print("=" * 100)
