"""
Test filtering formulas with different percentage values
to understand the extreme → middle progression
"""

import numpy as np
import math

def selector_greater(x):
    result = math.ceil(0.5 * (abs(x) - abs(x - 1) + 1))
    return result

def selecionador_absoluto(x, a, b, c, d, j, k, l, m, n, w):
    inner_abs = abs(x - b + c*k) - abs(x - b - c*(1-k)) + c
    fraction = (j - w) / (2 * c)
    constant_offset = math.pi * l * (2/math.pi - 1/math.pi) / 2
    sine_input = math.pi * n * (fraction * inner_abs + w + constant_offset)
    amplitude = (a - m) / 2
    result = amplitude * math.sin(sine_input) + d + m
    return result

def positive_numbers(x, a, b, c, d, j, k, l, m, n, w):
    return selecionador_absoluto(x, a, b, c, d, j, k, l, m, n, w)

def calculate_filtered_indices(target, result, tq):
    """Calculate index_filtered for given result"""
    diff_res = abs(target - result)
    percentage = 1 - (diff_res / abs(target))
    selector_value = selector_greater(percentage)
    step = selector_value * percentage
    
    # Upper limit
    upper_limit = positive_numbers(step, tq, 0, 1, 0, 1, 0, 1, (tq+1)/2, 1, 0)
    
    # Lower limit
    lower_limit = positive_numbers(step, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)
    
    # Subtraction limits
    c_val = 1 + tq/(tq+1)
    upper_sub = positive_numbers(step, tq, -1, c_val, tq/2, 1, 0, 1, 0, 1, 0)
    lower_sub = positive_numbers(step, tq, -1, c_val, -tq/2, 1, 0, 0, 0, 1, 0)
    
    # Generate sequences
    add_numbers = list(range(math.floor(lower_limit), math.ceil(upper_limit) + 1))
    start_sub = math.ceil(lower_sub) + 1
    end_sub = math.floor(upper_sub) - 1
    
    if start_sub <= end_sub:
        substract_numbers = list(range(start_sub, end_sub + 1))
    else:
        substract_numbers = []
    
    # Remove
    index_filtered = [x for x in add_numbers if x not in substract_numbers]
    
    return {
        'percentage': percentage,
        'step': step,
        'upper_limit': upper_limit,
        'lower_limit': lower_limit,
        'upper_sub': upper_sub,
        'lower_sub': lower_sub,
        'add_count': len(add_numbers),
        'sub_count': len(substract_numbers),
        'filtered': index_filtered,
        'filtered_count': len(index_filtered)
    }

# Test different scenarios
print("=" * 80)
print("PROGRESSION TEST: Extreme → Middle")
print("=" * 80)
print()

target = 28
tq = 100

test_cases = [
    (0, "Very far (0%)"),
    (10, "Getting closer (30%)"),
    (15, "Halfway (53%)"),
    (20, "Close (71%)"),
    (25, "Very close (89%)"),
    (28, "Exact match (100%)")
]

for result, description in test_cases:
    data = calculate_filtered_indices(target, result, tq)
    
    print(f"Result = {result} | {description}")
    print(f"  Percentage: {data['percentage']*100:.1f}%")
    print(f"  Step: {data['step']:.4f}")
    print(f"  Limits: lower={data['lower_limit']:.2f}, upper={data['upper_limit']:.2f}")
    print(f"  Subtraction: lower={data['lower_sub']:.2f}, upper={data['upper_sub']:.2f}")
    print(f"  Sequences: add={data['add_count']}, sub={data['sub_count']}, filtered={data['filtered_count']}")
    
    if len(data['filtered']) > 0:
        first = min(data['filtered'])
        last = max(data['filtered'])
        avg = sum(data['filtered']) / len(data['filtered'])
        print(f"  Filtered range: [{first} to {last}], avg={avg:.1f}")
    else:
        print(f"  Filtered range: EMPTY")
    
    print()

print("=" * 80)
