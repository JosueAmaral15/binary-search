"""
Calculate add_numbers for negative result scenario:
- coeffs = [1, 7, 4, 9, 3, 4, -2, -7, 11, 20]
- target = 28
- result = -25 (NEGATIVE!)
"""

import math

def selecionador_absoluto(x, a, b, c, d, j, k, l, m, n, w):
    """
    Formula: ((a-m)/2) * (sin(π*n*(((j-w)/(2*c))*(abs(x-b+c*k)-abs(x-b-c*(1-k))+c)+w+π*((l*(2/π)-(1/π))/2)))+1) + d + m
    """
    # Step 1: Absolute value terms
    inner_abs = abs(x - b + c*k) - abs(x - b - c*(1-k)) + c
    
    # Step 2: Fraction
    fraction = (j - w) / (2 * c)
    
    # Step 3: Constant offset
    l_term = l * (2/math.pi) - (1/math.pi)
    constant_offset = math.pi * (l_term / 2)
    
    # Step 4: Sine input
    sine_input = math.pi * n * (fraction * inner_abs + w + constant_offset)
    
    # Step 5: Calculate sine
    sine_value = math.sin(sine_input)
    
    # Step 6: Amplitude
    amplitude = (a - m) / 2
    
    # Step 7: Multiply amplitude * (sine + 1)
    result = amplitude * (sine_value + 1) + d + m
    
    return result


def positive_numbers(x, a, b, c, d, j, k, l, m, n, w):
    return selecionador_absoluto(x, a, b, c, d, j, k, l, m, n, w)


print("=" * 100)
print("SCENARIO: 10 elements, target=28, result=-25 (NEGATIVE!)")
print("=" * 100)
print()

# Given data
coeffs = [1, 7, 4, 9, 3, 4, -2, -7, 11, 20]
target = 28
result = -25  # NEGATIVE!

print("STEP 1: Sort coefficients (smallest to largest)")
print("-" * 100)
print(f"  Original coefficients: {coeffs}")
coeffs_sorted = sorted(coeffs)
print(f"  Sorted coefficients:   {coeffs_sorted}")
print()

# tq = number of coefficients
tq = len(coeffs_sorted)
print("STEP 2: Calculate terms_quantity (tq)")
print("-" * 100)
print(f"  tq = len(coeffs) = {tq}")
print()

# Calculate percentage
print("STEP 3: Calculate percentage")
print("-" * 100)
diff_res = abs(target - result)
percentage = 1 - (diff_res / abs(target))

print(f"  diff_res = |target - result|")
print(f"           = |{target} - ({result})|")
print(f"           = |{target} - {result}|")
print(f"           = |{target - result}|")
print(f"           = {diff_res}")
print()
print(f"  percentage = 1 - (diff_res / |target|)")
print(f"             = 1 - ({diff_res} / |{target}|)")
print(f"             = 1 - ({diff_res} / {abs(target)})")
print(f"             = 1 - {diff_res / abs(target):.10f}")
print(f"             = {percentage:.10f}")
print()

if percentage < 0:
    print(f"  ⚠️  WARNING: Percentage is NEGATIVE: {percentage:.10f}")
    print(f"  This means result is VERY FAR from target (worse than 0%)")
    print()

print(f"  Accuracy: {percentage*100:.2f}%")
print()

# Calculate selector and step
print("STEP 4: Calculate selector_greater and step")
print("-" * 100)

print(f"  selector_greater({percentage:.10f}):")
print(f"    Formula: ceil(0.5 * (|x| - |x-1| + 1))")
print()
print(f"    |percentage| = |{percentage:.10f}| = {abs(percentage):.10f}")
print(f"    |percentage - 1| = |{percentage:.10f} - 1| = |{percentage - 1:.10f}| = {abs(percentage - 1):.10f}")
print()
print(f"    = ceil(0.5 * ({abs(percentage):.10f} - {abs(percentage - 1):.10f} + 1))")
print(f"    = ceil(0.5 * {abs(percentage) - abs(percentage - 1) + 1:.10f})")
print(f"    = ceil({0.5 * (abs(percentage) - abs(percentage - 1) + 1):.10f})")

selector = math.ceil(0.5 * (abs(percentage) - abs(percentage - 1) + 1))
print(f"    = {selector}")
print()

step = selector * percentage
print(f"  step = selector * percentage")
print(f"       = {selector} * {percentage:.10f}")
print(f"       = {step:.10f}")
print()

if step < 0:
    print(f"  ⚠️  WARNING: Step is NEGATIVE: {step:.10f}")
    print()

# Calculate upper_limit
print("=" * 100)
print("STEP 5: Calculate upper_limit(step)")
print("=" * 100)
print(f"  Formula: upper_limit(x) = positive_numbers(x, tq, 0, 1, 0, 1, 0, 1, (tq+1)/2, 1, 0)")
print()

upper_limit_value = positive_numbers(
    x=step,
    a=tq,
    b=0,
    c=1,
    d=0,
    j=1,
    k=0,
    l=1,
    m=(tq+1)/2,
    n=1,
    w=0
)

print(f"  Parameters:")
print(f"    x = {step:.10f}")
print(f"    a = tq = {tq}")
print(f"    m = (tq+1)/2 = {(tq+1)/2}")
print()
print(f"  ✓ upper_limit({step:.10f}) = {upper_limit_value:.10f}")
print()

# Calculate lower_limit
print("=" * 100)
print("STEP 6: Calculate lower_limit(step)")
print("=" * 100)
print(f"  Formula: lower_limit(x) = positive_numbers(x, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)")
print()

lower_limit_value = positive_numbers(
    x=step,
    a=(tq-1)/2,
    b=0,
    c=1,
    d=0,
    j=1,
    k=0,
    l=2,
    m=0,
    n=1,
    w=0
)

print(f"  Parameters:")
print(f"    x = {step:.10f}")
print(f"    a = (tq-1)/2 = {(tq-1)/2}")
print()
print(f"  ✓ lower_limit({step:.10f}) = {lower_limit_value:.10f}")
print()

# Calculate upper_substractionlimit
print("=" * 100)
print("STEP 7: Calculate upper_substractionlimit(step)")
print("=" * 100)
c_val = 1 + tq/(tq+1)
print(f"  Formula: upper_substractionlimit(x) = positive_numbers(x, tq, -1, 1+tq/(tq+1), tq/2, 1, 0, 1, 0, 1, 0)")
print()

upper_sub_value = positive_numbers(
    x=step,
    a=tq,
    b=-1,
    c=c_val,
    d=tq/2,
    j=1,
    k=0,
    l=1,
    m=0,
    n=1,
    w=0
)

print(f"  Parameters:")
print(f"    x = {step:.10f}")
print(f"    a = tq = {tq}")
print(f"    c = 1 + tq/(tq+1) = {c_val:.10f}")
print()
print(f"  ✓ upper_substractionlimit({step:.10f}) = {upper_sub_value:.10f}")
print()

# Calculate lower_substractionlimit
print("=" * 100)
print("STEP 8: Calculate lower_substractionlimit(step)")
print("=" * 100)
print(f"  Formula: lower_substractionlimit(x) = positive_numbers(x, tq, -1, 1+tq/(tq+1), -tq/2, 1, 0, 0, 0, 1, 0)")
print()

lower_sub_value = positive_numbers(
    x=step,
    a=tq,
    b=-1,
    c=c_val,
    d=-tq/2,
    j=1,
    k=0,
    l=0,
    m=0,
    n=1,
    w=0
)

print(f"  Parameters:")
print(f"    x = {step:.10f}")
print(f"    a = tq = {tq}")
print(f"    d = -tq/2 = {-tq/2}")
print()
print(f"  ✓ lower_substractionlimit({step:.10f}) = {lower_sub_value:.10f}")
print()

# Generate add_numbers
print("=" * 100)
print("STEP 9: Generate add_numbers sequence")
print("=" * 100)
print(f"  Formula: add_numbers = Sequence(i, i, floor(lower_limit), ceil(upper_limit), 1)")
print()

start_add = math.floor(lower_limit_value)
end_add = math.ceil(upper_limit_value)

print(f"  start = floor(lower_limit) = floor({lower_limit_value:.10f}) = {start_add}")
print(f"  end   = ceil(upper_limit)  = ceil({upper_limit_value:.10f}) = {end_add}")
print()

if start_add <= end_add:
    add_numbers = list(range(start_add, end_add + 1))
    print(f"  add_numbers = range({start_add}, {end_add + 1})")
    print(f"              = {add_numbers}")
    print(f"  Length: {len(add_numbers)} indices")
else:
    add_numbers = []
    print(f"  ⚠️  WARNING: start > end ({start_add} > {end_add})")
    print(f"  add_numbers = [] (EMPTY)")
print()

# Generate substract_numbers
print("=" * 100)
print("STEP 10: Generate substract_numbers sequence")
print("=" * 100)
print(f"  Formula: substract_numbers = Sequence(i, i, ceil(lower_substractionlimit)+1, floor(upper_substractionlimit)-1, 1)")
print()

start_sub = math.ceil(lower_sub_value) + 1
end_sub = math.floor(upper_sub_value) - 1

print(f"  start = ceil(lower_substractionlimit) + 1 = ceil({lower_sub_value:.10f}) + 1 = {start_sub}")
print(f"  end   = floor(upper_substractionlimit) - 1 = floor({upper_sub_value:.10f}) - 1 = {end_sub}")
print()

if start_sub <= end_sub:
    substract_numbers = list(range(start_sub, end_sub + 1))
    print(f"  substract_numbers = range({start_sub}, {end_sub + 1})")
    print(f"                    = {substract_numbers}")
    print(f"  Length: {len(substract_numbers)} indices")
else:
    substract_numbers = []
    print(f"  substract_numbers = [] (EMPTY, start > end)")
print()

# Calculate index_filtered
print("=" * 100)
print("STEP 11: Calculate index_filtered")
print("=" * 100)
print(f"  Formula: index_filtered = Remove(add_numbers, substract_numbers)")
print()

if len(add_numbers) > 0:
    index_filtered = [x for x in add_numbers if x not in substract_numbers]
    print(f"  index_filtered = {index_filtered}")
    print(f"  Length: {len(index_filtered)} indices")
else:
    index_filtered = []
    print(f"  index_filtered = [] (add_numbers was empty)")
print()

# Analysis
print("=" * 100)
print("ANALYSIS")
print("=" * 100)
print(f"  Sorted coefficients (tq={tq}): {coeffs_sorted}")
print(f"  Target: {target}")
print(f"  Result: {result}")
print(f"  Difference: {diff_res}")
print(f"  Percentage: {percentage*100:.2f}%")
print()

if percentage < 0:
    print("  🔴 Result is EXTREMELY FAR from target (negative percentage)")
    print("     This means: |target - result| > |target|")
    print(f"     Distance {diff_res} is greater than target magnitude {abs(target)}")
print()

print(f"  add_numbers: {add_numbers}")
print(f"  substract_numbers: {substract_numbers}")
print(f"  index_filtered: {index_filtered}")
print()

if len(index_filtered) > 0:
    print("  Which coefficients are selected?")
    for idx in index_filtered:
        if 0 <= idx < tq:
            print(f"    index_filtered[{idx}] → sorted_coeffs[{idx}] = {coeffs_sorted[idx]}")
        else:
            print(f"    index_filtered[{idx}] → OUT OF RANGE (valid: 0-{tq-1})")
    print()
    
    # Check position
    avg_idx = sum(index_filtered) / len(index_filtered)
    print(f"  Average index: {avg_idx:.2f} (midpoint is {tq/2})")
    
    if avg_idx < tq * 0.3:
        print(f"  Location: ⚡ LOWER EXTREME")
    elif avg_idx > tq * 0.7:
        print(f"  Location: ⚡ UPPER EXTREME")
    else:
        print(f"  Location: 🎯 MIDDLE")
else:
    print("  No coefficients selected (empty filtered list)")
print()

print("=" * 100)
print("COMPLETE")
print("=" * 100)
