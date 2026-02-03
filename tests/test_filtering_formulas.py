"""
Manual Testing of Index Filtering Formulas
Study and understand the mathematical behavior before implementation

Goal: Calculate index_filtered when result=15, target=28, tq=100
"""

import numpy as np
import math

# ============================================================================
# FORMULA DEFINITIONS FROM DOCUMENT
# ============================================================================

def selector_greater(x):
    """
    Binary selector: outputs 1 if x > 0, else 0
    Formula: ⌈(1/2)(|x| - |x-1| + 1)⌉
    """
    result = math.ceil(0.5 * (abs(x) - abs(x - 1) + 1))
    return result


def selecionador_absoluto(x, a, b, c, d, j, k, l, m, n, w):
    """
    Core sine-wave selector function
    
    Formula from document:
    s(x) = (a-m)/2 * sin(π*n*((j-w)/(2c) * (|x-b+ck| - |x-b-c(1-k)| + c) + w + π*l*(2/π - 1/π)/2)) + d + m
    
    Parameters:
    - x: input value (step)
    - a, b, c, d: amplitude and position controls
    - j, k, l, m, n, w: wave shaping parameters
    """
    # Inner absolute value term
    inner_abs = abs(x - b + c*k) - abs(x - b - c*(1-k)) + c
    
    # Fraction term
    fraction = (j - w) / (2 * c)
    
    # Constant offset term
    constant_offset = math.pi * l * (2/math.pi - 1/math.pi) / 2
    
    # Full inner term for sine
    sine_input = math.pi * n * (fraction * inner_abs + w + constant_offset)
    
    # Full formula
    amplitude = (a - m) / 2
    result = amplitude * math.sin(sine_input) + d + m
    
    return result


def positive_numbers(x, a, b, c, d, j, k, l, m, n, w):
    """
    Alias for selecionador_absoluto (used for limit calculations)
    """
    return selecionador_absoluto(x, a, b, c, d, j, k, l, m, n, w)


# ============================================================================
# TEST SCENARIO: result=15, target=28, tq=100
# ============================================================================

print("=" * 80)
print("MANUAL FORMULA TESTING - Index Filtering")
print("=" * 80)
print()

# Step 1: Calculate base values
target = 28
result = 15
tq = 100  # terms_quantity

print("STEP 1: Calculate Basic Values")
print("-" * 80)
print(f"  target = {target}")
print(f"  result = {result}")
print(f"  tq (terms_quantity) = {tq}")
print()

# Step 2: Calculate difference and percentage
diff_res = abs(target - result)
percentage = 1 - (diff_res / abs(target))

print("STEP 2: Calculate Proximity")
print("-" * 80)
print(f"  diff_res = |{target} - {result}| = {diff_res}")
print(f"  percentage = 1 - ({diff_res}/{target}) = {percentage:.4f}")
print(f"  Accuracy: {percentage*100:.2f}%")
print()

# Step 3: Calculate selector and step
selector_value = selector_greater(percentage)
step = selector_value * percentage

print("STEP 3: Calculate Step Control")
print("-" * 80)
print(f"  selector_greater({percentage:.4f}) = {selector_value}")
print(f"  step = {selector_value} × {percentage:.4f} = {step:.4f}")
print()

# Step 4: Calculate upper_limit(step)
# From document: upper_limit(x) = positive_numbers(x, tq, 0, 1, 0, 1, 0, 1, (tq+1)/2, 1, 0)
print("STEP 4: Calculate Upper Limit")
print("-" * 80)
print("  Formula: upper_limit(x) = positive_numbers(x, tq, 0, 1, 0, 1, 0, 1, (tq+1)/2, 1, 0)")
print(f"  Parameters: (x={step:.4f}, a={tq}, b=0, c=1, d=0, j=1, k=0, l=1, m={(tq+1)/2}, n=1, w=0)")

upper_limit_value = positive_numbers(
    x=step,
    a=tq,      # 100
    b=0,
    c=1,
    d=0,
    j=1,
    k=0,
    l=1,
    m=(tq+1)/2,  # 50.5
    n=1,
    w=0
)

print(f"  upper_limit({step:.4f}) = {upper_limit_value:.4f}")
print()

# Step 5: Calculate lower_limit(step)
# From document: lower_limit(x) = positive_numbers(x, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)
print("STEP 5: Calculate Lower Limit")
print("-" * 80)
print("  Formula: lower_limit(x) = positive_numbers(x, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)")
print(f"  Parameters: (x={step:.4f}, a={(tq-1)/2}, b=0, c=1, d=0, j=1, k=0, l=2, m=0, n=1, w=0)")

lower_limit_value = positive_numbers(
    x=step,
    a=(tq-1)/2,  # 49.5
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

print(f"  lower_limit({step:.4f}) = {lower_limit_value:.4f}")
print()

# Step 6: Calculate upper_subtractionlimit(step)
# From document: upper_subtractionlimit(x) = positive_numbers(x, tq, -1, 1+tq/(tq+1), tq/2, 1, 0, 1, 0, 1, 0)
print("STEP 6: Calculate Upper Subtraction Limit")
print("-" * 80)
print("  Formula: upper_subtractionlimit(x) = positive_numbers(x, tq, -1, 1+tq/(tq+1), tq/2, 1, 0, 1, 0, 1, 0)")

c_value = 1 + tq/(tq+1)  # 1.99...
print(f"  Parameters: (x={step:.4f}, a={tq}, b=-1, c={c_value:.4f}, d={tq/2}, j=1, k=0, l=1, m=0, n=1, w=0)")

upper_substraction_value = positive_numbers(
    x=step,
    a=tq,      # 100
    b=-1,
    c=c_value,  # 1.99
    d=tq/2,    # 50
    j=1,
    k=0,
    l=1,
    m=0,
    n=1,
    w=0
)

print(f"  upper_subtractionlimit({step:.4f}) = {upper_substraction_value:.4f}")
print()

# Step 7: Calculate lower_subtractionlimit(step)
# From document: lower_subtractionlimit(x) = positive_numbers(x, tq, -1, 1+tq/(tq+1), -tq/2, 1, 0, 0, 0, 1, 0)
print("STEP 7: Calculate Lower Subtraction Limit")
print("-" * 80)
print("  Formula: lower_subtractionlimit(x) = positive_numbers(x, tq, -1, 1+tq/(tq+1), -tq/2, 1, 0, 0, 0, 1, 0)")
print(f"  Parameters: (x={step:.4f}, a={tq}, b=-1, c={c_value:.4f}, d={-tq/2}, j=1, k=0, l=0, m=0, n=1, w=0)")

lower_substraction_value = positive_numbers(
    x=step,
    a=tq,      # 100
    b=-1,
    c=c_value,  # 1.99
    d=-tq/2,   # -50
    j=1,
    k=0,
    l=0,
    m=0,
    n=1,
    w=0
)

print(f"  lower_subtractionlimit({step:.4f}) = {lower_substraction_value:.4f}")
print()

# Step 8: Generate add_numbers sequence
print("STEP 8: Generate add_numbers Sequence")
print("-" * 80)
start_add = math.floor(lower_limit_value)
end_add = math.ceil(upper_limit_value)
print(f"  Start: floor({lower_limit_value:.4f}) = {start_add}")
print(f"  End: ceil({upper_limit_value:.4f}) = {end_add}")
add_numbers = list(range(start_add, end_add + 1))
print(f"  add_numbers = [{start_add}, {start_add+1}, ..., {end_add}]")
print(f"  Length: {len(add_numbers)} indices")
print()

# Step 9: Generate substract_numbers sequence
print("STEP 9: Generate substract_numbers Sequence")
print("-" * 80)
start_sub = math.ceil(lower_substraction_value) + 1
end_sub = math.floor(upper_substraction_value) - 1
print(f"  Start: ceil({lower_substraction_value:.4f}) + 1 = {start_sub}")
print(f"  End: floor({upper_substraction_value:.4f}) - 1 = {end_sub}")

if start_sub <= end_sub:
    substract_numbers = list(range(start_sub, end_sub + 1))
    print(f"  substract_numbers = [{start_sub}, {start_sub+1}, ..., {end_sub}]")
    print(f"  Length: {len(substract_numbers)} indices")
else:
    substract_numbers = []
    print(f"  substract_numbers = [] (empty, start > end)")
print()

# Step 10: Calculate index_filtered
print("STEP 10: Calculate index_filtered (Remove subtraction from addition)")
print("-" * 80)
index_filtered = [x for x in add_numbers if x not in substract_numbers]
print(f"  index_filtered = add_numbers - substract_numbers")
if len(index_filtered) <= 20:
    print(f"  index_filtered = {index_filtered}")
else:
    print(f"  index_filtered = [{index_filtered[0]}, {index_filtered[1]}, ..., {index_filtered[-2]}, {index_filtered[-1]}]")
print(f"  Length: {len(index_filtered)} indices")
print()

# Step 11: Analyze the result
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print(f"  Percentage: {percentage*100:.2f}% (result is {percentage*100:.2f}% close to target)")
print(f"  Filtered indices: {len(index_filtered)} out of {tq} total")
print(f"  Reduction: {(1 - len(index_filtered)/tq)*100:.1f}% fewer indices to optimize")
print()

if len(index_filtered) > 0:
    print("  Filtered index range:")
    print(f"    First index: {min(index_filtered)}")
    print(f"    Last index: {max(index_filtered)}")
    
    # Check if they're at extremes or middle
    mid_point = tq / 2
    avg_index = sum(index_filtered) / len(index_filtered)
    print(f"    Average index: {avg_index:.1f} (midpoint is {mid_point})")
    
    if avg_index < tq * 0.3 or avg_index > tq * 0.7:
        print(f"    Location: ⚡ EXTREMES (far from center)")
    else:
        print(f"    Location: 🎯 MIDDLE (near center)")

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
