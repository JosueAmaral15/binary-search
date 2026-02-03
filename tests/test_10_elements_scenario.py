"""
Calculate add_numbers for specific scenario:
- coeffs = [1, 7, 4, 9, 3, 4, -2, -7, 11, 20]
- target = 28
- result = 25
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
print("SCENARIO: 10 elements, target=28, result=25")
print("=" * 100)
print()

# Given data
coeffs = [1, 7, 4, 9, 3, 4, -2, -7, 11, 20]
target = 28
result = 25

print("STEP 1: Sort coefficients (smallest to largest)")
print("-" * 100)
print(f"  Original coefficients: {coeffs}")
coeffs_sorted = sorted(coeffs)
print(f"  Sorted coefficients:   {coeffs_sorted}")
print()

# Create index mapping
original_indices = {val: idx for idx, val in enumerate(coeffs)}
print("  Original position mapping:")
for i, val in enumerate(coeffs_sorted):
    original_pos = coeffs.index(val)
    print(f"    sorted_index[{i}] = {val} was at original_index[{original_pos}]")
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
print(f"  diff_res = |{target} - {result}| = {diff_res}")
print(f"  percentage = 1 - ({diff_res}/{abs(target)}) = {percentage:.10f}")
print(f"  Accuracy: {percentage*100:.2f}%")
print()

# Calculate selector and step
print("STEP 4: Calculate selector_greater and step")
print("-" * 100)
selector = math.ceil(0.5 * (abs(percentage) - abs(percentage - 1) + 1))
step = selector * percentage

print(f"  selector_greater({percentage:.10f}):")
print(f"    = ceil(0.5 * (|{percentage:.10f}| - |{percentage:.10f} - 1| + 1))")
print(f"    = ceil(0.5 * ({abs(percentage):.10f} - {abs(percentage - 1):.10f} + 1))")
print(f"    = ceil(0.5 * {abs(percentage) - abs(percentage - 1) + 1:.10f})")
print(f"    = ceil({0.5 * (abs(percentage) - abs(percentage - 1) + 1):.10f})")
print(f"    = {selector}")
print()
print(f"  step = selector * percentage")
print(f"       = {selector} * {percentage:.10f}")
print(f"       = {step:.10f}")
print()

# Calculate upper_limit
print("=" * 100)
print("STEP 5: Calculate upper_limit(step)")
print("=" * 100)
print(f"  Formula: upper_limit(x) = positive_numbers(x, tq, 0, 1, 0, 1, 0, 1, (tq+1)/2, 1, 0)")
print()
print(f"  Parameters:")
print(f"    x = {step:.10f}")
print(f"    a = tq = {tq}")
print(f"    b = 0")
print(f"    c = 1")
print(f"    d = 0")
print(f"    j = 1")
print(f"    k = 0")
print(f"    l = 1")
print(f"    m = (tq+1)/2 = ({tq}+1)/2 = {(tq+1)/2}")
print(f"    n = 1")
print(f"    w = 0")
print()

# Detailed calculation for upper_limit
x = step
a = tq
b = 0
c = 1
d = 0
j = 1
k = 0
l = 1
m = (tq+1)/2
n = 1
w = 0

print("  Calculation steps:")
print()

# Inner abs
inner_abs = abs(x - b + c*k) - abs(x - b - c*(1-k)) + c
print(f"    1) inner_abs = |x - b + c*k| - |x - b - c*(1-k)| + c")
print(f"                 = |{x:.10f} - {b} + {c}*{k}| - |{x:.10f} - {b} - {c}*{1-k}| + {c}")
print(f"                 = |{x:.10f}| - |{x - c:.10f}| + {c}")
print(f"                 = {abs(x):.10f} - {abs(x - c):.10f} + {c}")
print(f"                 = {inner_abs:.10f}")
print()

# Fraction
fraction = (j - w) / (2 * c)
print(f"    2) fraction = (j - w) / (2 * c)")
print(f"                = ({j} - {w}) / (2 * {c})")
print(f"                = {fraction:.10f}")
print()

# Constant offset
l_term = l * (2/math.pi) - (1/math.pi)
constant_offset = math.pi * (l_term / 2)
print(f"    3) l_term = l * (2/π) - (1/π)")
print(f"              = {l} * {2/math.pi:.10f} - {1/math.pi:.10f}")
print(f"              = {l_term:.10f}")
print()
print(f"       constant_offset = π * (l_term / 2)")
print(f"                       = {math.pi:.10f} * ({l_term:.10f} / 2)")
print(f"                       = {constant_offset:.10f}")
print()

# Sine input
sine_input = math.pi * n * (fraction * inner_abs + w + constant_offset)
print(f"    4) sine_input = π * n * (fraction * inner_abs + w + constant_offset)")
print(f"                  = {math.pi:.10f} * {n} * ({fraction:.10f} * {inner_abs:.10f} + {w} + {constant_offset:.10f})")
print(f"                  = {math.pi:.10f} * {n} * ({fraction * inner_abs:.10f} + {w} + {constant_offset:.10f})")
print(f"                  = {math.pi:.10f} * {n} * {fraction * inner_abs + w + constant_offset:.10f}")
print(f"                  = {sine_input:.10f}")
print()

# Sine value
sine_value = math.sin(sine_input)
print(f"    5) sin(sine_input) = sin({sine_input:.10f})")
print(f"                       = {sine_value:.10f}")
print()

# Amplitude
amplitude = (a - m) / 2
print(f"    6) amplitude = (a - m) / 2")
print(f"                 = ({a} - {m}) / 2")
print(f"                 = {amplitude:.10f}")
print()

# Final result
upper_limit_value = amplitude * (sine_value + 1) + d + m
print(f"    7) result = amplitude * (sin + 1) + d + m")
print(f"              = {amplitude:.10f} * ({sine_value:.10f} + 1) + {d} + {m}")
print(f"              = {amplitude:.10f} * {sine_value + 1:.10f} + {d} + {m}")
print(f"              = {amplitude * (sine_value + 1):.10f} + {d} + {m}")
print(f"              = {upper_limit_value:.10f}")
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
print(f"    a = (tq-1)/2 = ({tq}-1)/2 = {(tq-1)/2}")
print(f"    b = 0, c = 1, d = 0, j = 1, k = 0, l = 2, m = 0, n = 1, w = 0")
print()
print(f"  ✓ lower_limit({step:.10f}) = {lower_limit_value:.10f}")
print()

# Generate add_numbers
print("=" * 100)
print("STEP 7: Generate add_numbers sequence")
print("=" * 100)
print(f"  Formula: add_numbers = Sequence(i, i, floor(lower_limit), ceil(upper_limit), 1)")
print()

start_add = math.floor(lower_limit_value)
end_add = math.ceil(upper_limit_value)

print(f"  start = floor(lower_limit) = floor({lower_limit_value:.10f}) = {start_add}")
print(f"  end   = ceil(upper_limit)  = ceil({upper_limit_value:.10f}) = {end_add}")
print()

add_numbers = list(range(start_add, end_add + 1))

print(f"  add_numbers = range({start_add}, {end_add + 1})")
print(f"              = {add_numbers}")
print()
print(f"  Length: {len(add_numbers)} indices")
print()

# Analysis
print("=" * 100)
print("ANALYSIS")
print("=" * 100)
print(f"  Sorted coefficients (tq={tq}): {coeffs_sorted}")
print(f"  Result accuracy: {percentage*100:.2f}%")
print(f"  add_numbers: {add_numbers}")
print()

if len(add_numbers) > 0:
    print("  Which coefficients are selected?")
    for idx in add_numbers:
        if 0 <= idx < tq:
            print(f"    add_numbers[{idx}] → sorted_coeffs[{idx}] = {coeffs_sorted[idx]}")
        else:
            print(f"    add_numbers[{idx}] → OUT OF RANGE (valid: 0-{tq-1})")
print()

print("=" * 100)
print("COMPLETE")
print("=" * 100)
