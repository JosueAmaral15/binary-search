"""
DETAILED STEP-BY-STEP CALCULATION
Showing all intermediate values for debugging

Formula from document:
selecionador_absoluto(x,a,b,c,d,j,k,l,m,n,w) = 
    ((a-m)/2) * (sin(π*n*(((j-w)/(2*c))*(abs(x-b+c*k)-abs(x-b-c*(1-k))+c)+w+π*((l*(2/π)-(1/π))/2)))+1) + d + m

Note: The formula has "+1" inside, AFTER the sine!
"""

import math

def selecionador_absoluto_detailed(x, a, b, c, d, j, k, l, m, n, w):
    """Show every step of the calculation"""
    print(f"    Input parameters:")
    print(f"      x={x}, a={a}, b={b}, c={c}, d={d}")
    print(f"      j={j}, k={k}, l={l}, m={m}, n={n}, w={w}")
    print()
    
    # Step 1: Calculate absolute value terms
    term1 = x - b + c*k
    term2 = x - b - c*(1-k)
    print(f"    Step 1: Absolute value terms")
    print(f"      x - b + c*k = {x} - {b} + {c}*{k} = {term1}")
    print(f"      x - b - c*(1-k) = {x} - {b} - {c}*{1-k} = {term2}")
    
    abs_term1 = abs(term1)
    abs_term2 = abs(term2)
    print(f"      |{term1}| = {abs_term1}")
    print(f"      |{term2}| = {abs_term2}")
    
    inner_abs = abs_term1 - abs_term2 + c
    print(f"      inner_abs = {abs_term1} - {abs_term2} + {c} = {inner_abs}")
    print()
    
    # Step 2: Calculate fraction
    fraction = (j - w) / (2 * c)
    print(f"    Step 2: Fraction term")
    print(f"      fraction = (j-w)/(2*c) = ({j}-{w})/(2*{c}) = {fraction}")
    print()
    
    # Step 3: Calculate constant offset
    l_term = l * (2/math.pi) - (1/math.pi)
    constant_offset = math.pi * (l_term / 2)
    print(f"    Step 3: Constant offset")
    print(f"      l * (2/π) - (1/π) = {l} * {2/math.pi:.6f} - {1/math.pi:.6f} = {l_term:.6f}")
    print(f"      π * (l_term / 2) = {math.pi:.6f} * ({l_term:.6f}/2) = {constant_offset:.6f}")
    print()
    
    # Step 4: Calculate sine input
    sine_input = math.pi * n * (fraction * inner_abs + w + constant_offset)
    print(f"    Step 4: Sine input")
    print(f"      fraction * inner_abs = {fraction} * {inner_abs} = {fraction * inner_abs:.6f}")
    print(f"      + w = {fraction * inner_abs:.6f} + {w} = {fraction * inner_abs + w:.6f}")
    print(f"      + constant_offset = {fraction * inner_abs + w:.6f} + {constant_offset:.6f} = {fraction * inner_abs + w + constant_offset:.6f}")
    print(f"      π * n * (...) = {math.pi:.6f} * {n} * {fraction * inner_abs + w + constant_offset:.6f} = {sine_input:.6f}")
    print()
    
    # Step 5: Calculate sine
    sine_value = math.sin(sine_input)
    print(f"    Step 5: Sine calculation")
    print(f"      sin({sine_input:.6f}) = {sine_value:.6f}")
    print()
    
    # Step 6: Calculate amplitude (before adding 1!)
    amplitude = (a - m) / 2
    print(f"    Step 6: Amplitude")
    print(f"      (a-m)/2 = ({a}-{m})/2 = {amplitude:.6f}")
    print()
    
    # Step 7: Calculate sine + 1
    sine_plus_one = sine_value + 1
    print(f"    Step 7: Add 1 to sine")
    print(f"      sin + 1 = {sine_value:.6f} + 1 = {sine_plus_one:.6f}")
    print()
    
    # Step 8: Multiply amplitude by (sine+1)
    multiplied = amplitude * sine_plus_one
    print(f"    Step 8: Multiply amplitude * (sine+1)")
    print(f"      {amplitude:.6f} * {sine_plus_one:.6f} = {multiplied:.6f}")
    print()
    
    # Step 9: Add d and m
    result = multiplied + d + m
    print(f"    Step 9: Final addition")
    print(f"      result = {multiplied:.6f} + {d} + {m} = {result:.6f}")
    print()
    
    return result


print("=" * 100)
print("DETAILED FORMULA CALCULATION - target=28, result=15")
print("=" * 100)
print()

# Given values
target = 28
result = 15
tq = 100

print("SETUP")
print("-" * 100)
print(f"  target = {target}")
print(f"  result = {result}")
print(f"  tq = {tq}")
print()

# Calculate percentage
diff_res = abs(target - result)
percentage = 1 - (diff_res / abs(target))

print("PERCENTAGE CALCULATION")
print("-" * 100)
print(f"  diff_res = |{target} - {result}| = {diff_res}")
print(f"  percentage = 1 - ({diff_res}/{target}) = {percentage}")
print(f"  percentage = {percentage:.10f}")
print()

# Calculate selector and step
selector = math.ceil(0.5 * (abs(percentage) - abs(percentage - 1) + 1))
step = selector * percentage

print("STEP CALCULATION")
print("-" * 100)
print(f"  selector_greater({percentage:.10f}):")
print(f"    = ceil(0.5 * (|{percentage:.10f}| - |{percentage:.10f} - 1| + 1))")
print(f"    = ceil(0.5 * ({abs(percentage):.10f} - {abs(percentage - 1):.10f} + 1))")
print(f"    = ceil(0.5 * {abs(percentage) - abs(percentage - 1) + 1:.10f})")
print(f"    = ceil({0.5 * (abs(percentage) - abs(percentage - 1) + 1):.10f})")
print(f"    = {selector}")
print()
print(f"  step = selector * percentage = {selector} * {percentage:.10f} = {step:.10f}")
print()

# Calculate upper_limit
print("=" * 100)
print("UPPER_LIMIT CALCULATION")
print("=" * 100)
print(f"  Formula: upper_limit(x) = positive_numbers(x, tq, 0, 1, 0, 1, 0, 1, (tq+1)/2, 1, 0)")
print(f"  Parameters: x={step:.10f}, a={tq}, b=0, c=1, d=0, j=1, k=0, l=1, m={(tq+1)/2}, n=1, w=0")
print()

upper_limit = selecionador_absoluto_detailed(
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

print(f"  ✓ upper_limit({step:.10f}) = {upper_limit:.10f}")
print()

# Calculate lower_limit
print("=" * 100)
print("LOWER_LIMIT CALCULATION")
print("=" * 100)
print(f"  Formula: lower_limit(x) = positive_numbers(x, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)")
print(f"  Parameters: x={step:.10f}, a={(tq-1)/2}, b=0, c=1, d=0, j=1, k=0, l=2, m=0, n=1, w=0")
print()

lower_limit = selecionador_absoluto_detailed(
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

print(f"  ✓ lower_limit({step:.10f}) = {lower_limit:.10f}")
print()

# Calculate upper_substractionlimit
print("=" * 100)
print("UPPER_SUBSTRACTIONLIMIT CALCULATION")
print("=" * 100)
c_val = 1 + tq/(tq+1)
print(f"  Formula: upper_substractionlimit(x) = positive_numbers(x, tq, -1, 1+tq/(tq+1), tq/2, 1, 0, 1, 0, 1, 0)")
print(f"  c = 1 + {tq}/{tq+1} = {c_val:.10f}")
print(f"  Parameters: x={step:.10f}, a={tq}, b=-1, c={c_val:.10f}, d={tq/2}, j=1, k=0, l=1, m=0, n=1, w=0")
print()

upper_sub = selecionador_absoluto_detailed(
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

print(f"  ✓ upper_substractionlimit({step:.10f}) = {upper_sub:.10f}")
print()

# Calculate lower_substractionlimit
print("=" * 100)
print("LOWER_SUBSTRACTIONLIMIT CALCULATION")
print("=" * 100)
print(f"  Formula: lower_substractionlimit(x) = positive_numbers(x, tq, -1, 1+tq/(tq+1), -tq/2, 1, 0, 0, 0, 1, 0)")
print(f"  Parameters: x={step:.10f}, a={tq}, b=-1, c={c_val:.10f}, d={-tq/2}, j=1, k=0, l=0, m=0, n=1, w=0")
print()

lower_sub = selecionador_absoluto_detailed(
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

print(f"  ✓ lower_substractionlimit({step:.10f}) = {lower_sub:.10f}")
print()

# Generate sequences
print("=" * 100)
print("SEQUENCE GENERATION")
print("=" * 100)

start_add = math.floor(lower_limit)
end_add = math.ceil(upper_limit)
print(f"  add_numbers:")
print(f"    start = floor({lower_limit:.10f}) = {start_add}")
print(f"    end = ceil({upper_limit:.10f}) = {end_add}")
add_numbers = list(range(start_add, end_add + 1))
print(f"    sequence = [{start_add}, {start_add+1}, ..., {end_add}]")
print(f"    length = {len(add_numbers)}")
print()

start_sub = math.ceil(lower_sub) + 1
end_sub = math.floor(upper_sub) - 1
print(f"  substract_numbers:")
print(f"    start = ceil({lower_sub:.10f}) + 1 = {start_sub}")
print(f"    end = floor({upper_sub:.10f}) - 1 = {end_sub}")

if start_sub <= end_sub:
    substract_numbers = list(range(start_sub, end_sub + 1))
    print(f"    sequence = [{start_sub}, {start_sub+1}, ..., {end_sub}]")
    print(f"    length = {len(substract_numbers)}")
else:
    substract_numbers = []
    print(f"    sequence = [] (empty)")
print()

# Final filtering
index_filtered = [x for x in add_numbers if x not in substract_numbers]

print("=" * 100)
print("FINAL RESULT")
print("=" * 100)
print(f"  index_filtered = Remove(add_numbers, substract_numbers)")
print(f"  length = {len(index_filtered)}")
if len(index_filtered) <= 30:
    print(f"  indices = {index_filtered}")
else:
    print(f"  indices = [{index_filtered[0]}, {index_filtered[1]}, ..., {index_filtered[-2]}, {index_filtered[-1]}]")
print()

print("  Expected from document: [63...76] (14 indices)")
print(f"  Got: length={len(index_filtered)}")
print()
print("=" * 100)
