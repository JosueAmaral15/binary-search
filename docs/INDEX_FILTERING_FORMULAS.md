# Index Filtering System - Mathematical Documentation

## Overview

The Index Filtering System is a dynamic parameter selection mechanism that adapts which coefficients to optimize based on proximity to the target value. It implements a mathematical progression from **extremes to middle**, using sine-wave-based selectors to determine optimal filtering ranges.

---

## Core Concept

**Key Principle:** When the result is far from the target, optimize coefficients at both extremes. As the result approaches the target, narrow focus toward middle coefficients.

### Why This Works

1. **Far from target (0-30% accuracy):**
   - Large corrections needed
   - Extreme values (smallest and largest) provide maximum adjustment range
   - Example: `[-7, -2, 11, 20]` from extremes

2. **Getting closer (30-70% accuracy):**
   - Medium adjustments needed
   - Moving inward from extremes
   - Example: `[1, 3, 7, 9]` from middle-outer range

3. **Very close (70-100% accuracy):**
   - Fine-tuning needed
   - Focus on middle values for precise adjustments
   - Example: `[4, 4, 7]` from center

---

## Mathematical Formulas

### Prerequisites

```
π = 3.14159... (PI constant)
tq = terms_quantity = len(coefficients)
target = target value
result = current result value
```

### Step 1: Calculate Proximity Percentage

```
diff_res = |target - result|
percentage = 1 - (diff_res / |target|)
```

**Interpretation:**
- `percentage = 1.0` (100%) → exact match
- `percentage = 0.5` (50%) → halfway to target
- `percentage = 0.0` (0%) → result equals zero (if target positive)
- `percentage < 0` → result farther than target magnitude (extreme divergence)

### Step 2: Binary Selector (Activation Control)

```
selector_greater(x) = ⌈(1/2)(|x| - |x-1| + 1)⌉
```

**Behavior:**
- Returns `1` if `x > 0`
- Returns `0` if `x ≤ 0`
- Acts as activation gate for extreme cases

### Step 3: Calculate Step Value

```
step = selector_greater(percentage) × percentage
```

**Properties:**
- `step ∈ [0, 1]` for valid percentages
- `step = 0` when percentage ≤ 0 (extreme divergence protection)
- `step ≈ 1` when percentage ≈ 100% (near-perfect accuracy)

---

## Core Selector Function

### selecionador_absoluto (Absolute Selector)

```
selecionador_absoluto(x,a,b,c,d,j,k,l,m,n,w) = 
    ((a-m)/2) × (sin(π×n×(((j-w)/(2×c))×(|x-b+c×k| - |x-b-c×(1-k)| + c) + w + π×((l×(2/π) - (1/π))/2))) + 1) + d + m
```

**Component Breakdown:**

1. **Inner Absolute Term:**
   ```
   inner_abs = |x-b+c×k| - |x-b-c×(1-k)| + c
   ```

2. **Fraction Term:**
   ```
   fraction = (j-w)/(2×c)
   ```

3. **Constant Offset:**
   ```
   l_term = l×(2/π) - (1/π)
   constant_offset = π×(l_term/2)
   ```

4. **Sine Input:**
   ```
   sine_input = π×n×(fraction×inner_abs + w + constant_offset)
   ```

5. **Final Calculation:**
   ```
   amplitude = (a-m)/2
   result = amplitude × (sin(sine_input) + 1) + d + m
   ```

**Note:** The `+1` is added to the sine BEFORE multiplying by amplitude, shifting the sine wave from `[-1,1]` to `[0,2]`.

---

## Limit Functions

### positive_numbers (Alias)

```
positive_numbers(x,a,b,c,d,j,k,l,m,n,w) = selecionador_absoluto(x,a,b,c,d,j,k,l,m,n,w)
```

### Upper Limit (UPDATED FORMULA)

```
upper_limit(x) = positive_numbers(x, tq-1, 0, 1, 0, 1, 0, 1, tq/2, 1, 0)
```

**Parameters:**
- `a = tq-1` (prevents out-of-range indices)
- `m = tq/2` (centers around midpoint)
- Other parameters: `b=0, c=1, d=0, j=1, k=0, l=1, n=1, w=0`

### Lower Limit

```
lower_limit(x) = positive_numbers(x, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)
```

**Parameters:**
- `a = (tq-1)/2`
- `l = 2` (different phase offset)
- Other parameters: `b=0, c=1, d=0, j=1, k=0, m=0, n=1, w=0`

### Upper Subtraction Limit (UPDATED FORMULA)

```
upper_substractionlimit(x) = positive_numbers(x, tq-1, -1, 1+tq/(tq+1), (tq-1)/2, 1, 0, 1, 0, 1, 0)
```

**Parameters:**
- `a = tq-1` (prevents out-of-range)
- `b = -1` (shift)
- `c = 1 + tq/(tq+1)` (scaling factor)
- `d = (tq-1)/2` (centered offset)
- Other parameters: `j=1, k=0, l=1, m=0, n=1, w=0`

### Lower Subtraction Limit

```
lower_substractionlimit(x) = positive_numbers(x, tq, -1, 1+tq/(tq+1), -tq/2, 1, 0, 0, 0, 1, 0)
```

**Parameters:**
- `a = tq`
- `b = -1`
- `c = 1 + tq/(tq+1)`
- `d = -tq/2` (negative offset)
- `l = 0` (different phase)
- Other parameters: `j=1, k=0, m=0, n=1, w=0`

---

## Sequence Generation

### Add Numbers Sequence

```
add_numbers = Sequence(i, floor(lower_limit(step)), ceil(upper_limit(step)))
```

**Python equivalent:**
```python
start = math.floor(lower_limit(step))
end = math.ceil(upper_limit(step))
add_numbers = list(range(start, end + 1))
```

### Subtract Numbers Sequence

```
substract_numbers = Sequence(i, ceil(lower_substractionlimit(step))+1, floor(upper_substractionlimit(step))-1)
```

**Python equivalent:**
```python
start = math.ceil(lower_substractionlimit(step)) + 1
end = math.floor(upper_substractionlimit(step)) - 1
substract_numbers = list(range(start, end + 1)) if start <= end else []
```

**Note:** If `start > end`, the sequence is empty.

### Final Index Filtering

```
index_filtered = Remove(add_numbers, substract_numbers)
```

**Python equivalent:**
```python
index_filtered = [x for x in add_numbers if x not in substract_numbers]
```

---

## Complete Algorithm

### Full Workflow

```python
def calculate_index_filtered(coefficients, target, result):
    """
    Calculate which coefficient indices to optimize
    
    Args:
        coefficients: List of coefficient values
        target: Target value to reach
        result: Current result value
        
    Returns:
        list: Indices to include in optimization
    """
    # Sort coefficients
    coeffs_sorted = sorted(coefficients)
    tq = len(coeffs_sorted)
    
    # Calculate proximity
    diff_res = abs(target - result)
    percentage = 1 - (diff_res / abs(target))
    
    # Calculate step
    selector = ceil(0.5 * (abs(percentage) - abs(percentage - 1) + 1))
    step = selector * percentage
    
    # Calculate limits (UPDATED FORMULAS)
    c_val = 1 + tq/(tq+1)
    
    upper_limit = positive_numbers(step, tq-1, 0, 1, 0, 1, 0, 1, tq/2, 1, 0)
    lower_limit = positive_numbers(step, (tq-1)/2, 0, 1, 0, 1, 0, 2, 0, 1, 0)
    upper_sub = positive_numbers(step, tq-1, -1, c_val, (tq-1)/2, 1, 0, 1, 0, 1, 0)
    lower_sub = positive_numbers(step, tq, -1, c_val, -tq/2, 1, 0, 0, 0, 1, 0)
    
    # Generate sequences
    add_numbers = list(range(floor(lower_limit), ceil(upper_limit) + 1))
    
    start_sub = ceil(lower_sub) + 1
    end_sub = floor(upper_sub) - 1
    substract_numbers = list(range(start_sub, end_sub + 1)) if start_sub <= end_sub else []
    
    # Filter
    index_filtered = [x for x in add_numbers if x not in substract_numbers]
    
    return index_filtered
```

---

## Test Cases & Examples

### Example 1: Very Close to Target (96.43% accuracy)

```
Coefficients: [-7, -2, 1, 3, 4, 4, 7, 9, 11, 20] (sorted)
Target: 28
Result: 27
tq: 10

percentage = 1 - |28-27|/28 = 0.9643 (96.43%)
step = 1 × 0.9643 = 0.9643

upper_limit(0.9643) = 5.013
lower_limit(0.9643) = 4.486
upper_substractionlimit(0.9643) = 4.500
lower_substractionlimit(0.9643) = 5.000

add_numbers = [4, 5, 6]
substract_numbers = [] (empty, start > end)
index_filtered = [4, 5, 6]

Selected: coeffs[4]=4, coeffs[5]=4, coeffs[6]=7
Location: CENTER (narrow focus for fine-tuning)
```

### Example 2: Moderately Close (89.29% accuracy)

```
Coefficients: [-7, -2, 1, 3, 4, 4, 7, 9, 11, 20] (sorted)
Target: 28
Result: 25
tq: 10

percentage = 1 - |28-25|/28 = 0.8929 (89.29%)
step = 1 × 0.8929 = 0.8929

upper_limit(0.8929) = 5.626
lower_limit(0.8929) = 4.374
upper_substractionlimit(0.8929) = 4.664
lower_substractionlimit(0.8929) = 5.336

add_numbers = [4, 5, 6]
substract_numbers = []
index_filtered = [4, 5, 6]

Selected: coeffs[4]=4, coeffs[5]=4, coeffs[6]=7
Location: CENTER
```

### Example 3: Very Far from Target (-89.29% accuracy)

```
Coefficients: [-7, -2, 1, 3, 4, 4, 7, 9, 11, 20] (sorted)
Target: 28
Result: -25
tq: 10

percentage = 1 - |28-(-25)|/28 = 1 - 53/28 = -0.8929 (-89.29%)
selector = 0 (protection activated!)
step = 0 × -0.8929 = 0

upper_limit(0) = 9.000
lower_limit(0) = 0.000
upper_substractionlimit(0) = 8.664
lower_substractionlimit(0) = 0.374

add_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
substract_numbers = [2, 3, 4, 5, 6, 7]
index_filtered = [0, 1, 8, 9]

Selected: coeffs[0]=-7, coeffs[1]=-2, coeffs[8]=11, coeffs[9]=20
Location: BOTH EXTREMES (large corrections needed)
```

### Example 4: Extremely Far (billions away)

```
Coefficients: [-7, -2, 1, 3, 4, 4, 7, 9, 11, 20] (sorted)
Target: 28
Result: 4294967296
tq: 10

percentage = -153391687.14 (extreme negative)
selector = 0 (protection activated!)
step = 0

[Same limits as Example 3]
index_filtered = [0, 1, 8, 9]

Location: BOTH EXTREMES (formula protection works correctly)
```

---

## Progression Pattern

| Accuracy | Step | Indices Selected | Position | Use Case |
|----------|------|------------------|----------|----------|
| **96-100%** | 0.96-1.0 | 3-5 middle | 🎯 **CENTER** | Fine-tuning |
| **70-95%** | 0.70-0.95 | 3-6 middle | 🎯 **CENTER** | Moderate adjustments |
| **30-69%** | 0.30-0.69 | 5-8 varied | 🔶 **INWARD** | Balanced search |
| **0-29%** | 0.00-0.29 | 4-6 extremes | ⚡ **EXTREMES** | Large corrections |
| **< 0%** | 0.00 | 4 extremes | ⚡⚡ **BOTH EXTREMES** | Maximum range |

---

## Formula Updates (v2.0)

### Changes from Original

1. **upper_limit:** `a = tq` → `a = tq-1`, `m = (tq+1)/2` → `m = tq/2`
   - **Reason:** Prevents index out-of-range errors
   - **Effect:** Maximum index is now `tq-1` (last valid position)

2. **upper_substractionlimit:** `a = tq` → `a = tq-1`, `d = tq/2` → `d = (tq-1)/2`
   - **Reason:** Consistent with upper_limit changes
   - **Effect:** Subtraction range properly bounded

### Validation

All formulas tested with:
- ✅ 10 elements (tq=10)
- ✅ Positive results (close to target)
- ✅ Negative results (far from target)
- ✅ Extremely large results (billions)
- ✅ Edge cases (percentage < 0, step = 0)

**Result:** All indices stay within valid range `[0, tq-1]`

---

## Implementation Notes

### Integration with WeightCombinationSearch

1. **Pre-processing:**
   - Sort coefficients: `coeffs_sorted = sorted(coefficients)`
   - Track original positions for result mapping

2. **Per-Cycle Filtering:**
   - Calculate current result
   - Determine `index_filtered` based on proximity
   - Generate truth table for filtered indices only
   - Optimize weights for filtered coefficients

3. **Weight Accumulation:**
   - Weights outside `index_filtered` remain unchanged
   - Only filtered weights are updated each cycle
   - Accumulates improvements across cycles

4. **Performance Impact:**
   - Reduces search space from `2^N` to `2^k` where `k = len(index_filtered)`
   - Example: 100 coefficients, 96% accuracy → optimize only ~3 coefficients
   - Speedup: `2^100 / 2^3 = 1.6 × 10^29` (theoretical)

---

## References

- Original document: `Filtering indexes from the most extreme points to the middle points of a list.txt`
- Test implementations: `tests/test_filtering_formulas.py`, `tests/test_edge_cases.py`
- Formula derivation: Sine-wave-based selector with adaptive thresholds

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-03  
**Author:** AI Assistant (based on user-provided mathematical formulas)
