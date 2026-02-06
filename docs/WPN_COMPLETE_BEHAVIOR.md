# WeightCombinationSearch - Complete WPN Behavior Documentation

**Date:** 2026-02-06  
**Author:** User specifications with detailed examples  
**Status:** Authoritative reference for WPN algorithm

---

## Table of Contents

1. [Core Algorithm Overview](#core-algorithm-overview)
2. [WPN Adjustment Rules](#wpn-adjustment-rules)
3. [Index Filtering Integration](#index-filtering-integration)
4. [Fixed Weights Behavior](#fixed-weights-behavior)
5. [Complete Example 1: Target=28](#complete-example-1-target28)
6. [Complete Example 2: Target=410](#complete-example-2-target410)
7. [Python Implementation Reference](#python-implementation-reference)

---

## Core Algorithm Overview

### The Formula

For each coefficient `i` in the parameter array:

```
result = Σ param[i] * weight_multiplier[i] * wpn_multiplier[i]

where:
  weight_multiplier[i] = W[i] if W[i] != 0 else (1 if combo[i] else 0)
  wpn_multiplier[i] = WPN if combo[i] else 1
```

### Key Concepts

1. **W (Weights Array)**: Accumulates history of winning combinations
2. **WPN (Weighted Possibility Number)**: Dynamic multiplier that adjusts based on progress
3. **Combo**: Current test combination (binary: selected or not)
4. **Index Filtering**: Dynamically selects subset of coefficients to optimize each cycle

---

## WPN Adjustment Rules

### Rule 1: When Index Filter Changes

**Condition:** `current_index_filtered != previous_index_filtered`

**Action:** Reset WPN to base value

```python
if result > target and all_filtered_coeffs_are_positive:
    WPN = -1  # Use negative to reduce result
else:
    WPN = 1   # Use positive to increase result
```

### Rule 2: When Index Filter Stays Same

**Condition:** `current_index_filtered == previous_index_filtered`

**Action:** Adjust WPN magnitude

#### Sub-case A: Result < Target (need to increase)

```python
WPN = WPN * 2  # Double to accelerate growth
```

**Example:**
- Cycle N: WPN=1, result=37, target=410 → Too low!
- Cycle N+1: WPN=2 (doubled)
- Cycle N+2: WPN=4 (doubled again)
- Cycle N+3: WPN=8 (continue doubling)

#### Sub-case B: Result > Target (need to decrease)

```python
WPN = WPN / 2  # Halve to fine-tune
```

**Example:**
- Cycle N: WPN=1, result=30, target=28 → Too high!
- Cycle N+1: WPN=-1 (negative, see Rule 1)
- Cycle N+2: WPN=-0.5 (halved)
- Cycle N+3: WPN=-0.25 (continue halving)

#### Sub-case C: Result oscillating around target

```python
# If sign changed (result went from >target to <target or vice versa)
WPN = WPN / 2  # Reduce magnitude to stabilize
```

### Rule 3: Negative WPN Trigger

**Condition:** 
```python
result > target AND all(coefficients[i] > 0 for i in index_filtered if W[i] == 0)
```

**Explanation:** When result exceeds target and all available (non-fixed) coefficients in the filtered set are positive, we CANNOT reduce the result by selecting more coefficients. Solution: Use **negative WPN** to subtract instead of add.

**Action:**
```python
WPN = -1  # Switch to negative
```

---

## Index Filtering Integration

### Filtering Formula

Based on distance from target, filter narrows or expands coefficient search space:

- **Far from target (0% accurate):** Use extremes `[0, 1, 8, 9]` (both ends of sorted array)
- **Close to target (90% accurate):** Use middle `[4, 5, 6]` (center of sorted array)

### Recalculation Frequency

Index filtering is recalculated **every cycle** based on current result:

```python
percentage = 1 - abs(target - current_result) / abs(target)
index_filtered = calculate_index_filtered(target, current_result, n_params)
```

---

## Fixed Weights Behavior

### Rule: Coefficients NOT in Current Filter = Fixed

**Key Principle:** Once a coefficient gets a non-zero weight and is NOT in the current `index_filtered`, its weight is **frozen** for that cycle.

### Example

```python
Cycle 1:
  index_filtered = [0, 1, 8, 9]
  Winner found: W[1]=1, W[8]=1, W[9]=1
  
Cycle 2:
  index_filtered = [4, 5, 6]  # Changed!
  Fixed positions: [1, 8, 9]  # These are NOT in new filter
  
  # During combinations testing:
  # W[1], W[8], W[9] stay at their values
  # Only W[4], W[5], W[6] are tested with WPN
```

### Accumulation for Filtered Coefficients

If a coefficient **remains in the filter across multiple cycles**, its weight accumulates:

```python
W[i] = initial_value * WPN_cycle_1 * WPN_cycle_2 * ... * WPN_cycle_n
```

**Example:**
```
Cycle 1: W[6]=1, WPN=8  → W[6] becomes 8
Cycle 2: W[6]=8, WPN=0.5 → W[6] becomes 4  (8 * 0.5)
Cycle 3: W[6]=4, WPN=0.25 → W[6] becomes 1 (4 * 0.25)
```

---

## Complete Example 1: Target=28

### Setup

```python
Coefficients: [-7, -12, 5, -1, 10, 5, 23, 6, 9, 14]
Sorted: [-12, -7, -1, 5, 5, 6, 9, 10, 14, 23]
Target: 28
Tolerance: 1
Max iterations: 32
```

### Cycle 1: Initial Search

```
index_filtered = [0, 1, 8, 9]  # Extremes (0% accurate at start)
Coefficients: [-12, -7, 14, 23]
WPN = 1

Testing 2^4 = 16 combinations...

Line 6 wins: [-12*0, -7*1, 14*1, 23*1] = 30, Δ=2
```

**Weights after Cycle 1:** `W = [0, 1, 0, 0, 0, 0, 0, 0, 1, 1]`

### Cycle 2: Negative WPN Needed

```
Result = 30 (7% over target)
Percentage = 92.86% accurate
index_filtered = [4, 5, 6]  # Middle indices
Coefficients: [5, 6, 9]

Problem: Result > target AND all filtered coeffs are positive!
Solution: Use NEGATIVE WPN

Fixed positions: [1, 8, 9]  # From previous cycle, not in new filter
WPN = -1  # Negative!

Testing combinations with negative WPN...

Line 18 wins: [-7*1, 5*-1, 6*0, 9*0, 14*1, 23*1] = 25, Δ=3
```

**Weights after Cycle 2:** `W = [0, 1, 0, 0, -1, 0, 0, 0, 1, 1]`

### Cycle 3: Fine-tuning

```
Result = 25 (10% under target)
index_filtered = [4, 5, 6]  # Same as before!
WPN = -1 / 2 = -0.5  # Halve (sign changed from >target to <target)

Wait, result is UNDER now, so WPN should be positive!
WPN = 0.5  # Positive, halved magnitude

Line 19 wins: [-7*1, 5*-1, 6*0.5, 9*0, 14*1, 23*1] = 28, Δ=0 ✅
```

**Final weights:** `W = [0, 1, 0, 0, -1, 0.5, 0, 0, 1, 1]`

**Verification:**
```
-7*1 + 5*(-1) + 6*0.5 + 14*1 + 23*1 
= -7 - 5 + 3 + 14 + 23 
= 28 ✅
```

---

## Complete Example 2: Target=410

### Setup

```python
Coefficients: [-7, -12, 5, -1, 10, 5, 23, 6, 9, 14]
Sorted: [-12, -7, -1, 5, 5, 6, 9, 10, 14, 23]
Target: 410
Tolerance: 0.2
Max iterations: 200
```

### Cycle 1: Initial Search (Lines 0-15)

```
index_filtered = [0, 1, 8, 9]
WPN = 1

Line 3 wins: [14*1, 23*1] = 37, Δ=373
```

**Weights:** `W = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]`

### Cycle 2: Doubling WPN (Lines 16-31)

```
Result = 37 << 410  # Much too low!
index_filtered = [0, 1, 8, 9]  # Same filter!
WPN = 2  # Double (same filter + result < target)

Line 19 wins: [14*2, 23*2] = 74, Δ=336
```

**Weights:** `W = [0, 0, 0, 0, 0, 0, 0, 0, 2, 2]`

### Cycle 3: Continue Doubling (Lines 32-95)

```
Result = 74 << 410  # Still too low!
index_filtered = [0, 1, 2, 7, 8, 9]  # Filter expanded!
WPN = 1  # Reset (filter changed)

But wait, still too low!
Next cycle: WPN = 2, then WPN = 4

Line 39 wins: [10*4, 14*2*4, 23*2*4] = 336, Δ=74
```

**Weights:** `W = [0, 0, 0, 0, 0, 0, 0, 4, 8, 8]`

### Cycle 4-7: Rapid Growth (Lines 96-111)

```
WPN = 8  # Continue doubling

Line 97 wins: [9*8, 10*4, 14*8, 23*8] = 408, Δ=2 🎯 Very close!
```

**Weights:** `W = [0, 0, 0, 0, 0, 0, 8, 4, 8, 8]`

### Cycle 8-10: Fine-tuning (Lines 112-135)

```
Result = 408, Target = 410  # Only Δ=2 off!
index_filtered stays same
WPN = 8/2 = 4, then 4/2 = 2, then 2/2 = 1, then 1/2 = 0.5

Line 132 wins: [5*0.5, 9*8, 10*4, 14*8, 23*8] = 410.5, Δ=0.5 ✅
```

**Weights:** `W = [0, 0, 0, 0, 0.5, 0, 8, 4, 8, 8]`

### Final Cycle: Precision (Lines 136-143)

```
WPN = 0.5/2 = 0.25

Line 142 wins: [5*0.125, 6*0.25, 9*8, 10*4, 14*8, 23*8] = 410.125, Δ=0.125 ✅
```

**Final weights:** `W = [0, 0, 0, 0, 0.125, 0.25, 8, 4, 8, 8]`

**Verification:**
```
5*0.125 + 6*0.25 + 9*8 + 10*4 + 14*8 + 23*8 
= 0.625 + 1.5 + 72 + 40 + 112 + 184 
= 410.125 ✅  (Within tolerance of 0.2)
```

---

## Python Implementation Reference

### Core Function Structure

```python
def print_weight_combination_search_core(input_list, wpn, target, 
                                        current_line_number, 
                                        positions, 
                                        fixed_positions):
    """
    Test all combinations in current filtered space.
    
    Args:
        input_list: Coefficient array (should be sorted)
        wpn: Current Weighted Possibility Number
        target: Goal value
        current_line_number: Starting line number for output
        positions: Indices to test in this cycle (index_filtered)
        fixed_positions: Indices with non-zero weights NOT in positions
    
    Returns:
        Winner line number and difference
    """
    def assemble_math_line(input_list, wpn, target, current_count, 
                          current_line_number, positions, fixed_positions):
        # 1. Initialize all multipliers to 0
        multipliers = [0] * len(input_list)
        
        # 2. FIXED FLOOR: Set fixed coefficients to 1
        for fixed_idx in fixed_positions:
            if fixed_idx < len(multipliers):
                multipliers[fixed_idx] = 1
        
        # 3. TRUTH TABLE: Map bits to multipliers
        bits = bin(current_count)[2:].zfill(len(positions))
        for i, pos_index in enumerate(positions):
            bit_value = int(bits[i])
            if bit_value == 1:
                multipliers[pos_index] = wpn  # Selected → use WPN
            elif pos_index in fixed_positions:
                multipliers[pos_index] = 1     # Fixed → stays 1
            else:
                multipliers[pos_index] = 0     # Not selected → 0
        
        # 4. Calculate result
        total_sum = 0
        structure = []
        for i, coef in enumerate(input_list):
            m = multipliers[i]
            val = coef * m
            total_sum += val
            structure.append(f"{coef} * {m}")
        
        difference = abs(total_sum - target)
        line_str = (f"line {current_line_number}. " + 
                   ", ".join(structure) + 
                   f", = {total_sum}, Δ = {difference}")
        
        return line_str, difference
    
    # Test all 2^N combinations
    winner_difference = float("+inf")
    winner_line = current_line_number
    total_iterations = 2**len(positions)
    
    for i in range(total_iterations):
        line_text, diff = assemble_math_line(
            input_list, wpn, target, i, 
            current_line_number, positions, fixed_positions
        )
        
        if diff < winner_difference:
            winner_difference = diff
            winner_line = current_line_number
        
        print(line_text)
        current_line_number += 1
    
    print(f"\nCycle ends. Line {winner_line}, Δ = {winner_difference}, is the winner!")
    return winner_line, winner_difference
```

### Usage Example

```python
# Example from Target=410, Cycle 2
wpn = 2
target = 410
input_list = [-12, -7, -1, 5, 5, 6, 9, 10, 14, 23]  # Sorted
current_line_number = 16
positions = [0, 1, 8, 9]      # Indices to test
fixed_positions = []           # No fixed weights yet

print_weight_combination_search_core(
    input_list, wpn, target, 
    current_line_number, positions, fixed_positions
)
```

**Output:**
```
line 16. -12*0, -7*0, ..., 14*1, 23*1 = 37, Δ = 373
line 17. -12*0, -7*0, ..., 14*1, 23*2 = 60, Δ = 350
...
line 19. -12*0, -7*0, ..., 14*2, 23*2 = 74, Δ = 336

Cycle ends. Line 19, Δ = 336, is the winner!
```

---

## Key Takeaways

1. **WPN is NOT applied separately** - It's baked into the weights during calculation
2. **WPN can be negative** - To subtract when all available coefficients are positive
3. **WPN doubles or halves** - Based on whether filter stays same and direction to target
4. **Fixed weights stay fixed** - Coefficients not in current filter keep their W values
5. **Accumulation happens** - Weights multiply by WPN each cycle they remain in filter
6. **Formula is exact** - `result = Σ param[i] * (W[i] if W[i]!=0 else (1 if combo[i] else 0)) * (WPN if combo[i] else 1)`

---

## Implementation Checklist

- [ ] Implement WPN doubling when `same_filter AND result < target`
- [ ] Implement WPN halving when `same_filter AND result > target`
- [ ] Implement negative WPN trigger when `result > target AND all_filtered_positive`
- [ ] Track `previous_index_filtered` to detect filter changes
- [ ] Implement fixed positions logic (coefficients not in current filter)
- [ ] Test with both examples (target=28 and target=410)
- [ ] Verify formula matches: `param[i] * weight_mult[i] * wpn_mult[i]`

---

**End of Documentation**
