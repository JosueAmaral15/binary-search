# WeightCombinationSearch Optimization Strategies - Investigation

**Date:** 2026-02-01  
**Problem:** WeightCombinationSearch is 2,853x slower than AdamW at 20 parameters (70s vs 0.025s)  
**Goal:** Make it practical for 10-20+ parameters without losing accuracy

---

## 🔍 CURRENT BOTTLENECK ANALYSIS

### Performance at Different Scales

| Parameters | Combinations (2^N-1) | Time per Cycle | Total Time (50 iter) | Status |
|------------|---------------------|----------------|----------------------|--------|
| 3 | 7 | <1ms | 0.935ms | ✅ FAST |
| 5 | 31 | <1ms | 1.602ms | ✅ FAST |
| 8 | 255 | 1ms | 3.761ms | ✅ FAST |
| 10 | 1,023 | 3ms | 29.842ms | ⚠️ OK |
| 12 | 4,095 | 8ms | 84.949ms | ⚠️ SLOW |
| 15 | 32,767 | 33ms | 1638.899ms | ❌ VERY SLOW |
| 20 | 1,048,575 | 1400ms | **70,580ms** | ❌ IMPRACTICAL |

### Where Time is Spent (20 params profiling estimate)

```
Total: 70.58 seconds per run

1. Generate combinations        ~5% (3.5s)   - itertools.product()
2. Test each combination        ~85% (60s)   - _calculate_result() × 1M times
3. Find winner per cycle        ~8% (5.5s)   - min() across 1M results
4. Update weights              ~2% (1.4s)    - WPN adjustment
```

**Critical bottleneck:** Testing 1,048,575 combinations × 50 iterations = **52 MILLION evaluations**

---

## 💡 OPTIMIZATION IDEAS - RANKED BY IMPACT

### 🥇 TIER 1: ALGORITHMIC IMPROVEMENTS (10-100x speedup)

#### 1. **Adaptive Sampling Strategy** ⭐⭐⭐⭐⭐
**Idea:** Don't test ALL 2^N combinations - sample intelligently

**Strategy A: Importance Sampling**
```python
# Instead of testing all 1M combos, sample based on:
# - Previous winners (high probability)
# - Random exploration (diversity)
# - Gradient hints (which params matter)

if n_params <= 10:
    # Full enumeration (fast anyway)
    combos = all_combinations(n_params)
else:
    # Adaptive sampling
    sample_size = min(10000, 2**(n_params-5))  # Cap at 10k combos
    combos = smart_sample(n_params, sample_size, previous_winners)
```

**Expected speedup:** 100x for 20 params (1M → 10k combos)  
**Accuracy trade-off:** May miss optimal solution, but finds "good enough" fast  
**Implementation:** Medium complexity

---

#### 2. **Hierarchical/Divide-and-Conquer** ⭐⭐⭐⭐⭐
**Idea:** Split N parameters into groups, solve separately, then combine

**Strategy B: Group-Based Search**
```python
# Example: 20 params → 4 groups of 5 params
# Each group: 2^5 = 32 combos (FAST)
# Then combine best from each group

groups = [params[0:5], params[5:10], params[10:15], params[15:20]]

# Phase 1: Solve each group independently (parallel!)
group_weights = []
for group in groups:
    w = solve_small(group, target/4)  # Fast: 2^5 = 32 combos
    group_weights.append(w)

# Phase 2: Fine-tune combined solution
final_weights = refine(group_weights, target)
```

**Expected speedup:** 50-100x for 20 params  
**Accuracy trade-off:** May miss cross-group interactions  
**Implementation:** Medium complexity

---

#### 3. **Greedy Initialization + Local Refinement** ⭐⭐⭐⭐
**Idea:** Start with greedy solution, then refine locally

**Strategy C: Smart Initialization**
```python
# Phase 1: Greedy selection (O(N) instead of O(2^N))
def greedy_init(coefficients, target):
    W = np.zeros(len(coefficients))
    remaining = target
    
    # Sort by coefficient value
    sorted_idx = np.argsort(np.abs(coefficients))[::-1]
    
    for i in sorted_idx:
        # Greedily select if it helps
        if abs(remaining - coefficients[i]) < abs(remaining):
            W[i] = 1.0
            remaining -= coefficients[i]
    
    return W

# Phase 2: Local refinement (test neighbors only)
def local_refine(W, coefficients, target):
    # Only test flipping each bit (N combos instead of 2^N)
    for i in range(len(W)):
        W_test = W.copy()
        W_test[i] = 1 - W_test[i]  # Flip
        
        if better(W_test):
            W = W_test
    return W
```

**Expected speedup:** 10-50x  
**Accuracy trade-off:** May get stuck in local optimum  
**Implementation:** Easy

---

#### 4. **Early Convergence Detection** ⭐⭐⭐⭐
**Idea:** Stop testing combos once we find good enough solution in a cycle

**Strategy D: Intra-Cycle Stopping**
```python
# Inside cycle loop
best_delta = float('inf')

for combo in combos:
    result = calculate_result(...)
    delta = abs(result - target)
    
    # EARLY STOP: If we found excellent solution this cycle
    if delta <= tolerance * 0.1:  # 10x better than needed
        break  # Don't test remaining combos
    
    if delta < best_delta:
        best_delta = delta
        winner = combo
```

**Expected speedup:** 2-5x average (depends on problem)  
**Accuracy trade-off:** None (still converges correctly)  
**Implementation:** Very easy

---

### 🥈 TIER 2: COMPUTATIONAL OPTIMIZATIONS (2-10x speedup)

#### 5. **Parallel Processing (Multiprocessing)** ⭐⭐⭐⭐
**Idea:** Test combinations in parallel across CPU cores

**Strategy E: Parallel Evaluation**
```python
from multiprocessing import Pool
import os

def evaluate_batch(combo_batch):
    return [calculate_result(c) for c in combo_batch]

# Split 1M combos across 4 cores
n_cores = os.cpu_count()
combo_chunks = np.array_split(combos, n_cores)

with Pool(n_cores) as pool:
    results = pool.map(evaluate_batch, combo_chunks)
```

**Expected speedup:** 3-4x (on 4-core CPU)  
**Accuracy trade-off:** None  
**Implementation:** Easy (built-in multiprocessing)

---

#### 6. **Numba JIT Compilation** ⭐⭐⭐⭐
**Idea:** Compile hot loop with numba for C-like speed

**Strategy F: JIT Compilation**
```python
from numba import jit

@jit(nopython=True)
def calculate_result_fast(coefficients, W, combo, WPN):
    result = 0.0
    for i in range(len(coefficients)):
        weight = W[i] if W[i] != 0 else (1.0 if combo[i] else 0.0)
        multiplier = WPN if combo[i] else 1.0
        result += coefficients[i] * weight * multiplier
    return result
```

**Expected speedup:** 10-50x (Python → C speed)  
**Accuracy trade-off:** None  
**Implementation:** Easy (just add decorator)

---

#### 7. **Vectorized NumPy Operations** ⭐⭐⭐
**Idea:** Use NumPy broadcasting instead of loops

**Strategy G: Vectorization**
```python
# Current: Loop over each combo
for combo in combos:
    result = sum(coef[i] * weight[i] * mult[i] for i in range(N))

# Optimized: Vectorize entire batch
combos_array = np.array(combos)  # Shape: (2^N, N)
weights_matrix = np.where(
    W != 0, 
    W, 
    np.where(combos_array, 1.0, 0.0)
)
multipliers = np.where(combos_array, WPN, 1.0)

# Single matrix multiplication!
results = np.sum(coefficients * weights_matrix * multipliers, axis=1)
```

**Expected speedup:** 3-5x  
**Accuracy trade-off:** None  
**Implementation:** Medium (rewrite calculation logic)

---

### 🥉 TIER 3: MEMORY/STORAGE OPTIMIZATIONS (1-2x speedup)

#### 8. **Lazy Combination Generation** ⭐⭐⭐
**Idea:** Generate combos on-the-fly instead of storing all in memory

```python
# Current: Store all combos (memory intensive)
combos = list(itertools.product([False, True], repeat=20))  # 8 MB

# Optimized: Generator (zero memory)
combos = itertools.product([False, True], repeat=20)  # Iterator
```

**Expected speedup:** 1.5x (less memory thrashing)  
**Implementation:** Very easy

---

#### 9. **Skip Truth Table Logging** ⭐⭐
**Idea:** Only store truth table if explicitly requested

```python
if return_truth_table or save_csv:
    truth_table.append(...)  # Only if needed
```

**Expected speedup:** 1.2x (less memory allocation)  
**Implementation:** Very easy (already partially done)

---

## 🚀 RECOMMENDED IMPLEMENTATION PLAN

### Phase 1: Quick Wins (1-2 days) - Target: 10x speedup

1. ✅ **Numba JIT** (#6) - 10-50x on hot loop
2. ✅ **Early Stopping** (#4) - 2-5x average
3. ✅ **Lazy Generation** (#8) - 1.5x

**Expected result:** 20 params: 70s → 5-7s

---

### Phase 2: Major Optimizations (3-5 days) - Target: 50x total

4. ✅ **Parallel Processing** (#5) - 3-4x additional
5. ✅ **Vectorization** (#7) - 3-5x additional
6. ✅ **Adaptive Sampling** (#1) - 10-100x for large N

**Expected result:** 20 params: 70s → 0.5-1s

---

### Phase 3: Advanced Algorithms (1-2 weeks) - Target: 100x+ total

7. ✅ **Hierarchical Search** (#2) - 50-100x
8. ✅ **Greedy Init** (#3) - 10-50x

**Expected result:** 20 params: 70s → 0.1-0.5s (competitive with AdamW!)

---

## 📊 EXPECTED PERFORMANCE AFTER OPTIMIZATIONS

### Conservative Estimates

| Parameters | Current | Phase 1 (Quick) | Phase 2 (Major) | Phase 3 (Advanced) |
|------------|---------|-----------------|-----------------|-------------------|
| 10 | 29.8ms | 3ms | 1ms | 0.5ms |
| 12 | 84.9ms | 8ms | 3ms | 1ms |
| 15 | 1.64s | 164ms | 50ms | 10ms |
| 20 | 70.6s | **7s** | **1.5s** | **0.5s** ⭐ |
| 25 | ~30min | **3min** | **40s** | **5s** |

### Aggressive Estimates (if all optimizations work well)

| Parameters | Current | Best Case |
|------------|---------|-----------|
| 20 | 70.6s | **0.1-0.5s** ✨ |
| 25 | ~30min | **1-5s** ✨ |

---

## ⚠️ TRADE-OFFS TO CONSIDER

### Accuracy vs Speed

| Optimization | Accuracy Impact | Speed Gain |
|--------------|----------------|------------|
| Numba/Parallel/Vector | ✅ Zero | Moderate |
| Early Stopping | ✅ Zero | Moderate |
| Adaptive Sampling | ⚠️ May miss optimal | Huge |
| Hierarchical Search | ⚠️ May miss interactions | Huge |
| Greedy Init | ⚠️ Local optimum risk | Large |

**Recommendation:** Start with zero-accuracy-loss optimizations, then add sampling/hierarchical with user flags

---

## 🎯 HYBRID APPROACH (BEST OF BOTH WORLDS)

```python
class WeightCombinationSearch:
    def find_optimal_weights(self, coefficients, target, 
                           strategy='auto'):  # NEW PARAMETER
        
        n_params = len(coefficients)
        
        if strategy == 'auto':
            if n_params <= 10:
                strategy = 'exact'
            elif n_params <= 15:
                strategy = 'parallel'
            else:
                strategy = 'adaptive'
        
        if strategy == 'exact':
            # Full enumeration (current algorithm)
            return self._full_search(...)
        
        elif strategy == 'parallel':
            # Full enumeration + Numba + Multiprocessing
            return self._parallel_search(...)
        
        elif strategy == 'adaptive':
            # Adaptive sampling + Hierarchical
            return self._adaptive_search(...)
        
        elif strategy == 'greedy':
            # Greedy init + local refinement
            return self._greedy_search(...)
```

---

## 📝 RECOMMENDED NEXT STEPS

1. **Prototype Numba optimization** (1 hour) - Test on 20 params
2. **Implement early stopping** (30 min) - Easy win
3. **Add parallel processing** (2 hours) - Test on 4-core machine
4. **Benchmark Phase 1** (30 min) - Measure actual speedup
5. **If successful:** Implement adaptive sampling (1-2 days)
6. **If needed:** Implement hierarchical search (3-5 days)

---

## 🔬 TESTING STRATEGY

For each optimization:

1. **Correctness test:** Verify same results as original
2. **Speed test:** Measure actual speedup (3, 10, 15, 20 params)
3. **Accuracy test:** Check convergence rate and error
4. **Stability test:** Multiple runs with different random seeds

**Success criteria:**
- ✅ Same or better accuracy (within 1% error tolerance)
- ✅ 10x+ speedup for 20 parameters
- ✅ 100% convergence rate maintained

---

## 💭 MY HONEST OPINION

**Most promising optimizations:**

1. **Numba JIT** (#6) - Easy, huge win, zero accuracy loss
2. **Adaptive Sampling** (#1) - Biggest potential, needs tuning
3. **Parallel Processing** (#5) - Easy, guaranteed 3-4x

**Realistic goal:** Get 20-param time from 70s → 5-10s with Phase 1-2  
**Stretch goal:** Get to 0.5-1s with Phase 3 (competitive with AdamW!)

**Trade-off sweet spot:** Use exact search for N≤10, adaptive for N>10

Would you like me to prototype any of these optimizations?
