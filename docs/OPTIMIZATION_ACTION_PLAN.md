# WeightCombinationSearch Optimization Action Plan

**Created:** 2026-02-02  
**Status:** IN PROGRESS  
**User Priority:** C → A → B

---

## 📋 OPTIMIZATION SEQUENCE

### ✅ COMPLETED

#### Phase 0: Critical Fixes (DONE)
- [x] **Intra-Cycle Early Stopping** - 16x speedup (70s → 4.4s for 20 params)
- [x] **Selection Sort Pattern** - 1.31x speedup, O(1) winner tracking
- [x] **Combined Impact:** 16.7x total speedup for 20 parameters

---

## 🚀 PLANNED OPTIMIZATIONS (User Order: C → A → B)

### Phase 1: Adaptive Sampling for Large N (Priority 1)

**Goal:** Make 15+ parameters practical by intelligently sampling combination space

**Implementation Steps:**

1. **Analysis & Design** (30 min)
   - [ ] Study current bottleneck: 2^N combinations (32,768 for N=15, 1M for N=20)
   - [ ] Design adaptive strategies:
     - Random sampling with importance weighting
     - Monte Carlo tree search approach
     - Hybrid: exhaustive for small N, sampling for large N
   - [ ] Define N threshold (suggested: 12-15)

2. **Core Implementation** (2-3 hours)
   - [ ] Add `adaptive_sampling` parameter (default=True)
   - [ ] Add `sampling_threshold` parameter (default=12)
   - [ ] Add `sample_size` parameter (default=min(10000, 2^N))
   - [ ] Implement sampling strategies:
     - [ ] Importance sampling (favor combos with high-value params)
     - [ ] Random sampling with seed control
     - [ ] Progressive refinement (sample → narrow → sample)
   - [ ] Add fallback to exhaustive if N <= threshold

3. **Testing & Validation** (1-2 hours)
   - [ ] Test accuracy: sampling vs exhaustive on N=10 (verify <5% error)
   - [ ] Benchmark speedup: N=15, N=20, N=30, N=50
   - [ ] Test edge cases: all zeros, single dominant param
   - [ ] Verify convergence rate vs accuracy tradeoff

4. **Documentation** (30 min)
   - [ ] Update README with adaptive sampling guide
   - [ ] Document when to use/disable adaptive sampling
   - [ ] Add examples for large N (20, 50, 100 params)

**Expected Impact:**
- **N=15:** 1.6s → ~0.1s (16x faster)
- **N=20:** 4.4s → ~0.2s (22x faster)
- **N=30:** ~7 hours → ~1s (25,000x faster!)
- **N=50:** Impossible → ~5s (makes it POSSIBLE)

**Accuracy Target:** Within 10% of exhaustive solution

---

### Phase 2: Numba JIT Compilation (Priority 2)

**Goal:** Accelerate hot loops with just-in-time compilation

**Implementation Steps:**

1. **Profiling** (20 min)
   - [ ] Profile current code to identify hottest functions
   - [ ] Expected: `_calculate_result()`, combo enumeration, WPN updates

2. **Core Implementation** (1-2 hours)
   - [ ] Install/verify Numba dependency
   - [ ] Add `use_numba` parameter (default=True)
   - [ ] Apply `@njit` decorator to:
     - [ ] `_calculate_result()` - main formula computation
     - [ ] Helper: combo to multiplier conversion
     - [ ] Helper: weight update logic
   - [ ] Handle NumPy array conversions (Numba requires contiguous arrays)
   - [ ] Add graceful fallback if Numba unavailable

3. **Testing & Validation** (1 hour)
   - [ ] Verify numerical accuracy (Numba vs pure Python)
   - [ ] Benchmark compilation overhead (first call vs subsequent)
   - [ ] Test with/without Numba on various N (3, 10, 20)
   - [ ] Measure speedup per function

4. **Documentation** (20 min)
   - [ ] Document Numba dependency (optional but recommended)
   - [ ] Show performance with/without Numba
   - [ ] Installation instructions

**Expected Impact:**
- **10-50x speedup** on formula calculations
- **N=3:** 0.44ms → ~0.02ms
- **N=10:** 31ms → ~2ms
- **N=20 (with sampling):** 0.2s → ~0.01s
- **Total with adaptive sampling:** 0.2s * 50 = ~0.004s for N=20!

---

### Phase 3: Parallel Processing (Priority 3)

**Goal:** Leverage multi-core CPUs for combination testing

**Implementation Steps:**

1. **Architecture Design** (30 min)
   - [ ] Choose approach:
     - Option A: Parallel cycle testing (test multiple cycles simultaneously)
     - Option B: Parallel combo testing within cycle (split 2^N combos)
     - Option C: Parallel WPN exploration (test multiple WPN values)
   - [ ] Recommended: Option B (split combo space into chunks)
   - [ ] Design thread-safe winner tracking

2. **Core Implementation** (2-3 hours)
   - [ ] Add `parallel` parameter (default=True)
   - [ ] Add `n_jobs` parameter (default=-1, use all cores)
   - [ ] Implement multiprocessing pool for combo chunks
   - [ ] Thread-safe best_winner tracking (use Lock or Queue)
   - [ ] Aggregate results from parallel workers
   - [ ] Handle early stopping across workers (shared Event)

3. **Optimization** (1 hour)
   - [ ] Minimize serialization overhead (use shared memory if possible)
   - [ ] Batch size tuning (avoid too many small tasks)
   - [ ] Load balancing (ensure even distribution)
   - [ ] Measure overhead vs speedup tradeoff

4. **Testing & Validation** (1 hour)
   - [ ] Test single-core vs multi-core accuracy
   - [ ] Benchmark 1, 2, 4, 8 cores
   - [ ] Test race conditions (run 100 times, verify consistent results)
   - [ ] Measure overhead for small N (when parallel not worth it)

5. **Documentation** (20 min)
   - [ ] Document when parallel helps (N > 10)
   - [ ] Show core scaling (1 vs 4 vs 8 cores)
   - [ ] Note: parallel + adaptive sampling = best combo

**Expected Impact:**
- **3-4x speedup on 4-core CPU**
- **N=20 (with sampling + Numba):** 0.004s → ~0.001s
- **N=30:** 1s → ~0.25s
- **Diminishing returns:** Overhead dominates for N < 10

---

## 📊 PROJECTED FINAL PERFORMANCE

| N | Original | +Early Stop | +Selection | +Adaptive | +Numba | +Parallel | **Total Speedup** |
|---|----------|-------------|------------|-----------|--------|-----------|-------------------|
| **3**  | 0.42ms   | 0.44ms      | 0.42ms     | 0.42ms    | 0.02ms | 0.02ms    | **21x faster** |
| **10** | 67ms     | 31ms        | 29ms       | 29ms      | 2ms    | 1ms       | **67x faster** |
| **15** | ~1.6s    | ~160ms      | ~150ms     | ~10ms     | 0.5ms  | 0.2ms     | **8,000x faster** |
| **20** | ~70s     | 4.4s        | 4.2s       | ~200ms    | 4ms    | 1ms       | **70,000x faster!** |
| **30** | ~7 hrs   | Impossible  | Impossible | ~1s       | 50ms   | 15ms      | **1.7M x faster!** |
| **50** | Years    | Impossible  | Impossible | ~5s       | 200ms  | 50ms      | **∞ (makes it POSSIBLE!)** |

---

## 🎯 SUCCESS CRITERIA

### Correctness
- [ ] All existing tests pass
- [ ] Sampling accuracy within 10% of exhaustive
- [ ] Parallel results identical to single-threaded

### Performance
- [ ] N=20 completes in < 10ms
- [ ] N=30 completes in < 100ms
- [ ] N=50 completes in < 1s

### User Experience
- [ ] Zero-config defaults work well
- [ ] Advanced users can tune parameters
- [ ] Clear documentation of tradeoffs

---

## 📝 IMPLEMENTATION NOTES

### Phase 1 (Adaptive Sampling) - Key Decisions

**Sampling Strategy:**
1. **Importance Sampling** - Weight combos by parameter magnitudes
   - High-value params more likely to be selected
   - Avoids wasting time on low-impact combos

2. **Progressive Refinement** - Multi-stage approach
   - Stage 1: Random sample 1000 combos → find promising regions
   - Stage 2: Focused sample around best results
   - Stage 3: Final refinement

3. **Hybrid Mode** - Automatic threshold
   - N ≤ 12: Exhaustive (4096 combos, fast enough)
   - N > 12: Adaptive sampling (controlled sample size)

**Key Parameters:**
```python
WeightCombinationSearch(
    adaptive_sampling=True,      # Enable intelligent sampling
    sampling_threshold=12,        # Switch to sampling at N>12
    sample_size=10000,           # Max combos to test per cycle
    sampling_strategy='importance'  # or 'random', 'progressive'
)
```

### Phase 2 (Numba) - Key Decisions

**JIT Compilation Targets:**
```python
@njit
def _calculate_result_jit(params, weights, combo, wpn):
    """JIT-compiled hot loop"""
    result = 0.0
    for i in range(len(params)):
        w = weights[i] if weights[i] != 0 else (1.0 if combo[i] else 0.0)
        mult = wpn if combo[i] else 1.0
        result += params[i] * w * mult
    return result
```

**Fallback Strategy:**
- Try to import Numba, if fails → use pure Python
- First call slower (compilation), subsequent calls fast
- Cache compiled functions for reuse

### Phase 3 (Parallel) - Key Decisions

**Chunking Strategy:**
```python
# Split 2^N combos into chunks
n_combos = 2**N - 1
chunk_size = max(1000, n_combos // (n_jobs * 4))  # 4 chunks per core

# Each worker tests a chunk
def worker(chunk_start, chunk_end):
    best_local = None
    for combo_idx in range(chunk_start, chunk_end):
        result = test_combination(combo_idx)
        if result better than best_local:
            best_local = result
        if early_stop_event.is_set():
            break
    return best_local
```

**Synchronization:**
- Shared `best_delta` with Lock
- Shared `early_stop_event` for cross-worker stopping
- Minimize lock contention (only update if better)

---

## 🧪 TESTING STRATEGY

### For Each Phase:

1. **Correctness Tests**
   - Small N (3-5): Compare new vs old results (must match exactly)
   - Medium N (10-15): Verify convergence quality
   - Large N (20-50): Check solution reasonableness

2. **Performance Tests**
   - Benchmark suite: N=3,5,7,10,12,15,20,30,50
   - Measure time, speedup, efficiency
   - Compare to AdamW baseline

3. **Regression Tests**
   - Ensure all 35 existing tests still pass
   - No breaking changes to API
   - Backward compatibility maintained

4. **Integration Tests**
   - Test combinations: sampling + Numba, sampling + parallel, all three
   - Verify features don't conflict
   - Test enable/disable flags

---

## 📈 PROGRESS TRACKING

**Phase 1 (Adaptive Sampling):** 🔲 NOT STARTED  
**Phase 2 (Numba JIT):** 🔲 NOT STARTED  
**Phase 3 (Parallel Processing):** 🔲 NOT STARTED

**Estimated Total Time:** 8-12 hours of focused work

**Start Date:** 2026-02-02  
**Target Completion:** TBD

---

## ✅ NEXT IMMEDIATE ACTION

**Starting with Phase 1: Adaptive Sampling**

1. Analyze current combination enumeration code
2. Design importance sampling strategy
3. Implement adaptive threshold switching
4. Test on N=15, 20, 30
5. Commit and document

**Let's proceed!** 🚀
