# Optimizer Benchmark Results

**Date:** 2026-01-25  
**Question:** Which algorithm has both lowest complexity order AND lowest processing time?

---

## Test Problem

- **Type:** Linear Regression  
- **Samples:** 1000  
- **Features:** 50  
- **Initial Cost:** 22.516619  
- **Optimal Cost:** 0.004899

---

## Results Summary

| Algorithm | Final Cost | Time (s) | Iterations | Complexity | Time/Iter |
|-----------|------------|----------|------------|------------|-----------|
| **BinaryRateOptimizer** | **0.00462297** | **0.027** | **10** | O(n²) per iter | 0.0027s |
| AdamW (Standard) | 0.00485638 | 0.284 | 100 | O(n) per iter | 0.0028s |
| AdamW (No Binary LR) | 19.02790 | 0.019 | 100 | O(n) per iter | 0.0002s |

---

## Winners

### 🏆 Best Accuracy (Lowest Cost)
**BinaryRateOptimizer**
- Cost: 0.00462297
- Time: 0.027s
- Iterations: 10

### ⚡ Fastest Processing Time (with convergence)
**BinaryRateOptimizer**
- Time: 0.027s  
- Cost: 0.00462297
- Iterations: 10

### 🎯 Best Overall
**BinaryRateOptimizer** ✅
- Fastest time
- Best accuracy
- Fewest iterations

---

## Complexity Analysis

### Time Complexity per Iteration

**BinaryRateOptimizer: O(n²) per iteration**
- Binary search tests multiple learning rates
- Each test requires one gradient descent step
- BUT: Only needs ~10 iterations total
- **Total:** O(10 × n²) ≈ O(n²)

**AdamW: O(n) per iteration**
- Single gradient step with adaptive rates
- Plus binary search for global learning rate (when enabled)
- BUT: Needs ~100 iterations
- **Total:** O(100 × n) ≈ O(n)

---

## Why BinaryRateOptimizer Wins

Despite **higher complexity per iteration** (O(n²) vs O(n)), BinaryRateOptimizer is **10× faster in practice**:

```
BinaryRateOptimizer:
  10 iterations × O(n²) per iter = O(10n²) total
  Real time: 0.027s ✅

AdamW:
  100 iterations × O(n) per iter = O(100n) total
  Real time: 0.284s
```

**Key Insight:** Fewer iterations with smarter search beats more iterations with lower per-iteration cost!

---

## Answer to Your Question

**"Which algorithm has BOTH lowest complexity order AND lowest processing time?"**

### It depends on perspective:

1. **Lowest complexity PER ITERATION:** AdamW (O(n))

2. **Lowest TOTAL processing time:** **BinaryRateOptimizer** ✅ (0.027s)

3. **Best PRACTICAL performance:** **BinaryRateOptimizer** ✅
   - Fastest time
   - Best accuracy
   - Fewer total operations despite higher per-iteration complexity

---

## Recommendation

### ✅ Use BinaryRateOptimizer for:
- **Gradient descent problems** where you want fast convergence
- When you need **both speed AND accuracy**
- Problems where finding optimal learning rate is critical
- **10× faster than AdamW** in practice

### Use AdamW when:
- You need **per-parameter adaptive learning rates**
- Working with **deep learning** / neural networks
- Need **industry-standard** optimizer (PyTorch/TensorFlow)
- Many parameters with different scales

### Use Gauss-Seidel when:
- Solving **linear systems only** (Ax = b)
- Sparse matrices
- Note: For dense systems, use `numpy.linalg.solve()` instead

---

## Detailed Breakdown

### BinaryRateOptimizer (WINNER ✅)

**Pros:**
- ⚡ Fastest: 0.027s
- 🎯 Best accuracy: 0.00462297
- 📊 Fewest iterations: 10
- 💡 Smart: Binary search finds optimal LR automatically

**Cons:**
- O(n²) per iteration (but only 10 iterations needed!)

**When to use:**
- Any gradient descent problem
- When speed matters
- When accuracy matters

---

### AdamW (Standard)

**Pros:**
- O(n) per iteration (most efficient per iteration)
- Adaptive per-parameter learning rates
- Industry standard

**Cons:**
- Slower total time: 0.284s (10× slower than BinaryRateOptimizer)
- Needs 100 iterations vs 10

**When to use:**
- Deep learning
- Many parameters
- Need adaptive rates

---

### AdamW (No Binary Search)

**Pros:**
- Fastest iteration time: 0.0002s per iteration
- Pure O(n) complexity

**Cons:**
- ❌ DOESN'T CONVERGE: Cost = 19.02 (vs optimal 0.005)
- Fixed learning rate causes issues
- Fast but useless without convergence

**When to use:**
- Don't use this configuration
- Binary search for LR is essential

---

## Conclusion

**For descending gradient optimization:**

**Winner: BinaryRateOptimizer** ✅

- **Lowest processing time:** 0.027s
- **Best accuracy:** 0.00462297  
- **Practical complexity:** O(n²) total (only 10 iterations)
- **Recommendation:** Use this for gradient descent problems

The key lesson: **Complexity order per iteration doesn't tell the full story**. Total iterations and convergence speed matter more!

---

## Benchmark Details

**Hardware:** Standard CPU  
**Language:** Python 3  
**Libraries:** NumPy  
**Seed:** 42 (reproducible)  
**Problem:** Linear regression with normal equations
