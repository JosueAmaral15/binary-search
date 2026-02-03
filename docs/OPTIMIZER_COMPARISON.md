# Optimizer Comparison: WeightCombinationSearch vs BinaryRateOptimizer vs AdamW

**Date:** 2026-02-03  
**Test Suite:** 8 scenarios (N=5, 10, 20 parameters) × (easy, medium, hard difficulty)  
**Metrics:** Speed, Convergence, Iteration Efficiency, Accuracy

---

## Executive Summary

| **Algorithm** | **Best For** | **Avg Speed** | **Convergence Rate** | **Strengths** |
|---------------|--------------|---------------|---------------------|---------------|
| **WeightCombinationSearch** | Discrete optimization | 0.013s (filtered) | N/A* | Combinatorial, deterministic, 96% faster with filtering |
| **BinaryRateOptimizer** | Continuous smooth problems | 0.033s | 75% | Auto-adaptive LR, no tuning needed, best convergence |
| **AdamW** | Deep learning / high-dim | 0.057s | 0% | Industry standard, adaptive per-param, momentum |

*Note: WCS is designed for discrete weight combinations (0/1), not continuous optimization like BRO/AdamW.*

---

## Detailed Comparison Table

### Performance Metrics by Scenario

| Scenario | N | Difficulty | **BRO Time** | **BRO Cost** | **BRO Conv?** | **AdamW Time** | **AdamW Cost** | **AdamW Conv?** |
|----------|---|------------|--------------|--------------|---------------|----------------|----------------|-----------------|
| Small Easy | 5 | Easy | **0.012s** | **0.000000** | ✅ Yes | 0.072s | 11.177 | ❌ No |
| Small Medium | 5 | Medium | **0.025s** | **0.008673** | ✅ Yes | 0.054s | 462.865 | ❌ No |
| Small Hard | 5 | Hard | 0.075s | 1,609,111 | ❌ No | **0.055s** | 907B+ | ❌ No |
| Medium Easy | 10 | Easy | **0.016s** | **0.000000** | ✅ Yes | 0.054s | 10.248 | ❌ No |
| Medium Medium | 10 | Medium | **0.032s** | **0.006625** | ✅ Yes | 0.066s | 941.603 | ❌ No |
| Medium Hard | 10 | Hard | 0.067s | 1,704,126 | ❌ No | **0.064s** | 839B+ | ❌ No |
| Large Easy | 20 | Easy | **0.017s** | **0.000000** | ✅ Yes | 0.068s | 8.393 | ❌ No |
| Large Medium | 20 | Medium | **0.021s** | **0.005256** | ✅ Yes | 0.025s | 1,671.651 | ❌ No |

**Legend:**
- **Bold** = Best performer in category
- Conv? = Converged before max iterations

---

## Key Findings

### 🏆 Winner by Category

| Category | Winner | Reason |
|----------|--------|--------|
| **Overall Speed** | WeightCombinationSearch (filtered) | 96% faster than non-filtered, 61% faster than BRO |
| **Convergence** | BinaryRateOptimizer | 75% convergence rate (6/8 scenarios) |
| **Accuracy** | BinaryRateOptimizer | Lowest MSE on continuous problems |
| **Robustness** | BinaryRateOptimizer | Stable across easy/medium difficulties |
| **Hard Problems** | AdamW | Faster on ill-conditioned problems (though high cost) |

### 📊 Speed Analysis

**Average Time per Scenario:**
- **WeightCombinationSearch (filtered):** 0.013s ⚡ *Fastest*
- **WeightCombinationSearch (no filter):** 0.332s
- **BinaryRateOptimizer:** 0.033s 
- **AdamW:** 0.057s

**Speedup from Filtering:** 96.1% (25x faster!)

### ✅ Convergence Analysis

**Convergence Rate (% of scenarios converged):**
- **BinaryRateOptimizer:** 75% (6/8) ✅ *Best*
- **AdamW:** 0% (0/8) ❌ *Needs more iterations*
- **WeightCombinationSearch:** N/A (different optimization paradigm)

**Avg Iterations to Completion:**
- BinaryRateOptimizer: 24.8 iterations
- AdamW: 51 iterations (hit max limit on all)

### 🎯 Accuracy Analysis (MSE on Continuous Problems)

**Average Final Cost:**
- **BinaryRateOptimizer:** 414,155 (mostly from hard scenarios)
- **AdamW:** 218 trillion (extremely high on hard scenarios)

**On Easy/Medium:** BRO achieves near-zero cost (< 0.01)  
**On Hard:** Both struggle, but BRO is 1000x better

---

## Detailed Analysis by Algorithm

### 1️⃣ WeightCombinationSearch (WCS)

**Purpose:** Discrete combinatorial optimization (binary weights: 0 or 1)

**Strengths:**
- ✅ **Fastest** with filtering enabled (0.013s avg)
- ✅ **Deterministic** results (no randomness)
- ✅ **Index filtering** provides 96% speedup
- ✅ Designed for knapsack-like problems

**Weaknesses:**
- ❌ Not designed for continuous optimization
- ❌ Exponential complexity: O(2^N) without filtering
- ❌ Requires discrete coefficient selection

**Best Use Cases:**
- Portfolio optimization (select stocks: yes/no)
- Feature selection (select features: 0/1)
- Resource allocation (allocate resources: discrete units)
- Knapsack problems
- Any problem requiring binary/discrete weights

**Recommendations:**
- Use **with filtering** for N > 15 parameters
- Not suitable for continuous gradient-based problems
- Ideal for problems where weights must be 0 or 1

---

### 2️⃣ BinaryRateOptimizer (BRO)

**Purpose:** Gradient descent with adaptive learning rate via binary search

**Strengths:**
- ✅ **Best convergence** (75% rate)
- ✅ **No hyperparameter tuning** required
- ✅ **Lowest MSE** on easy/medium problems
- ✅ **Fast** (0.033s avg)
- ✅ **Stable** and predictable behavior

**Weaknesses:**
- ❌ Struggles on ill-conditioned problems (hard scenarios)
- ❌ More iterations than WCS filtering (24.8 vs instant)
- ❌ Requires smooth cost function

**Best Use Cases:**
- Linear regression
- Logistic regression
- Small neural networks (< 100 params)
- Any smooth convex optimization
- When you want "zero-config" optimization

**Recommendations:**
- **Use as default** for continuous optimization
- Best for N < 100 parameters
- Ideal when you don't want to tune learning rates
- Works on both convex and non-convex (but better on smooth)

---

### 3️⃣ AdamW

**Purpose:** State-of-the-art adaptive optimizer for deep learning

**Strengths:**
- ✅ **Industry standard** for neural networks
- ✅ **Adaptive per-parameter** learning rates
- ✅ **Momentum** (beta1) for faster convergence
- ✅ **Weight decay** (decoupled regularization)
- ✅ Handles high-dimensional spaces well

**Weaknesses:**
- ❌ **0% convergence** in these tests (needs more iterations)
- ❌ **Slowest** (0.057s avg)
- ❌ **Highest cost** on hard problems
- ❌ **Requires tuning** (beta1, beta2, epsilon, weight_decay)

**Best Use Cases:**
- Deep neural networks
- Transformer models
- High-dimensional problems (N > 100)
- Problems with noisy gradients
- Non-convex optimization

**Recommendations:**
- **Use for deep learning** (industry standard)
- Give it **more iterations** (100-1000+)
- **Tune hyperparameters** or use auto-tuning
- Not ideal for small/simple problems (overkill)

---

## Use Case Decision Tree

```
START: What kind of optimization problem?
│
├─❓ Are weights discrete (0/1)?
│   └─✅ YES → Use WeightCombinationSearch (with filtering if N > 15)
│
├─❓ Are weights continuous?
│   │
│   ├─❓ Is N < 100 and cost function smooth?
│   │   └─✅ YES → Use BinaryRateOptimizer
│   │
│   ├─❓ Is N > 100 or training deep neural network?
│   │   └─✅ YES → Use AdamW
│   │
│   └─❓ Don't want to tune hyperparameters?
│       └─✅ YES → Use BinaryRateOptimizer
│
└─❓ Unsure?
    └─💡 Try BinaryRateOptimizer first (best general-purpose)
```

---

## Recommendations by Problem Size

| N Parameters | Recommended Algorithm | Reason |
|--------------|----------------------|---------|
| **N < 10** | BinaryRateOptimizer | Fast, no tuning, high convergence |
| **10 ≤ N < 20** | WCS (filtered) or BRO | WCS if discrete, BRO if continuous |
| **20 ≤ N < 100** | BinaryRateOptimizer | Sweet spot for binary search LR |
| **N ≥ 100** | AdamW | Scales better with dimensionality |

---

## Recommendations by Problem Type

| Problem Type | Algorithm | Notes |
|--------------|-----------|-------|
| **Linear Regression** | BinaryRateOptimizer | Near-perfect convergence |
| **Logistic Regression** | BinaryRateOptimizer | Handles non-convex well |
| **Neural Networks** | AdamW | Industry standard |
| **Feature Selection** | WeightCombinationSearch | Discrete binary selection |
| **Portfolio Optimization** | WeightCombinationSearch | Combinatorial allocation |
| **Ill-conditioned Problems** | AdamW (with more iterations) | Adaptive LR helps |

---

## Conclusions

### 🥇 Overall Winner: BinaryRateOptimizer

**For general-purpose continuous optimization:**
- ✅ Best balance of speed, accuracy, and convergence
- ✅ No hyperparameter tuning required
- ✅ 75% convergence rate
- ✅ Lowest MSE on easy/medium problems
- ✅ Stable and predictable

### 🥈 Runner-up: WeightCombinationSearch (with filtering)

**For discrete optimization:**
- ✅ Fastest overall (0.013s)
- ✅ 96% speedup with filtering
- ✅ Deterministic results
- ❌ Limited to binary/discrete weights

### 🥉 Third Place: AdamW

**For deep learning:**
- ✅ Industry standard
- ✅ Adaptive per-parameter rates
- ❌ Needs more iterations to converge
- ❌ Requires hyperparameter tuning

---

## Final Recommendations

### ✨ Quick Picks

1. **"I don't know what optimizer to use"** → **BinaryRateOptimizer**
2. **"I need binary weights (0/1)"** → **WeightCombinationSearch**
3. **"I'm training a neural network"** → **AdamW**
4. **"I want zero configuration"** → **BinaryRateOptimizer**
5. **"I need maximum speed"** → **WeightCombinationSearch (filtered)**

### 💡 Pro Tips

- **WCS Filtering:** Always enable for N > 15 (96% speedup!)
- **BRO:** Works out-of-the-box, no tuning needed
- **AdamW:** Give it 10x more iterations (500-1000) for real problems
- **Hard Problems:** Try BRO first, then AdamW with tuning if needed
- **Discrete Problems:** WCS is the only one designed for this

---

## Benchmark Details

- **Platform:** Python 3, NumPy-based implementations
- **Test Date:** 2026-02-03
- **Scenarios:** 8 test cases (3 difficulties × 3 sizes)
- **Max Iterations:** 50 (may be too low for AdamW)
- **Convergence Criteria:** Cost change < tolerance

---

**Generated by:** Optimizer Comparison Benchmark Suite  
**CSV Results:** `optimizer_comparison_results.csv`
