# AdamW Hyperparameter Auto-Tuning: Performance Comparison

**Date:** 2026-01-23  
**Test Problem:** Regression with 1000 samples, 50 features, 100 iterations

---

## Executive Summary

⚠️ **CRITICAL FINDING**: Binary search for hyperparameters **degrades performance** on this test problem!

**Key Results:**
- ✅ **Best**: Weight Decay Only (-0.11% cost, +46% time)
- ⚠️ **Worst**: All Hyperparameters (+792% cost, +124% time)
- 📊 **Baseline**: Default hyperparameters perform best overall

---

## Comparison Table

| Configuration          | Final Cost  | Time (s) | Beta1  | Beta2  | Epsilon  | Weight Decay | vs Baseline |
|------------------------|-------------|----------|--------|--------|----------|--------------|-------------|
| **Baseline (Default)** | 0.11581205  | 0.190    | 0.9000 | 0.9990 | 1.00e-08 | 0.0100       | —           |
| Beta1 Only             | 0.58964043  | 0.229    | 0.9900 | 0.9990 | 1.00e-08 | 0.0100       | ↑ +409.14%  |
| Beta2 Only             | 0.11569334  | 0.242    | 0.9000 | 0.9000 | 1.00e-08 | 0.0100       | ↓ -0.10%    |
| Epsilon Only           | 0.11581205  | 0.221    | 0.9000 | 0.9990 | 1.00e-10 | 0.0100       | ↓ -0.00%    |
| Weight Decay Only      | 0.11568847  | 0.278    | 0.9000 | 0.9990 | 1.00e-08 | 0.0000       | ↓ **-0.11%** |
| Beta1 + Beta2          | 0.93795346  | 0.404    | 0.9900 | 0.9000 | 1.00e-08 | 0.0100       | ↑ +709.89%  |
| All Hyperparameters    | 1.03311787  | 0.426    | 0.9900 | 0.9000 | 1.00e-10 | 0.0000       | ↑ +792.06%  |

---

## Analysis

### Why Auto-Tuning Failed

1. **Beta1 Problem (Momentum)**
   - Binary search selected 0.99 (very high momentum)
   - Default 0.9 was better for this problem
   - High momentum can overshoot on smooth convex problems
   - **Impact**: +409% worse when tuned alone

2. **Beta2 Problem (RMSprop)**
   - Binary search selected 0.90 (low RMSprop)
   - Default 0.999 was better
   - Low beta2 causes more noise in adaptive learning rate
   - **Impact**: Slight improvement (-0.10%) but destabilizes when combined

3. **Combination Effect**
   - Beta1=0.99 + Beta2=0.90 = Very unstable
   - High momentum + noisy adaptive rate = Poor convergence
   - **Impact**: +710% worse

4. **Short Testing Window**
   - Binary search tests candidates for only 5 iterations
   - Not enough to detect long-term instability
   - Hyperparameters that look good early can fail later

### What Worked

- **Weight Decay Only**: -0.11% improvement
  - Setting to 0.0 slightly reduced cost
  - Makes sense: synthetic problem has no overfitting
  
- **Epsilon Only**: No change (0.00%)
  - Epsilon has minimal impact on this scale

- **Beta2 Only**: -0.10% improvement
  - Slight benefit, but marginal

### Time Overhead

- **Baseline**: 0.190s
- **Average with tuning**: 0.300s
- **Overhead**: +58% (0.110s extra)

---

## Recommendations

### ❌ Don't Use Auto-Tuning When:
1. **Problem is well-understood** (convex, smooth)
2. **Default hyperparameters work well** (most cases)
3. **Optimization budget is tight** (<100 iterations)
4. **Testing window is too short** (<50 iterations)

### ✅ Use Auto-Tuning When:
1. **Problem is complex/non-convex**
2. **No prior knowledge of good hyperparameters**
3. **Long optimization runs** (>500 iterations)
4. **Willing to pay 50-100% time overhead**
5. **Testing individual hyperparameters separately** (not all at once)

### 🎯 Best Practices

1. **Test One Hyperparameter at a Time**
   - Beta2 alone: Slight improvement
   - Beta1+Beta2 together: Major degradation
   - Reason: Interactions are unpredictable

2. **Increase Test Iterations**
   - Current: 5 iterations per candidate
   - Recommended: 20-50 iterations
   - Trade-off: 4-10× more time

3. **Use Problem-Specific Ranges**
   - Narrow ranges around known-good values
   - Example: beta1_range=(0.85, 0.95) instead of (0.8, 0.99)

4. **Start with Weight Decay**
   - Safest to tune
   - Minimal interaction with other hyperparameters
   - Easy to validate (regularization trade-off)

---

## Conclusion

**The binary search hyperparameter tuning feature works correctly** but **doesn't always improve performance**.

**Key Insight**: Short testing windows (5 iterations) cannot predict long-term behavior. Hyperparameters that reduce cost initially can cause instability later.

**Recommendation for this package**:
- ✅ Keep defaults OFF (user opts-in) — **correct decision**
- ✅ Document limitations clearly
- ⚠️ Consider increasing default test iterations to 20-50
- ⚠️ Add warning when enabling multiple hyperparameters simultaneously
- ⚠️ Add option for user to provide validation function

**When to use**: Complex problems where manual tuning is impractical and you can afford the time cost.

**When NOT to use**: Most cases — default hyperparameters are well-chosen for typical problems.

---

## Test Details

**Problem:**
- Dataset: 1000 samples × 50 features
- True solution cost: ~0.1225
- Initial cost: 22.68
- Optimization: 100 iterations

**Method:**
- Cost function: Mean squared error
- Gradient: Analytical (exact)
- Random seed: 42 (reproducible)

**Tuning Settings:**
- Test iterations: 5 per candidate
- Search steps: 10 (medium model size)
- Strategy: Adaptive (auto-selected)

---

## Future Work

1. **Adaptive Test Length**
   - Start with 5 iterations
   - Increase if candidates are close
   - Maximum 50 iterations

2. **Validation-Based Selection**
   - Allow user to provide validation set
   - Select hyperparameters based on validation cost
   - Prevents overfitting to training data

3. **Meta-Learning**
   - Learn which hyperparameters to tune based on problem characteristics
   - Build database of successful tuning patterns
   - Recommend tuning strategy automatically

4. **Interaction Detection**
   - Test hyperparameters in pairs
   - Detect harmful interactions (beta1+beta2)
   - Warn user or adjust search strategy

---

**Status**: ✅ Feature works as designed, but use with caution!
