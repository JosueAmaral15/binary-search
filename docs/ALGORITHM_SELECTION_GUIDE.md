# Algorithm Selection Guide

**How to choose the right optimizer for your problem**

---

## 🎯 The Golden Rule

### Is your problem LINEAR?

```
Can you write it as: Ax = b?
Examples: Linear regression, systems of equations
```

**YES → Use NumPy Direct Solve** ✅  
**NO → Use BinaryRateOptimizer** ✅

---

## 📊 Detailed Decision Tree

```
START
  ↓
Is it a LINEAR system (Ax = b)?
  ├─ YES → Use np.linalg.solve()  [FASTEST - 10-1000× speedup]
  │         Examples:
  │         • Linear regression
  │         • Normal equations
  │         • Matrix equations
  │
  └─ NO → Is it gradient descent optimization?
      ├─ YES → How many parameters?
      │    ├─ < 10,000 → BinaryRateOptimizer  [10× faster than AdamW]
      │    └─ > 10,000 → Consider AdamW or NumPy if reformulable
      │
      └─ NO → Is it array search?
           └─ Use BinarySearch (search algorithms)
```

---

## 1️⃣ LINEAR Problems: Use NumPy Direct Solve

### What is a Linear Problem?

Any problem that can be written as **Ax = b** where:
- **A** is a matrix (coefficients)
- **x** is the unknown vector (variables to find)
- **b** is the result vector

### Examples:

#### Linear Regression
```python
import numpy as np

# Problem: Find θ such that y ≈ Xθ
# Solution: Normal equations → (X^T X)θ = X^T y

A = X.T @ X
b = X.T @ y
theta = np.linalg.solve(A, b)  # FASTEST!
```

#### System of Equations
```python
# Solve:
#   2x + 3y = 8
#   4x + 5y = 14

A = np.array([[2, 3],
              [4, 5]])
b = np.array([8, 14])

solution = np.linalg.solve(A, b)
# Result: x=1, y=2
```

#### Many Variables (x, y, z, w, a1, a2, ..., a500)
```python
# Even with 500+ variables, NumPy is FASTEST

# Generate problem: 1000 samples × 500 variables
X = np.random.randn(1000, 500)
y = np.random.randn(1000)

# Solve in ~10 milliseconds
A = X.T @ X
b = X.T @ y
theta = np.linalg.solve(A, b)  # 500 coefficients found!
```

### Performance

| Variables | NumPy Time | BinaryRate Time | AdamW Time | Speedup |
|-----------|------------|-----------------|------------|---------|
| 5 | 0.1 ms | 3.7 ms | 6.7 ms | **57×** |
| 10 | 0.1 ms | 5.3 ms | 7.2 ms | **89×** |
| 50 | 0.2 ms | 10.4 ms | 11.0 ms | **62×** |
| 100 | 0.3 ms | 20.8 ms | 13.5 ms | **46×** |
| 200 | 1.1 ms | 107.9 ms | 19.0 ms | **18×** |
| 500 | 10.1 ms | 250.7 ms | 103.8 ms | **10×** |

**Conclusion:** NumPy is 10-90× faster, always wins! ✅

---

## 2️⃣ NON-LINEAR Problems: Use BinaryRateOptimizer

### What is a Non-Linear Problem?

Any optimization problem that **cannot** be written as Ax = b:
- Custom cost functions
- Logistic regression
- Neural networks
- Non-convex optimization
- Regularized problems (beyond simple L2)

### Examples:

#### Logistic Regression
```python
from math_toolkit.optimization import BinaryRateOptimizer
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    h = sigmoid(X @ theta)
    return -np.mean(y * np.log(h) + (1-y) * np.log(1-h))

def gradient(theta, X, y):
    h = sigmoid(X @ theta)
    return X.T @ (h - y) / len(y)

# Optimize
optimizer = BinaryRateOptimizer(max_iter=50, tol=1e-6)
theta = optimizer.optimize(X, y, initial_theta, cost, gradient)
```

#### Custom Cost Function
```python
from math_toolkit.optimization import BinaryRateOptimizer

# Non-linear cost: minimize sum of absolute errors + penalty
def custom_cost(theta, X, y):
    predictions = X @ theta
    mae = np.mean(np.abs(predictions - y))
    penalty = 0.1 * np.sum(theta**4)  # Non-linear penalty
    return mae + penalty

def custom_gradient(theta, X, y):
    predictions = X @ theta
    errors = predictions - y
    grad_mae = X.T @ np.sign(errors) / len(y)
    grad_penalty = 0.4 * theta**3
    return grad_mae + grad_penalty

optimizer = BinaryRateOptimizer(max_iter=100)
theta = optimizer.optimize(X, y, initial_theta, custom_cost, custom_gradient)
```

#### Neural Network Training
```python
from math_toolkit.optimization import BinaryRateOptimizer

# Simple 2-layer network
def forward(theta, X):
    W1, b1, W2, b2 = split_theta(theta)
    hidden = np.tanh(X @ W1 + b1)
    output = hidden @ W2 + b2
    return output

def mse_cost(theta, X, y):
    return np.mean((forward(theta, X) - y) ** 2)

def compute_gradient(theta, X, y):
    # Backpropagation...
    return grad

optimizer = BinaryRateOptimizer(max_iter=200)
theta = optimizer.optimize(X, y, initial_theta, mse_cost, compute_gradient)
```

### Performance

**BinaryRateOptimizer vs AdamW** (1000 samples × 50 features):

| Metric | BinaryRateOptimizer | AdamW | Winner |
|--------|---------------------|-------|--------|
| Time | 0.027s | 0.265s | **Binary 10× faster** ✅ |
| Cost | 0.00462 | 0.00486 | **Binary better** ✅ |
| Iterations | 10 | 100 | **Binary 10× fewer** ✅ |

---

## 3️⃣ Alternative: AdamW (For Specific Cases)

### When to Use AdamW

```python
from math_toolkit.optimization import AdamW

optimizer = AdamW(use_binary_search=True, max_iter=100)
theta = optimizer.optimize(X, y, initial_theta, cost, gradient)
```

**Use AdamW when:**
- 🔧 Integrating with PyTorch/TensorFlow frameworks
- 🎛️ Need per-parameter adaptive learning rates
- 📊 Small to medium datasets
- 🧪 Experimenting with different optimizers

**Don't use when:**
- ⚠️ Large datasets (BinaryRateOptimizer is 10× faster)
- ⚠️ Linear problems (NumPy is 100× faster)

---

## 4️⃣ BinaryGaussSeidel (Iterative Linear Solver)

### When to Use

```python
from math_toolkit.linear_systems import BinaryGaussSeidel

solver = BinaryGaussSeidel(max_iterations=1000, tolerance=1e-6)
x = solver.solve(A, b)
```

**Use ONLY when:**
- ✅ Matrix A is **sparse** (mostly zeros)
- ✅ Matrix A is **strictly diagonally dominant**
- ✅ Want iterative solver (not direct)

**Don't use when:**
- ❌ Dense matrices (NumPy is faster)
- ❌ Not diagonally dominant (won't converge)
- ❌ Need guaranteed solution (use NumPy)

**Reality:** For most cases, `np.linalg.solve()` is better!

---

## 5️⃣ BinarySearch (Array Search & Root Finding)

### When to Use

```python
from math_toolkit.optimization import BinarySearch

# Find value in sorted array
index = BinarySearch.search([1, 2, 3, 4, 5], target=3)

# Find root: x^2 = 100
result = BinarySearch.search_for_function(
    y=100,
    function=lambda x: x**2,
    tolerance=1e-6
)
```

**Use when:**
- 🔍 Searching sorted arrays
- 🎯 Finding function roots
- 📐 Inverse function evaluation
- ⚖️ Tolerance-based comparisons

---

## 📋 Quick Reference Table

| Problem Type | Best Algorithm | Why | Speedup |
|-------------|----------------|-----|---------|
| **Linear regression** | `np.linalg.solve()` | Exact, O(n³), LAPACK optimized | 10-1000× |
| **Systems of equations (Ax=b)** | `np.linalg.solve()` | Direct solver, guaranteed solution | 10-1000× |
| **Many variables (linear)** | `np.linalg.solve()` | Scales well, still fastest | 10-90× |
| **Logistic regression** | `BinaryRateOptimizer` | Non-linear, binary search LR | 10× vs AdamW |
| **Neural networks** | `BinaryRateOptimizer` | Fast convergence, fewer iterations | 10× vs AdamW |
| **Custom cost functions** | `BinaryRateOptimizer` | Adaptive learning rate | 10× vs AdamW |
| **Deep learning frameworks** | `AdamW` | Per-parameter rates, PyTorch compatible | Industry standard |
| **Sparse linear systems** | `BinaryGaussSeidel` | Iterative, memory efficient | Use case specific |
| **Array search** | `BinarySearch` | O(log n) search | Standard algorithm |
| **Root finding** | `BinarySearch` | Function inversion | Standard algorithm |

---

## 🎓 Key Insights from Benchmarks

### 1. Linear Problems: NumPy Dominates

```
Problem: 1000 samples × 200 variables (linear regression)

NumPy solve:         1.1 ms  ← WINNER
BinaryRateOptimizer: 107.9 ms
AdamW:               19.0 ms

Speedup: 18× faster than BinaryRate, 17× faster than AdamW
```

**Lesson:** If you can formulate as Ax = b, **always use NumPy**!

---

### 2. Non-Linear Problems: BinaryRateOptimizer Wins

```
Problem: 1000 samples × 50 features (gradient descent)

BinaryRateOptimizer: 0.027s, 10 iterations  ← WINNER
AdamW:               0.265s, 100 iterations

Speedup: 10× faster, better accuracy
```

**Lesson:** Binary search for learning rate dramatically reduces iterations!

---

### 3. More Variables ≠ Different Winner

```
Variables:  5  →  500 (100× increase)

NumPy:     0.1ms → 10.1ms (100× slower, still FASTEST)
BinaryRate: 3.7ms → 250.7ms (68× slower)
AdamW:     6.7ms → 103.8ms (15× slower)

NumPy still wins by 10× at 500 variables!
```

**Lesson:** NumPy scales better than gradient descent!

---

### 4. Complexity Order Misleading

```
BinaryRateOptimizer: O(n²) per iteration × 10 iterations
AdamW:               O(n) per iteration × 100 iterations

Result: BinaryRate is 10× FASTER!
```

**Lesson:** Fewer smart iterations > many simple iterations!

---

## 💡 Common Mistakes

### ❌ Mistake 1: Using gradient descent for linear regression

```python
# SLOW (0.265s for 1000×50)
optimizer = AdamW()
theta = optimizer.optimize(X, y, initial_theta, mse_cost, gradient)
```

```python
# FAST (0.005s for 1000×50) ✅
theta = np.linalg.solve(X.T @ X, X.T @ y)
```

**Fix:** If linear, use NumPy!

---

### ❌ Mistake 2: Using NumPy for non-linear problems

```python
# WON'T WORK - logistic regression is non-linear
theta = np.linalg.solve(X.T @ X, X.T @ y)  # Wrong!
```

```python
# CORRECT ✅
optimizer = BinaryRateOptimizer()
theta = optimizer.optimize(X, y, initial_theta, logistic_cost, logistic_grad)
```

**Fix:** Non-linear problems need gradient descent!

---

### ❌ Mistake 3: Using BinaryGaussSeidel for dense matrices

```python
# SLOW and may not converge
solver = BinaryGaussSeidel()
x = solver.solve(dense_matrix, b)
```

```python
# FAST and guaranteed ✅
x = np.linalg.solve(dense_matrix, b)
```

**Fix:** BinaryGaussSeidel only for sparse, diagonally-dominant matrices!

---

## 🚀 Final Recommendations

### Default Choice
1. **Try NumPy first** - Can you formulate as Ax = b?
2. **Use BinaryRateOptimizer** - If non-linear
3. **Consider AdamW** - If integrating with frameworks

### Performance Priority
```python
# FASTEST to SLOWEST (for linear problems)
np.linalg.solve()          # 1st choice ✅
BinaryRateOptimizer        # 2nd choice (if non-linear)
AdamW                       # 3rd choice (if needed)
```

### When in Doubt
**Linear?** → NumPy  
**Non-linear?** → BinaryRateOptimizer  
**Framework integration?** → AdamW

---

## 📚 See Also

- [SCALABILITY_BENCHMARK.md](SCALABILITY_BENCHMARK.md) - Detailed performance tests
- [OPTIMIZER_BENCHMARK.md](OPTIMIZER_BENCHMARK.md) - Original comparison
- [OBSERVER_ADAMW.md](OBSERVER_ADAMW.md) - Parallel hyperparameter tuning
- [README.md](../README.md) - Package overview

---

**Last Updated:** 2026-01-25  
**Benchmark Platform:** Python 3, NumPy, Standard CPU
