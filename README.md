# Math Toolkit - Comprehensive Mathematical Optimization Library

**Version 2.0.0** - Reorganized package structure with improved cohesion

A comprehensive Python package for mathematical optimization, binary search algorithms, and linear system solvers, featuring innovative binary search paradigms for hyperparameter tuning.

> **⚠️ BREAKING CHANGES:** Package renamed from `binary_search` to `math_toolkit`. See [MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) for upgrade instructions. Old imports still work with deprecation warnings.

---

## 🎯 What's New in v2.0.0

### 🏗️ **Reorganized Package Structure**
Package reorganized for better cohesion and discoverability:

```
math_toolkit/
├── binary_search/           # Search algorithms
│   ├── algorithms.py        # BinarySearch
│   └── combinatorial.py     # WeightCombinationSearch (NEW!)
├── optimization/            # Gradient-based optimizers
│   ├── gradient_descent.py
│   ├── adaptive_optimizer.py
│   └── observer_tuning.py
└── linear_systems/          # Linear & nonlinear system solvers
    ├── iterative.py
    └── nonlinear.py
```

### 🆕 **New: WeightCombinationSearch**
Find optimal weights for linear combinations using combinatorial search + binary refinement:
```python
from math_toolkit.binary_search import WeightCombinationSearch

# Find W where: 15*W1 + 47*W2 + (-12)*W3 ≈ 28
search = WeightCombinationSearch(tolerance=2, max_iter=50)
weights = search.find_optimal_weights([15, 47, -12], target=28)
# Result: [0.5, 0.5, 0.125] → 29.5 (within tolerance!)
```

**Key Features:**
- Tests all 2^N-1 combinations per cycle (truth table approach)
- Binary refinement via Weighted Possibility Number (WPN)
- No learning rate tuning needed
- Converges faster than gradient descent for weight problems
- Optional truth table output (DataFrame/CSV)

### 📦 **Clear Module Organization**
- **math_toolkit.binary_search** - Binary search algorithms with tolerance-based comparisons
- **math_toolkit.optimization** - ML optimizers (BinaryRateOptimizer, AdamW, ObserverAdamW)
- **math_toolkit.linear_systems** - Iterative solvers (BinaryGaussSeidel, NonLinearGaussSeidel)

### ✅ **Backward Compatible**
Old imports still work (with deprecation warnings):
```python
# Still works in v2.x
from binary_search import BinarySearch, BinaryRateOptimizer, AdamW
```

See [MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) for details.

---

## 🎯 Algorithm Selection Guide

### 📊 When to Use Each Algorithm

#### ✅ **LINEAR Problems** (y = Ax + b, systems of equations)

**Winner: NumPy Direct Solve** - 10-1000× faster than gradient descent

```python
import numpy as np

# For linear regression or Ax = b
A = X.T @ X
b = X.T @ y
theta = np.linalg.solve(A, b)  # FASTEST - use this!
```

**Use when:**
- Linear regression (any number of variables)
- Solving systems of equations: Ax = b
- Problem can be formulated as matrix equation
- Need exact solution
- 5 to 10,000+ variables

---

**Winner: BinaryRateOptimizer** - 10× faster than AdamW

```python
from math_toolkit.optimization import BinaryRateOptimizer

# For any gradient-based optimization
optimizer = BinaryRateOptimizer(max_iter=50, tol=1e-6)
theta = optimizer.optimize(X, y, initial_theta, cost_fn, gradient_fn)
```

**Use when:**
- Non-linear optimization
- Custom/complex cost functions
- Logistic regression, neural networks
- Need both speed AND accuracy
- Large datasets (scales well)

**Alternative: AdamW**
```python
from math_toolkit.optimization import AdamW

# When you need per-parameter adaptive learning rates
optimizer = AdamW(use_binary_search=True, max_iter=100)
theta = optimizer.optimize(X, y, initial_theta, cost_fn, gradient_fn)
```

**Use AdamW when:**
- Deep learning / PyTorch integration
- Need per-parameter learning rates
- Small to medium datasets

---

### 📈 Performance Comparison

| Algorithm | Speed | Accuracy | Best For |
|-----------|-------|----------|----------|
| **NumPy solve** | ⚡⚡⚡ | 🎯🎯🎯 | Linear systems (Ax=b) |
| **BinaryRateOptimizer** | ⚡⚡ | 🎯🎯 | Non-linear gradient descent |
| **AdamW** | ⚡ | 🎯 | Deep learning frameworks |

**Benchmarks:** See [docs/SCALABILITY_BENCHMARK.md](docs/SCALABILITY_BENCHMARK.md) for detailed performance analysis across dataset sizes (5-500 variables).

---

## 📦 Installation

```bash
# Install from source
git clone https://github.com/JosueAmaral15/binary-search.git
cd binary-search
pip install -e .
```

---

## 🚀 Quick Start

### Binary Search Algorithms

```python
from math_toolkit.binary_search import BinarySearch

# Find root: x^2 = 100
result = BinarySearch.search_for_function(
    y=100,
    function=lambda x: x**2,
    tolerance=1e-6
)
print(f"sqrt(100) = {result}")  # 10.0
```

### Weight Combination Search

```python
from math_toolkit.binary_search import WeightCombinationSearch

# Find optimal weights: 15*W1 + 47*W2 + (-12)*W3 ≈ 28
coefficients = [15, 47, -12]
target = 28
tolerance = 2

search = WeightCombinationSearch(tolerance=tolerance, max_iter=50)
weights = search.find_optimal_weights(coefficients, target)

result = sum(c * w for c, w in zip(coefficients, weights))
print(f"Weights: {weights}")  # [0.5, 0.5, 0.125]
print(f"Result: {result:.2f} (target: {target})")  # 29.5 (within tolerance)

# Works with ANY number of parameters!
coefficients_6 = [15, 47, -12, 123, 56, 10]
weights_6 = search.find_optimal_weights(coefficients_6, target=28)
# Returns optimal weights for all 6 parameters
```

**Scalability:** Works with 2-20+ parameters. Fast for 2-7 params, good for 8-10, slower for 11+.

### Gradient Descent Optimization

```python
import numpy as np
from math_toolkit.optimization import BinaryRateOptimizer

# Define your problem
def cost(theta, X, y):
    return np.mean((X @ theta - y) ** 2)

def gradient(theta, X, y):
    return 2 * X.T @ (X @ theta - y) / len(y)

# Optimize - NO LEARNING RATE TUNING NEEDED!
optimizer = BinaryRateOptimizer(max_iter=50, tol=1e-6)
X, y = ...  # Your data
theta = optimizer.optimize(X, y, initial_theta, cost, gradient)
```

### AdamW Optimizer

```python
from math_toolkit.optimization import AdamW

optimizer = AdamW(use_binary_search=True, max_iter=100)
theta = optimizer.optimize(X, y, initial_theta, cost, gradient)
```

### Linear System Solver

```python
from math_toolkit.linear_systems import BinaryGaussSeidel

# Solve Ax = b iteratively
solver = BinaryGaussSeidel(max_iterations=1000, tolerance=1e-6)
result = solver.solve(A, b)
```

### Nonlinear System Solver

```python
from math_toolkit.linear_systems import NonLinearGaussSeidel

# Solve F(x) = 0 (no derivatives needed!)
# Example: x² + y - 11 = 0, x + y² - 7 = 0
f1 = lambda x, y: x**2 + y - 11
f2 = lambda x, y: x + y**2 - 7

solver = NonLinearGaussSeidel(functions=[f1, f2])
result = solver.solve(initial_guess=[0, 0])
print(f"Solution: x={result.x[0]:.4f}, y={result.x[1]:.4f}")  # x=3, y=2
```

---

## 📚 What's Included

### 1. **Binary Search Algorithms** (`math_toolkit.binary_search`)
- **BinarySearch**: Advanced search with tolerance-based comparisons
  - Array search, function root finding, stepped search
  - Mathematical comparison functions
- **WeightCombinationSearch**: Combinatorial weight optimization
  - Finds optimal weights for linear combinations (A·W ≈ Target)
  - Uses truth table + binary refinement via Weighted Possibility Number (WPN)
  - Converges faster than gradient descent for weight finding
  - No learning rate tuning needed
  - **Scalable:** Works with ANY number of parameters (2 to 20+)
  - **Performance:** Fast for 2-7 params, good for 8-10, slower for 11+

### 2. **Optimization Algorithms** (`math_toolkit.optimization`)
- **BinaryRateOptimizer**: Gradient descent with binary search learning rate (10× faster than AdamW)
- **AdamW**: Adaptive moment estimation with weight decay and binary search
- **ObserverAdamW**: Parallel hyperparameter tuning with observer pattern

### 3. **Linear System Solvers** (`math_toolkit.linear_systems`)
- **BinaryGaussSeidel**: Iterative solver for linear systems (Ax = b)
  - Polynomial regression support
  - Large system optimization (10×10 to 100×100)
- **NonLinearGaussSeidel**: Solver for nonlinear systems F(x) = 0
  - **No derivatives needed** - uses binary search for root finding
  - Robust error handling for domain errors
  - Works with transcendental, exponential, trigonometric functions
  - Solves 1D, 2D, 3D, ..., N-D systems

---

## 📊 WeightCombinationSearch Scalability Guide

**Works with ANY number of parameters**, but performance varies:

| Parameters | Combinations/Cycle | Performance | Use Case |
|------------|-------------------|-------------|----------|
| 2-3 | 3-7 | ⚡ **Instant** (< 0.1s) | Learning, simple cases |
| 4-5 | 15-31 | ✅ **Fast** (< 1s) | Most common use cases |
| 6-7 | 63-127 | ✅ **Good** (1-2s) | Complex problems |
| 8-10 | 255-1,023 | ⚠️ **Acceptable** (few seconds) | Large ensembles |
| 11-15 | 2,047-32,767 | 🐌 **Slow** (minutes) | Not recommended |
| 16+ | 65,535+ | ❌ **Very slow** (hours) | Use gradient methods |

**Complexity:** O(max_iter × 2^N) where N = number of parameters

**Recommendation:** 
- **2-7 parameters:** Use WeightCombinationSearch (fast & accurate)
- **8-10 parameters:** Still usable, increase `max_iter` if needed
- **11+ parameters:** Consider `BinaryRateOptimizer` or `AdamW` instead

**Example:**
```python
# Fast: 3 parameters
search.find_optimal_weights([15, 47, -12], target=28)  # < 0.1s

# Still fast: 6 parameters  
search.find_optimal_weights([15, 47, -12, 123, 56, 10], target=28)  # ~1s

# Acceptable: 10 parameters
search.find_optimal_weights([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], target=100)  # few seconds
```

---

**Winner: BinaryRateOptimizer** - 10× faster than AdamW

```python
from binary_search import BinaryRateOptimizer

# For any gradient-based optimization
optimizer = BinaryRateOptimizer(max_iter=50, tol=1e-6)
theta = optimizer.optimize(X, y, initial_theta, cost_fn, gradient_fn)
```

**Use when:**
- Non-linear optimization
- Custom/complex cost functions
- Logistic regression, neural networks
- Need both speed AND accuracy
- Large datasets (scales well)

**Alternative: AdamW**
```python
from binary_search import AdamW

# When you need per-parameter adaptive learning rates
optimizer = AdamW(use_binary_search=True, max_iter=100)
theta = optimizer.optimize(X, y, initial_theta, cost_fn, gradient_fn)
```

**Use AdamW when:**
- Deep learning / PyTorch integration
- Need per-parameter learning rates
- Small to medium datasets

---

### 📈 Performance Comparison

| Algorithm | Speed | Accuracy | Best For |
|-----------|-------|----------|----------|
| **NumPy solve** | ⚡⚡⚡ | 🎯🎯🎯 | Linear systems (Ax=b) |
| **BinaryRateOptimizer** | ⚡⚡ | 🎯🎯 | Non-linear gradient descent |
| **AdamW** | ⚡ | 🎯 | Deep learning frameworks |

**Benchmarks:** See [docs/SCALABILITY_BENCHMARK.md](docs/SCALABILITY_BENCHMARK.md) for detailed performance analysis across dataset sizes (5-500 variables).

---

## 📦 What's Included

### 1. **Optimizers** (ML/Gradient-Based)
- **BinaryRateOptimizer**: Gradient descent with dynamic learning rate via binary search
- **AdamW** ⭐ NEW: Adaptive optimizer with binary search learning rate

### 2. **Algorithms** (Search & Root Finding)
- **BinarySearch**: Array search, function root finding, tolerance-based comparisons

### 3. **Linear System Solvers**
- **BinaryGaussSeidel**: Iterative solver for sparse, diagonally-dominant matrices
- **Note:** For most linear systems, use `np.linalg.solve()` instead (much faster)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd binary_search

# Optional: Install in development mode
pip install -e .
```

### Basic Usage

#### AdamW Optimizer (Recommended) ⭐

```python
import numpy as np
from binary_search.optimizers import AdamW

# Define your problem
def cost(theta, X, y):
    return np.mean((X @ theta - y) ** 2)

def gradient(theta, X, y):
    return 2 * X.T @ (X @ theta - y) / len(y)

# Optimize - NO LEARNING RATE TUNING NEEDED!
optimizer = AdamW(use_binary_search=True, max_iter=100)
X, y = ...  # Your data
theta = optimizer.optimize(X, y, initial_theta, cost, gradient)
```

#### BinaryRateOptimizer

```python
from binary_search.optimizers import BinaryRateOptimizer

optimizer = BinaryRateOptimizer(max_iter=50, tol=1e-6)
theta = optimizer.optimize(X, y, initial_theta, cost, gradient)
```

#### Binary Search Algorithms

```python
from binary_search.algorithms import BinarySearch

# Find root: x^2 = 100
result = BinarySearch.search_for_function(
    y=100,
    function=lambda x: x**2,
    tolerance=1e-6
)
print(f"sqrt(100) = {result}")  # 10.0
```

---

## 📊 Performance Validation

**Test Problem:** Linear regression `y = 2x`

| Optimizer | Error | Status |
|-----------|-------|--------|
| BinaryRateOptimizer | 0.000128 | ✓ Excellent |
| AdamW (fixed LR) | 0.237977 | ✓ Needs tuning |
| **AdamW (binary search)** | **0.000008** | **✓ Perfect** ⭐ |

**Winner:** AdamW with binary search - highest accuracy without manual tuning!

---

## 📚 Documentation

### Main Documentation
- **[docs/REVIEW_RESULTS.md](docs/REVIEW_RESULTS.md)** - Comprehensive test results (24/24 passed)
- **[docs/STRUCTURE_v1.1.md](docs/STRUCTURE_v1.1.md)** - Architecture guide and detailed usage
- **[docs/RESTRUCTURING_SUMMARY.md](docs/RESTRUCTURING_SUMMARY.md)** - What changed in v1.1
- **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Current project status

### Quick Navigation
- **[docs/INDEX.md](docs/INDEX.md)** - Documentation index
- **[docs/FILE_ORGANIZATION.md](docs/FILE_ORGANIZATION.md)** - File structure guide

### Examples
Run the comprehensive comparison:
```bash
PYTHONPATH='.' python3 examples/adamw_comparison.py
```

Available examples:
- `examples/adamw_comparison.py` - Compare all three optimizers
- `examples/optimizer_linear_regression.py` - BinaryRateOptimizer demo
- `examples/search_algorithms_demo.py` - BinarySearch demo

---

## 🧪 Testing

### Run All Tests

```bash
# Using pytest (if installed)
pytest tests/ --cov=binary_search

# Manual validation
python3 -c "from binary_search import AdamW, BinaryRateOptimizer, BinarySearch; print('✓ All imports work')"
```

### Test Results
- **Total Tests:** 24
- **Passed:** 24 ✅
- **Failed:** 0
- **Details:** See [docs/REVIEW_RESULTS.md](docs/REVIEW_RESULTS.md)

---

## 🎓 When to Use Each Tool

### **AdamW with Binary Search** ⭐ RECOMMENDED
**Use when:**
- You want robust optimization without hyperparameter tuning
- Working with ill-conditioned problems
- Need adaptive learning rates
- Want production-ready optimization

**Benefits:**
- No manual learning rate tuning
- Adapts to problem scale
- Combines momentum + adaptive rates + optimal step size

### **BinaryRateOptimizer**
**Use when:**
- Simple/convex optimization problems
- You want transparent vanilla gradient descent
- Fast prototyping

### **BinarySearch Algorithms**
**Use when:**
- Searching sorted arrays
- Finding function roots
- Solving equations numerically

---

## 📖 Import Options

### New Way (Recommended - Explicit)
```python
from binary_search.optimizers import AdamW, BinaryRateOptimizer
from binary_search.algorithms import BinarySearch
```

### Old Way (Still Supported - Backward Compatible)
```python
from binary_search import AdamW, BinaryRateOptimizer, BinarySearch
```

Both work! Choose based on your preference.

---

## 🔧 Key Features

### AdamW Optimizer ⭐
- ✅ Adaptive moment estimation (momentum)
- ✅ Decoupled weight decay
- ✅ Binary search for learning rate
- ✅ No manual tuning required
- ✅ Parameter validation
- ✅ History tracking

### BinaryRateOptimizer
- ✅ Dynamic learning rate via binary search
- ✅ Two-phase approach (expansion + refinement)
- ✅ Convergence tolerance
- ✅ History tracking

### BinarySearch
- ✅ Array search (sorted lists)
- ✅ Function root finding
- ✅ Negative value support
- ✅ Configurable tolerance
- ✅ Reset functionality

---

## 📈 Version History

### v1.1.0 (2026-01-22) - Current
- ⭐ Added AdamW optimizer with binary search learning rate
- 🏗️ Restructured into modular architecture
- ✅ 24/24 tests passed
- 📚 Complete documentation (see [docs/](docs/))
- 🔄 Backward compatibility maintained

### v1.0.0 (Previous)
- Split into two packages
- Fixed critical bugs
- 81% test coverage

### v0.1 (Original)
- Single file, known bugs

---

## 🛠️ Dependencies

- Python 3.6+
- NumPy
- (Optional) Matplotlib for visualization

---

## 📄 License

[Your License Here]

---

## 🤝 Contributing

Contributions welcome! Please:
1. Read [docs/STRUCTURE_v1.1.md](docs/STRUCTURE_v1.1.md) for architecture
2. Check [docs/REVIEW_RESULTS.md](docs/REVIEW_RESULTS.md) for test standards
3. Maintain backward compatibility

---

## 📞 Support

- **Complete Documentation:** See [docs/](docs/) folder
- **Examples:** Check [examples/](examples/) directory
- **Test Results:** [docs/REVIEW_RESULTS.md](docs/REVIEW_RESULTS.md)
- **Architecture:** [docs/STRUCTURE_v1.1.md](docs/STRUCTURE_v1.1.md)

---

## ⭐ Highlights

**Why AdamW with Binary Search?**

Traditional optimizers require careful learning rate tuning:
```python
# Traditional approach - requires experimentation
optimizer = AdamW(lr=0.001)  # Too small? Too large? 🤷
```

Our innovation eliminates this:
```python
# Our approach - automatic tuning!
optimizer = AdamW(use_binary_search=True)  # Just works! ✨
```

**Result:** Robust optimization without hyperparameter headaches.

---

**Built with ❤️ following software engineering best practices**

**Status:** ✅ Production-Ready | **Version:** 1.1.0 | **Tests:** 24/24 Passed | **Docs:** [docs/](docs/)
