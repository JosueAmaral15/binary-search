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
│   └── algorithms.py
├── optimization/            # Gradient-based optimizers
│   ├── gradient_descent.py
│   ├── adaptive_optimizer.py
│   └── observer_tuning.py
└── linear_systems/          # Linear system solvers
    └── iterative.py
```

### 📦 **Clear Module Organization**
- **math_toolkit.binary_search** - Binary search algorithms with tolerance-based comparisons
- **math_toolkit.optimization** - ML optimizers (BinaryRateOptimizer, AdamW, ObserverAdamW)
- **math_toolkit.linear_systems** - Iterative linear solvers (BinaryGaussSeidel)

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

solver = BinaryGaussSeidel(max_iterations=1000, tolerance=1e-6)
x = solver.solve(A, b)
```

---

## 📚 What's Included

### 1. **Binary Search Algorithms** (`math_toolkit.binary_search`)
- **BinarySearch**: Advanced search with tolerance-based comparisons
- Array search, function root finding, stepped search
- Mathematical comparison functions

### 2. **Optimization Algorithms** (`math_toolkit.optimization`)
- **BinaryRateOptimizer**: Gradient descent with binary search learning rate (10× faster than AdamW)
- **AdamW**: Adaptive moment estimation with weight decay and binary search
- **ObserverAdamW**: Parallel hyperparameter tuning with observer pattern

### 3. **Linear System Solvers** (`math_toolkit.linear_systems`)
- **BinaryGaussSeidel**: Iterative solver with binary search optimizations
- Polynomial regression support
- Large system optimization (10×10 to 100×100)

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
