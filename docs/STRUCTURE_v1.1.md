# Binary Search Project Structure v1.1.0

## 📁 New Modular Architecture

The project has been restructured to separate concerns:

```
binary_search/
├── __init__.py           # Package entry (backward compatible)
├── optimizers.py         # ML optimizers (NEW!)
│   ├── BinaryRateOptimizer
│   └── AdamW            # NEW: AdamW + Binary Search
└── algorithms.py         # Binary search algorithms
    └── BinarySearch
```

## 🆕 What's New in v1.1.0

### **AdamW Optimizer** 
NEW optimizer combining:
- ✅ Adaptive Moment Estimation (Adam algorithm)
- ✅ Decoupled Weight Decay (AdamW improvement over Adam)
- ✅ Binary Search for Learning Rate (our innovation!)

**Benefits:**
- No manual learning rate tuning required
- Robust convergence across different problem scales
- Combines momentum + adaptive rates + optimal step size

### **Modular Structure**
Classes are now organized by function:
- **optimizers.py**: Machine learning gradient-based optimizers
- **algorithms.py**: Pure search/root-finding algorithms

## 📥 Import Options

### **New Recommended Way** (Explicit)
```python
from binary_search.optimizers import AdamW, BinaryRateOptimizer
from binary_search.algorithms import BinarySearch
```

### **Legacy Way** (Still Supported)
```python
from binary_search import BinaryRateOptimizer, AdamW, BinarySearch
```

Both work! Choose based on your preference.

## 🚀 Quick Start Examples

### Example 1: AdamW Optimizer
```python
import numpy as np
from binary_search.optimizers import AdamW

# Define problem
def cost(theta, X, y):
    return np.mean((X @ theta - y) ** 2)

def gradient(theta, X, y):
    return 2 * X.T @ (X @ theta - y) / len(y)

# Generate data
X = np.random.randn(100, 5)
y = X @ np.array([1, 2, 3, 4, 5])

# Optimize with AdamW (no learning rate tuning needed!)
optimizer = AdamW(max_iter=100, use_binary_search=True)
theta = optimizer.optimize(X, y, np.zeros(5), cost, gradient)
```

### Example 2: BinaryRateOptimizer
```python
from binary_search.optimizers import BinaryRateOptimizer

optimizer = BinaryRateOptimizer(max_iter=50, tol=1e-6)
theta = optimizer.optimize(X, y, initial_theta, cost, gradient)
```

### Example 3: Binary Search Algorithms
```python
from binary_search.algorithms import BinarySearch

# Find root: x^2 = 100
result = BinarySearch.search_for_function(
    y=100,
    function=lambda x: x**2,
    tolerance=1e-6
)
print(f"sqrt(100) ≈ {result}")  # 10.0
```

## 🔬 Full Comparison Example

Run the comprehensive comparison:
```bash
cd binary_search
PYTHONPATH='.' python3 examples/adamw_comparison.py
```

This compares:
1. BinaryRateOptimizer (vanilla GD + binary search)
2. AdamW with fixed learning rate
3. AdamW with binary search learning rate

**Typical Results:**
- BinaryRateOptimizer: Fast, ~7 iterations
- AdamW (fixed LR): Slow convergence if LR not tuned
- **AdamW (binary search)**: Best - no tuning + robust convergence

## 📊 When to Use Each Optimizer

### **BinaryRateOptimizer**
- ✅ Simple problems, convex optimization
- ✅ When you want transparency (vanilla gradient descent)
- ✅ Fast prototyping
- ❌ May struggle with ill-conditioned problems

### **AdamW (fixed LR)**
- ✅ When you have good learning rate from prior experiments
- ✅ Faster than binary search if LR is already tuned
- ❌ Requires hyperparameter experimentation

### **AdamW (binary search)** ⭐ **RECOMMENDED**
- ✅ **No learning rate tuning needed**
- ✅ Robust to problem scale
- ✅ Handles ill-conditioned problems better
- ✅ Works well out-of-the-box
- ❌ Slightly slower per iteration (binary search overhead)

## 🔧 AdamW Parameters

```python
AdamW(
    max_iter=100,              # Maximum iterations
    beta1=0.9,                 # First moment decay (momentum)
    beta2=0.999,               # Second moment decay (adaptive LR)
    epsilon=1e-8,              # Numerical stability
    weight_decay=0.01,         # L2 regularization strength
    tol=1e-6,                  # Convergence tolerance
    use_binary_search=True,    # Enable binary search for LR ⭐
    base_lr=0.001,             # Base LR (fallback if binary search off)
    expansion_factor=2.0,      # Binary search expansion rate
    binary_search_steps=10,    # Binary search refinement iterations
    verbose=True               # Print progress
)
```

**Key Parameter:** `use_binary_search`
- `True`: Adaptive learning rate (recommended!)
- `False`: Fixed `base_lr` (traditional AdamW)

## 📁 Project Files

### Core Module
- `binary_search/__init__.py` - Package entry point
- `binary_search/optimizers.py` - BinaryRateOptimizer + AdamW (543 lines)
- `binary_search/algorithms.py` - BinarySearch algorithms (644 lines)

### Examples
- `examples/adamw_comparison.py` - Full optimizer comparison
- `examples/optimizer_linear_regression.py` - BinaryRateOptimizer example
- `examples/search_algorithms_demo.py` - BinarySearch examples

### Tests (v1.0 refactored packages)
- `tests/binary_rate_optimizer/test_optimizer.py` (22 tests, 98% coverage)
- `tests/binary_search_algorithms/test_algorithms.py` (37 tests, 76% coverage)

### Consumer Tests
- `../binary_search_consumer/test_all_features.py` - External consumer tests

## 🐛 Redundancy Notes

**Found redundant code:**
- Original `binary_search/__init__.py` (558 lines) contains old versions of both classes
- Refactored versions in separate packages:
  - `binary_rate_optimizer/` (v1.0 refactored)
  - `binary_search_algorithms/` (v1.0 refactored)
- New v1.1 structure consolidates everything cleanly

**Migration Path:**
- v1.0: Separate packages (binary_rate_optimizer, binary_search_algorithms)
- v1.1: Unified package with submodules (optimizers, algorithms)
- Old code in `__init__.py` kept for backward compatibility

## 📈 Version History

**v1.1.0** (Current)
- ✅ Added AdamW optimizer with binary search
- ✅ Restructured into optimizers.py and algorithms.py
- ✅ Maintained backward compatibility
- ✅ Comprehensive comparison examples

**v1.0.0** (Refactored)
- Split into two packages
- Added tests (81% coverage)
- Fixed critical bugs (imports, overflow)

**v0.1** (Original)
- Single file with both classes
- Has known bugs

## 🎯 Recommendations

1. **New Projects**: Use AdamW with binary search
   ```python
   from binary_search.optimizers import AdamW
   optimizer = AdamW(use_binary_search=True)
   ```

2. **Existing Code**: Works unchanged (backward compatible)
   ```python
   from binary_search import BinaryRateOptimizer  # Still works!
   ```

3. **Search Problems**: Use BinarySearch algorithms
   ```python
   from binary_search.algorithms import BinarySearch
   ```

## 🔗 Next Steps

Try the comparison example:
```bash
cd /path/to/binary_search
PYTHONPATH='.' python3 examples/adamw_comparison.py
```

See which optimizer works best for your use case!
