# Migration Guide: binary_search → math_toolkit

**Version 2.0.0** - Package Reorganization

---

## 📋 Overview

The package has been reorganized to improve cohesion and clarity. The package is now called `math_toolkit` and is organized by functionality:

- **math_toolkit.binary_search** - Search algorithms
- **math_toolkit.optimization** - Gradient-based optimizers  
- **math_toolkit.linear_systems** - Linear system solvers

---

## ⚡ Quick Migration

### Old Code (v1.x)
```python
from binary_search import BinarySearch, BinaryRateOptimizer, AdamW
```

### New Code (v2.0+) - Recommended
```python
from math_toolkit.binary_search import BinarySearch
from math_toolkit.optimization import BinaryRateOptimizer, AdamW
```

### Backward Compatibility (Still Works)
```python
# This still works but shows deprecation warning
from binary_search import BinarySearch, BinaryRateOptimizer, AdamW
```

---

## 🔄 Complete Import Changes

| Old Import (v1.x) | New Import (v2.0+) | Status |
|-------------------|-------------------|--------|
| `from binary_search import BinarySearch` | `from math_toolkit.binary_search import BinarySearch` | ✅ Both work |
| `from binary_search import BinaryRateOptimizer` | `from math_toolkit.optimization import BinaryRateOptimizer` | ✅ Both work |
| `from binary_search import AdamW` | `from math_toolkit.optimization import AdamW` | ✅ Both work |
| `from binary_search.observer_tuning import ObserverAdamW` | `from math_toolkit.optimization import ObserverAdamW` | ✅ Both work |
| `from binary_search.linear_systems import BinaryGaussSeidel` | `from math_toolkit.linear_systems import BinaryGaussSeidel` | ✅ Both work |
| `from binary_search.algorithms import BinarySearch` | `from math_toolkit.binary_search import BinarySearch` | ⚠️ Update required |
| `from binary_search.optimizers import AdamW` | `from math_toolkit.optimization import AdamW` | ⚠️ Update required |

---

## 📦 Package Structure Changes

### Old Structure (v1.x)
```
binary_search/
├── __init__.py
├── algorithms.py          # BinarySearch
├── optimizers.py          # BinaryRateOptimizer, AdamW
├── observer_tuning.py     # ObserverAdamW
└── linear_systems/
    └── binary_gauss_seidel.py
```

### New Structure (v2.0+)
```
math_toolkit/
├── __init__.py            # Top-level exports
├── binary_search/
│   ├── __init__.py
│   └── algorithms.py      # BinarySearch
├── optimization/
│   ├── __init__.py
│   ├── gradient_descent.py       # BinaryRateOptimizer
│   ├── adaptive_optimizer.py     # AdamW
│   └── observer_tuning.py        # ObserverAdamW
└── linear_systems/
    ├── __init__.py
    └── iterative.py       # BinaryGaussSeidel
```

---

## 🛠️ Step-by-Step Migration

### Step 1: Update Imports

**Before:**
```python
from binary_search import BinarySearch
from binary_search import BinaryRateOptimizer
from binary_search import AdamW
from binary_search.observer_tuning import ObserverAdamW
from binary_search.linear_systems import BinaryGaussSeidel
```

**After:**
```python
from math_toolkit.binary_search import BinarySearch
from math_toolkit.optimization import BinaryRateOptimizer, AdamW, ObserverAdamW
from math_toolkit.linear_systems import BinaryGaussSeidel
```

### Step 2: Update setup.py/requirements.txt

**Before:**
```python
install_requires=[
    'binary-search',
]
```

**After:**
```python
install_requires=[
    'math-toolkit>=2.0.0',
]
```

### Step 3: Test Your Code

```bash
# Install new version
pip install --upgrade math-toolkit

# Run your tests
python -m pytest tests/

# Check for deprecation warnings
python -W all your_script.py
```

---

## 💡 Why This Change?

### Problem: Low Cohesion (v1.x)

The `binary_search` package contained:
- ✅ Binary search algorithms (cohesive!)
- ❌ Gradient descent optimizers (not really binary search)
- ❌ Linear system solvers (not binary search)

**Result:** Confusing package name and mixed responsibilities

### Solution: High Cohesion (v2.0+)

The `math_toolkit` package now clearly organizes by function:
- ✅ `math_toolkit.binary_search` - Pure search algorithms
- ✅ `math_toolkit.optimization` - ML/gradient descent
- ✅ `math_toolkit.linear_systems` - Linear solvers

**Result:** Clear organization, better discoverability

---

## 🔗 Backward Compatibility

### How It Works

A compatibility stub (`binary_search.py`) is provided at the root level:

```python
# binary_search.py (compatibility stub)
import warnings
warnings.warn("Use 'math_toolkit' instead", DeprecationWarning)
from math_toolkit import *
```

### When to Migrate

- **Immediately:** For new projects
- **v2.x releases:** Old imports work with deprecation warnings
- **v3.0+ (future):** Old imports will be removed

---

## 🚀 Examples

### Example 1: Binary Search

**Before (v1.x):**
```python
from binary_search import BinarySearch

searcher = BinarySearch()
result = searcher.search_for_function(y=100, function=lambda x: x**2)
print(f"sqrt(100) = {result}")
```

**After (v2.0+):**
```python
from math_toolkit.binary_search import BinarySearch

searcher = BinarySearch()
result = searcher.search_for_function(y=100, function=lambda x: x**2)
print(f"sqrt(100) = {result}")
```

### Example 2: Gradient Descent

**Before (v1.x):**
```python
from binary_search import BinaryRateOptimizer

optimizer = BinaryRateOptimizer(max_iter=50)
theta = optimizer.optimize(X, y, initial_theta, cost, gradient)
```

**After (v2.0+):**
```python
from math_toolkit.optimization import BinaryRateOptimizer

optimizer = BinaryRateOptimizer(max_iter=50)
theta = optimizer.optimize(X, y, initial_theta, cost, gradient)
```

### Example 3: AdamW Optimizer

**Before (v1.x):**
```python
from binary_search import AdamW

optimizer = AdamW(use_binary_search=True)
theta = optimizer.optimize(X, y, initial_theta, cost, gradient)
```

**After (v2.0+):**
```python
from math_toolkit.optimization import AdamW

optimizer = AdamW(use_binary_search=True)
theta = optimizer.optimize(X, y, initial_theta, cost, gradient)
```

### Example 4: Linear Systems

**Before (v1.x):**
```python
from binary_search.linear_systems import BinaryGaussSeidel

solver = BinaryGaussSeidel(max_iterations=1000)
x = solver.solve(A, b)
```

**After (v2.0+):**
```python
from math_toolkit.linear_systems import BinaryGaussSeidel

solver = BinaryGaussSeidel(max_iterations=1000)
x = solver.solve(A, b)
```

---

## 📚 Documentation Updates

All documentation has been updated:
- [README.md](../README.md) - Package overview
- [ALGORITHM_SELECTION_GUIDE.md](ALGORITHM_SELECTION_GUIDE.md) - When to use each algorithm
- [SCALABILITY_BENCHMARK.md](SCALABILITY_BENCHMARK.md) - Performance comparisons
- [OPTIMIZER_BENCHMARK.md](OPTIMIZER_BENCHMARK.md) - Optimizer benchmarks

---

## ❓ FAQ

### Q: Will my old code break?

**A:** No, old imports still work in v2.x but show deprecation warnings.

### Q: When will old imports be removed?

**A:** Planned for v3.0 (not yet scheduled). You'll have plenty of time to migrate.

### Q: Do I need to change anything else besides imports?

**A:** No! All class names, method signatures, and functionality remain identical.

### Q: Can I use both old and new imports in the same project?

**A:** Yes, but not recommended. Choose one style for consistency.

### Q: How do I silence deprecation warnings during migration?

**A:**
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

### Q: What if I find issues after migrating?

**A:** Please report on GitHub: https://github.com/JosueAmaral15/binary-search/issues

---

## 🔧 Troubleshooting

### Issue: Import errors after upgrade

**Solution:**
```bash
# Reinstall package
pip uninstall binary-search math-toolkit
pip install math-toolkit

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -r {} +
```

### Issue: Tests fail after migration

**Solution:**
Update test imports:
```python
# Old
from binary_search import BinarySearch

# New
from math_toolkit.binary_search import BinarySearch
```

### Issue: Module not found

**Solution:**
Check you're using the correct package name:
```bash
pip list | grep math-toolkit
```

---

## 📞 Support

- **Documentation:** [docs/](.)
- **Examples:** [examples/](../examples/)
- **Issues:** https://github.com/JosueAmaral15/binary-search/issues
- **Discussions:** https://github.com/JosueAmaral15/binary-search/discussions

---

## ✅ Migration Checklist

- [ ] Update all `from binary_search import ...` statements
- [ ] Update `setup.py`/`requirements.txt`
- [ ] Update documentation/comments
- [ ] Run tests to verify nothing broke
- [ ] Check for deprecation warnings
- [ ] Update CI/CD configuration if needed
- [ ] Inform team members of changes

---

**Last Updated:** 2026-01-26  
**Package Version:** 2.0.0
