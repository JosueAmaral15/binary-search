# Package Reorganization Action Plan

**Date:** 2026-01-26  
**Objective:** Reorganize package structure to improve cohesion by separating concerns into distinct modules

---

## 🎯 Goals

1. **Improve cohesion** - Group related functionality together
2. **Maintain backward compatibility** - Old imports still work
3. **Clear separation of concerns** - Search, Optimization, Linear Systems
4. **Better discoverability** - Intuitive package structure

---

## 📊 Current Structure

```
binary_search/                    # Top-level package
├── __init__.py                   # Exports: BinaryRateOptimizer, AdamW, BinarySearch
├── algorithms.py                 # BinarySearch class
├── optimizers.py                 # BinaryRateOptimizer, AdamW
├── observer_tuning.py            # ObserverAdamW
└── linear_systems/
    ├── __init__.py
    └── binary_gauss_seidel.py    # BinaryGaussSeidel

tests/
├── binary_search_algorithms/     # Tests for BinarySearch
└── binary_rate_optimizer/        # Tests for optimizers

examples/
├── search_algorithms_demo.py
├── optimizer_*.py
├── adamw_*.py
├── test_linear_systems_*.py
└── test_observer_adamw.py
```

**Problems:**
- ❌ Package name "binary_search" doesn't reflect all functionality
- ❌ Optimizers mixed with search algorithms in same top-level
- ❌ Low cohesion - different domains mixed together
- ❌ Test structure doesn't match package structure

---

## 🎯 Target Structure

```
math_toolkit/                          # NEW: Top-level package
├── __init__.py                        # Backward compatibility exports
├── binary_search/                     # Search algorithms module
│   ├── __init__.py
│   └── algorithms.py                  # BinarySearch class
├── optimization/                      # ML optimizers module
│   ├── __init__.py
│   ├── gradient_descent.py            # BinaryRateOptimizer
│   ├── adaptive_optimizer.py          # AdamW
│   └── observer_tuning.py             # ObserverAdamW
└── linear_systems/                    # Linear solvers module
    ├── __init__.py
    └── iterative.py                   # BinaryGaussSeidel (renamed)

tests/
├── binary_search/                     # Tests for search algorithms
│   └── test_algorithms.py
├── optimization/                      # Tests for optimizers
│   ├── test_gradient_descent.py
│   ├── test_adaptive_optimizer.py
│   └── test_observer_tuning.py
└── linear_systems/                    # Tests for linear solvers
    └── test_iterative.py

examples/
├── binary_search_examples/            # Search examples
│   └── search_algorithms_demo.py
├── optimization_examples/             # Optimizer examples
│   ├── optimizer_linear_regression.py
│   ├── adamw_comparison.py
│   └── test_observer_adamw.py
└── linear_systems_examples/           # Linear system examples
    ├── test_linear_systems_phase1.py
    ├── test_linear_systems_phase2.py
    └── test_linear_systems_100x100.py
```

---

## 📝 Import Changes

### New Imports (Recommended)

```python
# Search algorithms
from math_toolkit.binary_search import BinarySearch

# Optimizers
from math_toolkit.optimization import BinaryRateOptimizer, AdamW, ObserverAdamW

# Linear solvers
from math_toolkit.linear_systems import BinaryGaussSeidel
```

### Backward Compatible (Still Works)

```python
# OLD CODE - Still functional via __init__.py
from binary_search import BinaryRateOptimizer, AdamW, BinarySearch
```

**Note:** We'll create a `binary_search.py` stub at project root that imports from `math_toolkit` for backward compatibility.

---

## 🔧 Implementation Steps

### Phase 1: Create New Structure ✅

1. Create `math_toolkit/` directory
2. Create subdirectories:
   - `math_toolkit/binary_search/`
   - `math_toolkit/optimization/`
   - `math_toolkit/linear_systems/`
3. Create all `__init__.py` files

### Phase 2: Move and Rename Files ✅

**Binary Search:**
- Move: `binary_search/algorithms.py` → `math_toolkit/binary_search/algorithms.py`

**Optimization:**
- Move: `binary_search/optimizers.py` → Split into:
  - `math_toolkit/optimization/gradient_descent.py` (BinaryRateOptimizer)
  - `math_toolkit/optimization/adaptive_optimizer.py` (AdamW)
- Move: `binary_search/observer_tuning.py` → `math_toolkit/optimization/observer_tuning.py`

**Linear Systems:**
- Move: `binary_search/linear_systems/binary_gauss_seidel.py` → `math_toolkit/linear_systems/iterative.py`

### Phase 3: Update Imports ✅

**In moved files:**
- Update internal imports to use new paths
- Fix relative imports

**Create backward compatibility:**
- `math_toolkit/__init__.py` - Export all classes
- Create stub `binary_search.py` at root that redirects to `math_toolkit`

### Phase 4: Reorganize Tests ✅

**Move tests:**
- `tests/binary_search_algorithms/` → `tests/binary_search/`
- `tests/binary_rate_optimizer/test_optimizer.py` → `tests/optimization/test_gradient_descent.py`
- Create new test files:
  - `tests/optimization/test_adaptive_optimizer.py`
  - `tests/optimization/test_observer_tuning.py`
  - `tests/linear_systems/test_iterative.py`

**Update test imports:**
- Change all `from binary_search import ...` → `from math_toolkit... import ...`

### Phase 5: Reorganize Examples ✅

**Create directories:**
- `examples/binary_search_examples/`
- `examples/optimization_examples/`
- `examples/linear_systems_examples/`

**Move files:**
- `search_algorithms_demo.py` → `binary_search_examples/`
- `optimizer_*.py`, `adamw_*.py`, `test_observer_adamw.py` → `optimization_examples/`
- `test_linear_systems_*.py` → `linear_systems_examples/`

**Update example imports:**
- Change to use new `math_toolkit` paths

### Phase 6: Update Configuration Files ✅

**setup.py:**
- Change `packages` to find `math_toolkit`
- Update package name: `name='math-toolkit'`
- Update entry points

**README.md:**
- Update import examples
- Update package name
- Add migration guide

**Documentation:**
- Update all code examples in `docs/`
- Add `MIGRATION_GUIDE.md`

### Phase 7: Update GitHub/CI ✅

**.github/workflows:**
- Update paths if necessary
- Verify CI still works

**Repository name:**
- Consider renaming repository to `math-toolkit` (optional)

### Phase 8: Cleanup ✅

- Remove old `binary_search/` directory (after verification)
- Remove `binary_rate_optimizer/` directory if exists
- Remove `binary_search_algorithms/` directory if exists
- Clean build artifacts
- Update `.gitignore` if needed

### Phase 9: Testing ✅

Run comprehensive tests:
```bash
# Test new imports
python -c "from math_toolkit.binary_search import BinarySearch"
python -c "from math_toolkit.optimization import BinaryRateOptimizer, AdamW"
python -c "from math_toolkit.linear_systems import BinaryGaussSeidel"

# Test backward compatibility
python -c "from binary_search import BinarySearch, BinaryRateOptimizer"

# Run all tests
pytest tests/

# Test examples
python examples/binary_search_examples/search_algorithms_demo.py
python examples/optimization_examples/optimizer_linear_regression.py
```

### Phase 10: Documentation & Commit ✅

1. Create `MIGRATION_GUIDE.md`
2. Update `CHANGELOG.md`
3. Update version to `2.0.0` (breaking change in package structure)
4. Commit with detailed message
5. Push to repository

---

## ⚠️ Breaking Changes

### What Breaks:

```python
# This will require stub/redirect
from binary_search import BinaryRateOptimizer  # OLD

# This breaks completely (never was recommended)
import binary_search.optimizers  # OLD internal import
```

### What Still Works:

```python
# Backward compatible via stub
from binary_search import BinarySearch, BinaryRateOptimizer, AdamW

# New recommended way
from math_toolkit.binary_search import BinarySearch
from math_toolkit.optimization import BinaryRateOptimizer
```

---

## 📋 File Mapping

| Old Path | New Path | Notes |
|----------|----------|-------|
| `binary_search/__init__.py` | `math_toolkit/__init__.py` | Backward compat exports |
| `binary_search/algorithms.py` | `math_toolkit/binary_search/algorithms.py` | Direct move |
| `binary_search/optimizers.py` | Split into 2 files | See below |
| → BinaryRateOptimizer | `math_toolkit/optimization/gradient_descent.py` | Extract class |
| → AdamW | `math_toolkit/optimization/adaptive_optimizer.py` | Extract class |
| `binary_search/observer_tuning.py` | `math_toolkit/optimization/observer_tuning.py` | Direct move |
| `binary_search/linear_systems/binary_gauss_seidel.py` | `math_toolkit/linear_systems/iterative.py` | Rename file |

---

## 🎯 Success Criteria

- [ ] All tests pass (24/24)
- [ ] Old imports still work (backward compatibility)
- [ ] New imports work correctly
- [ ] Examples run without errors
- [ ] Documentation updated
- [ ] No duplicate code
- [ ] Clear package structure
- [ ] CI/CD passes

---

## 🚨 Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking user code | Maintain backward compatibility via stubs |
| Import cycles | Careful dependency management |
| Test failures | Run tests after each phase |
| Lost functionality | Verify all classes exported correctly |
| Documentation out of sync | Update docs in same commit |

---

## 📊 Timeline Estimate

- **Phase 1-2:** 15 minutes (structure + move files)
- **Phase 3:** 20 minutes (update imports)
- **Phase 4-5:** 20 minutes (reorganize tests + examples)
- **Phase 6:** 15 minutes (configuration)
- **Phase 7:** 5 minutes (CI)
- **Phase 8:** 5 minutes (cleanup)
- **Phase 9:** 15 minutes (testing)
- **Phase 10:** 10 minutes (documentation)

**Total:** ~2 hours

---

## ✅ Ready to Execute

All questions answered:
- ✅ Top-level: `math_toolkit/`
- ✅ Module names: Descriptive (gradient_descent.py, etc.)
- ✅ Import paths: `from math_toolkit.* import ...`
- ✅ Backward compatibility: Yes, maintained
- ✅ Tests: Reorganized to match structure
- ✅ Examples: Grouped by category

**Status:** Ready to implement

**Next Step:** Execute phases 1-10 sequentially with verification at each step.
