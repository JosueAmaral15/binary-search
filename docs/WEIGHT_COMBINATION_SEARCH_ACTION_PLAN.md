# WeightCombinationSearch Implementation - Action Plan

**Date:** 2026-02-01  
**Feature:** Combinatorial weight search with binary refinement  
**Protocol:** Simplicity 3 - Small commits, test-driven, pragmatic  

---

## 📋 REQUIREMENTS SPECIFICATION

### User Requirements
- Find optimal weights W such that: A·W ≈ Target
- Use truth table (test all 2^N-1 combinations per cycle)
- Apply binary refinement via Weighted Possibility Number (WPN)
- Support N parameters (3, 4, 5, ... dimensions)
- Return final weights as array
- Optional: Return full truth table (DataFrame/CSV)

### Algorithm Specification

**Input:**
- `coefficients`: Array [A, B, C, ...] of N parameters
- `target`: Desired result value
- `tolerance`: Convergence threshold (default: 0)
- `max_iter`: Maximum iterations (default: 32)
- `initial_wpn`: Starting WPN value (default: 1.0)
- `wpn_bounds`: (min, max) tuple (default: (-inf, inf))

**Output:**
- `weights`: Array [W1, W2, ..., WN] of optimal weights
- `truth_table` (optional): DataFrame with all tested combinations

**Core Formula:**
```python
result = sum(param[i] * (W[i] if W[i]!=0 else (1 if combo[i] else W[i])) * (WPN if combo[i] else 1)
             for i in range(N))
```

**Key Rules:**
1. Cycle size: 2^N - 1 (exclude all-zeros combo)
2. Weight update: `W[i] *= WPN` if selected in winner
3. WPN adjustment:
   - All Δ_cond < 0 → WPN *= 2
   - All Δ_cond > 0 OR mixed → WPN /= 2
4. Tie breaking: Choose result > target (higher value)
5. Convergence: Stop when Δ_abs ≤ tolerance

---

## 🎯 IMPLEMENTATION PLAN

### Phase 1: Core Algorithm (Task 1)
**Goal:** Implement basic WeightCombinationSearch class

**File to create:** `math_toolkit/binary_search/combinatorial.py`

**Steps:**
1. Create class structure
2. Implement `find_optimal_weights()` method
3. Implement cycle logic (2^N-1 combinations)
4. Implement weight update mechanism
5. Implement WPN adjustment logic
6. Add convergence checking

**Deliverable:** Working class with core algorithm

**Time estimate:** 1-2 hours

---

### Phase 2: Truth Table Support (Task 2)
**Goal:** Add truth table tracking and export

**File to modify:** `math_toolkit/binary_search/combinatorial.py`

**Steps:**
1. Add truth table tracking in main loop
2. Implement DataFrame export
3. Implement CSV export
4. Add optional return of truth table
5. Format truth table with clear column names

**Deliverable:** Truth table functionality complete

**Time estimate:** 30-45 minutes

---

### Phase 3: Testing (Task 3)
**Goal:** Comprehensive test suite

**File to create:** `tests/binary_search/test_combinatorial.py`

**Tests to implement:**
1. **Test basic functionality** (user's example: A=15, B=47, C=-12, Target=28)
2. **Test 2 parameters** (simpler case)
3. **Test 4 parameters** (larger case)
4. **Test convergence** (tolerance checking)
5. **Test WPN bounds** (min/max limits)
6. **Test tie breaking** (multiple combos with same Δ)
7. **Test truth table export** (DataFrame and CSV)
8. **Test edge cases:**
   - All parameters positive
   - All parameters negative
   - Mixed positive/negative
   - Target = 0
   - Target unreachable (max_iter reached)

**Deliverable:** 15-20 comprehensive tests

**Time estimate:** 1-1.5 hours

---

### Phase 4: Integration (Task 4)
**Goal:** Integrate with existing package

**Files to modify:**
- `math_toolkit/binary_search/__init__.py` - Export new class
- `README.md` - Add usage example
- `docs/ALGORITHM_SELECTION_GUIDE.md` - Add when to use

**Steps:**
1. Add import in `__init__.py`
2. Update README with example
3. Update algorithm selection guide
4. Create example script in `examples/`

**Deliverable:** Integrated and documented

**Time estimate:** 30 minutes

---

### Phase 5: Visualization (Task 5 - Optional)
**Goal:** Add visualization support

**File to create:** Example in existing visualization module

**Steps:**
1. Create plot function for truth table
2. Show convergence progression
3. Show WPN evolution

**Deliverable:** Visualization examples

**Time estimate:** 30-45 minutes (OPTIONAL)

---

## 📊 TASK BREAKDOWN

### Task 1: Core Algorithm Implementation ✅ REQUIRED

**File:** `math_toolkit/binary_search/combinatorial.py`

**Code structure:**
```python
class WeightCombinationSearch:
    """
    Find optimal weights using combinatorial search with binary refinement.
    
    Uses truth table to test all 2^N-1 combinations at each refinement level,
    then applies binary search via Weighted Possibility Number (WPN) to 
    converge on best solution.
    
    Algorithm:
    1. Test all 2^N-1 combinations (exclude all-zeros)
    2. Find winner (minimum absolute difference)
    3. Update weights based on winner combo
    4. Adjust WPN based on conditional differences
    5. Repeat until convergence or max iterations
    """
    
    def __init__(self, tolerance=0, max_iter=32, initial_wpn=1.0,
                 wpn_bounds=(-np.inf, np.inf), verbose=False):
        """Initialize search parameters"""
        
    def find_optimal_weights(self, coefficients, target, 
                            return_truth_table=False,
                            save_csv=False, csv_filename='truth_table.csv'):
        """Main search method"""
        
    def _generate_combinations(self, n_params):
        """Generate all 2^N-1 combinations (exclude all zeros)"""
        
    def _calculate_result(self, coefficients, weights, combo, wpn):
        """Calculate result using core formula"""
        
    def _find_winner(self, cycle_results):
        """Find winner with tie breaking (result > target)"""
        
    def _adjust_wpn(self, cycle_results, current_wpn):
        """Adjust WPN based on conditional differences"""
```

**Checklist:**
- [ ] Create class with __init__
- [ ] Implement find_optimal_weights()
- [ ] Implement _generate_combinations()
- [ ] Implement _calculate_result() with exact formula
- [ ] Implement _find_winner() with tie breaking
- [ ] Implement _adjust_wpn()
- [ ] Add convergence check
- [ ] Add logging support (verbose mode)
- [ ] Add docstrings
- [ ] Manual test with user's example

**Success criteria:**
- User's example (A=15, B=47, C=-12, Target=28) produces correct weights
- Converges in expected number of iterations
- WPN adjusts correctly

---

### Task 2: Truth Table Support ✅ REQUIRED

**Modifications to:** `math_toolkit/binary_search/combinatorial.py`

**Implementation:**
```python
def find_optimal_weights(self, ...):
    # ... existing code ...
    
    # Track truth table
    truth_table = []
    
    for cycle in range(max_iter):
        for line_num, combo in enumerate(combinations):
            # Calculate result
            # ...
            
            # Record in truth table
            truth_table.append({
                'cycle': cycle,
                'line': line_num + 1,
                'combo': combo,
                'weights_before': W.copy(),
                'wpn': WPN,
                'result': result,
                'delta_abs': delta_abs,
                'delta_cond': delta_cond,
                'is_winner': False  # Updated later
            })
        
        # Mark winner
        truth_table[winner_index]['is_winner'] = True
    
    # Export if requested
    if save_csv:
        self._save_truth_table_csv(truth_table, csv_filename)
    
    if return_truth_table:
        df = pd.DataFrame(truth_table) if HAS_PANDAS else truth_table
        return W, df
    
    return W
```

**Checklist:**
- [ ] Add truth_table tracking list
- [ ] Record all tested combinations
- [ ] Mark winner in each cycle
- [ ] Implement _save_truth_table_csv()
- [ ] Implement DataFrame conversion (if pandas available)
- [ ] Handle pandas not installed gracefully
- [ ] Add clear column names and formatting

**Success criteria:**
- Truth table contains all 2^N-1 * num_cycles entries
- CSV export works correctly
- DataFrame export works (if pandas available)
- Winner marked correctly in each cycle

---

### Task 3: Comprehensive Testing ✅ REQUIRED

**File:** `tests/binary_search/test_combinatorial.py`

**Test structure:**
```python
import pytest
import numpy as np
from math_toolkit.binary_search import WeightCombinationSearch

class TestWeightCombinationSearchBasic:
    """Basic functionality tests"""
    
    def test_user_example_3_params(self):
        """Test with user's exact example: A=15, B=47, C=-12, Target=28"""
        
    def test_simple_2_params(self):
        """Test with 2 parameters"""
        
    def test_4_params(self):
        """Test with 4 parameters"""

class TestConvergence:
    """Test convergence behavior"""
    
    def test_exact_convergence(self):
        """Test when exact solution exists"""
        
    def test_tolerance_convergence(self):
        """Test convergence with tolerance"""
        
    def test_max_iter_reached(self):
        """Test when max iterations reached"""

class TestWPNBehavior:
    """Test WPN adjustment logic"""
    
    def test_wpn_doubling(self):
        """Test WPN doubles when all Δ_cond < 0"""
        
    def test_wpn_halving(self):
        """Test WPN halves when all Δ_cond > 0 or mixed"""
        
    def test_wpn_bounds(self):
        """Test WPN respects min/max bounds"""

class TestTieBreaking:
    """Test tie breaking logic"""
    
    def test_tie_break_higher_result(self):
        """Test chooses result > target when tied"""

class TestTruthTable:
    """Test truth table functionality"""
    
    def test_truth_table_return(self):
        """Test truth table return as DataFrame"""
        
    def test_truth_table_csv(self):
        """Test CSV export"""
        
    def test_truth_table_content(self):
        """Test truth table has correct entries"""

class TestEdgeCases:
    """Test edge cases"""
    
    def test_all_positive_params(self):
        """Test with all positive parameters"""
        
    def test_all_negative_params(self):
        """Test with all negative parameters"""
        
    def test_target_zero(self):
        """Test with target = 0"""
        
    def test_single_param(self):
        """Test with N=1 (edge case)"""
```

**Checklist:**
- [ ] Create test file
- [ ] Implement basic functionality tests (3)
- [ ] Implement convergence tests (3)
- [ ] Implement WPN behavior tests (3)
- [ ] Implement tie breaking tests (1)
- [ ] Implement truth table tests (3)
- [ ] Implement edge case tests (4)
- [ ] All tests pass
- [ ] Test coverage > 90%

**Success criteria:**
- All 17+ tests passing
- User's exact example produces correct result
- Edge cases handled properly
- No regressions

---

### Task 4: Integration & Documentation ✅ REQUIRED

**Files to modify:**
1. `math_toolkit/binary_search/__init__.py`
2. `README.md`
3. `docs/ALGORITHM_SELECTION_GUIDE.md`
4. `examples/combinatorial_search_example.py` (NEW)

**Implementation:**

**1. Update `__init__.py`:**
```python
from .combinatorial import WeightCombinationSearch

__all__ = [
    'BinarySearch',
    'WeightCombinationSearch',  # NEW
]
```

**2. Update `README.md`:**
```markdown
### WeightCombinationSearch

Find optimal weights for linear combinations using combinatorial search:

```python
from math_toolkit.binary_search import WeightCombinationSearch

# Find weights W such that: 15*W1 + 47*W2 + (-12)*W3 ≈ 28
search = WeightCombinationSearch(tolerance=0, max_iter=32)
weights = search.find_optimal_weights(
    coefficients=[15, 47, -12],
    target=28
)
# Result: [0.5, 0.5, 0.25] approximately
```
```

**3. Update `ALGORITHM_SELECTION_GUIDE.md`:**
```markdown
### When to Use WeightCombinationSearch

**Use for:**
- Finding optimal weights for linear combinations
- Feature selection (which parameters contribute)
- Parameter tuning with discrete search space
- Problems where gradient information not available

**Don't use for:**
- Large N (N > 10, becomes exponentially slow)
- Continuous optimization (use BinaryRateOptimizer instead)
- Linear systems Ax=b (use BinaryGaussSeidel instead)
```

**4. Create example script:**
```python
# examples/combinatorial_search_example.py
"""
Example: Using WeightCombinationSearch for parameter weight finding.
"""
```

**Checklist:**
- [ ] Update __init__.py
- [ ] Add README section
- [ ] Update algorithm selection guide
- [ ] Create example script
- [ ] Test imports work
- [ ] Documentation clear and accurate

---

## 🔄 WORKFLOW (Simplicity 3 Protocol)

### Step-by-Step Process

**1. Task 1: Core Implementation**
```bash
# Create file
# Implement class
# Manual test
git add math_toolkit/binary_search/combinatorial.py
git commit -m "Add WeightCombinationSearch core algorithm

Implements combinatorial search with binary refinement:
- 2^N-1 combinations per cycle
- Weight update based on winner
- WPN adjustment (double/halve)
- User's example verified manually"
git push
```

**2. Task 2: Truth Table**
```bash
# Add truth table tracking
# Test CSV export
git add math_toolkit/binary_search/combinatorial.py
git commit -m "Add truth table support to WeightCombinationSearch

Features:
- Track all tested combinations
- Export to DataFrame
- Export to CSV
- Optional return of truth table"
git push
```

**3. Task 3: Testing**
```bash
# Write tests
# Run: pytest tests/binary_search/test_combinatorial.py -v
# All tests pass
git add tests/binary_search/test_combinatorial.py
git commit -m "Add comprehensive tests for WeightCombinationSearch

17+ tests covering:
- Basic functionality (user's example)
- Convergence behavior
- WPN adjustment logic
- Tie breaking
- Truth table export
- Edge cases

All tests passing ✅"
git push
```

**4. Task 4: Integration**
```bash
# Update __init__.py, README, docs, examples
# Test imports
git add math_toolkit/binary_search/__init__.py README.md docs/ examples/
git commit -m "Integrate WeightCombinationSearch into package

- Export in __init__.py
- Add README example
- Update algorithm selection guide
- Add example script

Feature complete and documented ✅"
git push
```

**5. Final: Run full test suite**
```bash
pytest tests/ -v
# All 271+ tests pass (254 + 17 new)
git tag v3.1.0-combinatorial
git push origin v3.1.0-combinatorial
```

---

## ✅ SUCCESS CRITERIA

### Functional Requirements
- [x] Algorithm specification understood
- [ ] Core algorithm implemented correctly
- [ ] User's example produces correct result
- [ ] Truth table export works
- [ ] All tests pass (17+)
- [ ] Integrated into package
- [ ] Documentation complete

### Quality Requirements
- [ ] Code follows package style
- [ ] Docstrings complete
- [ ] Type hints where appropriate
- [ ] Logging support (verbose mode)
- [ ] Error handling (invalid inputs)
- [ ] Test coverage > 90%

### Protocol Compliance
- [ ] Small, focused commits
- [ ] Test after each change
- [ ] No breaking changes
- [ ] Clear documentation
- [ ] Following Simplicity 3 Protocol

---

## 📈 ESTIMATED TIMELINE

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| 1 | Core Algorithm | 1-2h | 1-2h |
| 2 | Truth Table | 0.5-0.75h | 1.5-2.75h |
| 3 | Testing | 1-1.5h | 2.5-4.25h |
| 4 | Integration | 0.5h | 3-4.75h |
| 5 | Visualization (Optional) | 0.5-0.75h | 3.5-5.5h |

**Total: 3-5 hours** (without optional visualization)

---

## 🚀 READY TO START

**First step:** Implement Task 1 (Core Algorithm)

**Confirm to proceed?** ✅
