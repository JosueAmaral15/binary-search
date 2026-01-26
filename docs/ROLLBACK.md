# Rollback Plan - Binary Search Refactoring v1.0

## When to Rollback

Rollback if any of these occur:

- [ ] Critical bugs discovered in production
- [ ] Performance regression > 50%
- [ ] Import errors breaking existing code
- [ ] Tests fail in production environment
- [ ] User reports package unusable

## Quick Rollback (<5 minutes)

### Step 1: Switch to main branch
```bash
git checkout main
```

### Step 2: Verify old code works
```bash
python3 -c "from math_toolkit.optimization import BinaryRateOptimizer, BinarySearch; print('OK')"
```

### Step 3: Delete COM branch (optional)
```bash
git branch -D COM-5bc5a2da-c9a4-462c-bcfb-96a8cd57dce7
```

## Partial Rollback Options

### Option A: Use only one package

If one package works but the other doesn't:

```bash
# Use Binary Rate Optimizer only
cd binary_rate_optimizer
pip install -e .

# Use Binary Search Algorithms only  
cd binary_search_algorithms
pip install -e .
```

### Option B: Fix and continue

If bug is minor:
1. Create hotfix branch
2. Fix the bug
3. Run tests
4. Deploy fix

## Recovery Testing

After rollback, verify:

```bash
# Test original package works
python3 -c "
from math_toolkit.optimization import BinaryRateOptimizer, BinarySearch
import numpy as np

# Test optimizer
opt = BinaryRateOptimizer()
print('BinaryRateOptimizer: OK')

# Test search
searcher = BinarySearch()
print('BinarySearch: OK')
"
```

## Known Limitations of v0.1 (Old Version)

If you rollback, remember these bugs exist:

1. **Missing imports**: `ceil`/`floor` cause NameError
   - Workaround: Don't use round_to_int parameter

2. **Overflow bug**: `search_for_function` crashes with exp()
   - Workaround: Avoid exponential functions

3. **No validation**: Invalid inputs cause cryptic errors
   - Workaround: Validate inputs manually

## Migration Back from v1.0 to v0.1

If users migrated to v1.0 and need to rollback:

```python
# v1.0 code (won't work after rollback)
from binary_rate_optimizer import BinaryRateOptimizer
from binary_search_algorithms import BinarySearch

# Change to v0.1 imports
from math_toolkit.optimization import BinaryRateOptimizer, BinarySearch
```

## Rollback Decision Matrix

| Situation | Action | Time |
|-----------|--------|------|
| Import error | Quick rollback | 5 min |
| One package broken | Use other package | 10 min |
| Minor bug | Hotfix | 30-60 min |
| Major design flaw | Full rollback | 5 min |
| Performance issue | Investigate first | 1-2 hours |

## Post-Rollback Actions

1. Document what went wrong in DECISIONS.md
2. Create GitHub issue with details
3. Notify users via README
4. Plan fix for next version

## Prevention

To minimize rollback risk:

- ✅ All 59 tests pass
- ✅ 81% code coverage
- ✅ Examples tested manually
- ✅ CI/CD checks on all Python versions
- ✅ Documentation complete

## Emergency Contact

If rollback needed urgently:
1. Switch to main branch immediately
2. Document issues
3. Inform users
4. Fix calmly, don't rush

## Success Metrics (No Rollback Needed If)

- ✅ All tests pass in production
- ✅ No import errors reported
- ✅ Performance acceptable (< 2x slower OK)
- ✅ Users can install and use both packages
- ✅ Documentation accurate

---

**Rollback is OK**: It's better to rollback quickly than to struggle with broken code. The v0.1 code still works (with known bugs).

**Remember**: The goal is working software, not perfect software. If v1.0 doesn't work for you, v0.1 is always available.
