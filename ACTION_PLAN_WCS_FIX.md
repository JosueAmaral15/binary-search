# WeightCombinationSearch - Bug Fix Action Plan

## Date: 2026-02-03

## Problem Identified

**BUG:** History not recorded when early convergence occurs

### Symptoms
1. `history['cycles']` shows 0 entries even when optimization completes successfully
2. `history['wpn_evolution']` is empty
3. `history['best_delta_evolution']` is empty
4. Benchmark comparison was invalid because iteration count was always 0

### Root Cause
In `combinatorial.py` lines 537-580:
```python
# Check convergence
if winner['delta_abs'] <= tolerance:
    if self.verbose:
        logger.info(f"\n✅ Converged! Δ={winner['delta_abs']:.4f} ≤ {tolerance}")
    
    # Update weights one final time
    for i in range(n_params):
        if winner['combo'][i]:
            if W[i] == 0:
                W[i] = 1
            W[i] *= WPN
    
    # Save truth table if requested
    if save_csv:
        self._save_truth_table_csv(truth_table, csv_filename)
    
    if return_truth_table:
        df = self._format_truth_table(truth_table)
        return W, df
    
    return W  # ← RETURNS HERE (line 556)

# ... (code continues)

# Track history ← NEVER REACHED IF CONVERGED
self.history['cycles'].append(cycle + 1)       # line 578
self.history['wpn_evolution'].append(WPN)       # line 579
self.history['best_delta_evolution'].append(winner['delta_abs'])  # line 580
```

**The function returns at line 556 BEFORE recording history at lines 578-580.**

---

## Action Plan

### Phase 1: Fix History Recording ✅ PRIORITY

**Task 1.1: Move history recording before convergence check**

Current flow:
```
for cycle in range(max_iter):
    1. Generate combinations
    2. Find winner
    3. Check convergence → RETURN if converged ← BUG HERE
    4. Update weights
    5. Adjust WPN
    6. Record history ← NEVER REACHED
```

Fixed flow:
```
for cycle in range(max_iter):
    1. Generate combinations
    2. Find winner
    3. Record history (ALWAYS, before any return) ← FIX HERE
    4. Check convergence → RETURN if converged
    5. Update weights (if not converged)
    6. Adjust WPN (if not converged)
```

**Changes needed:**
- Move lines 578-580 (history recording) to occur BEFORE line 537 (convergence check)
- OR: Add history recording inside convergence block before return
- Ensure history is recorded for EVERY cycle that executes, regardless of outcome

**Lines to modify:** 537-580 in `combinatorial.py`

---

### Phase 2: Verify Fix with Tests

**Task 2.1: Create test to verify history recording**
- Test case with immediate convergence (1 cycle)
- Test case with multiple cycles needed (5+ cycles)
- Test case hitting max_iter without convergence
- Verify all history keys populated correctly

**Task 2.2: Re-run benchmark comparison**
- After fix, WCS should show actual iteration counts
- Comparison with BRO/AdamW will be more accurate
- Update OPTIMIZER_COMPARISON.md with corrected results

---

### Phase 3: Additional Improvements (Optional)

**Task 3.1: Add history validation**
- Assert that history is recorded for every completed cycle
- Add sanity checks: len(cycles) == len(wpn_evolution) == len(best_delta_evolution)

**Task 3.2: Improve early stopping documentation**
- Document that early stopping occurs WITHIN cycle (intra-cycle)
- Document that convergence check occurs BETWEEN cycles (inter-cycle)
- Clarify that both should record history

---

## Implementation Order

1. **Fix history recording** (lines 537-580)
2. **Test the fix** (verify history populated)
3. **Re-run benchmark** (get accurate comparison)
4. **Update documentation** (correct the comparison table)
5. **Commit and push** (with clear bug fix message)

---

## Expected Results After Fix

### Before Fix:
```
Cycles executed: 0
WPN evolution: []
Best delta evolution: []
```

### After Fix:
```
Cycles executed: 1 (or more)
WPN evolution: [1.0, ...]
Best delta evolution: [0.0, ...]
```

### Impact on Benchmark:
- WCS will show **real iteration counts** (1-10 cycles typically)
- Time measurements will be **more accurate**
- Comparison with BRO/AdamW will be **valid** (comparing actual work done)

---

## Questions to Address

**Q1: Should history be recorded for cycle 0 if it converges immediately?**
- **A:** YES - if a cycle executed (even partially), it should be in history

**Q2: Should early-stopped cycles (intra-cycle) show full iteration count?**
- **A:** YES - the cycle number should be recorded, along with early_stop info

**Q3: What if winner is None (edge case)?**
- **A:** Check for this before recording history, handle gracefully

---

## Testing Checklist

- [ ] Fix history recording bug
- [ ] Test: Immediate convergence (1 cycle)
- [ ] Test: Multiple cycles (5+ cycles)
- [ ] Test: Max iterations without convergence
- [ ] Test: With index filtering enabled
- [ ] Test: With adaptive sampling
- [ ] Verify: All history keys have same length
- [ ] Re-run: Optimizer comparison benchmark
- [ ] Update: OPTIMIZER_COMPARISON.md
- [ ] Commit and push

---

## Protocol Compliance

✅ Following SIMPLICITY_PROTOCOL_3.md:
- Identified root cause before making changes
- Created action plan with clear phases
- Will test each change before proceeding
- Will update documentation after verification
- Will commit with descriptive message

✅ Following previous instructions:
- Fixing algorithm behavior
- Ensuring proper history tracking
- Making benchmark comparison valid
- Will test with consumer after fix

---

## Next Step

**Proceed with Phase 1: Fix history recording bug**
- Modify lines 537-580 in `combinatorial.py`
- Ensure history recorded before any return statement
- Test the fix immediately

Awaiting confirmation to proceed...
