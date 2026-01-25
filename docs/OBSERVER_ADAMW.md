# ObserverAdamW: Parallel Hyperparameter Tuning with Observer Pattern

**Date:** 2026-01-25  
**Architecture:** Multiprocessing Observer Pattern  
**Status:** ✅ Implemented and Tested

---

## Overview

**ObserverAdamW** is an AdamW optimizer with **parallel hyperparameter tuning** using the **Observer Pattern** and **multiprocessing**. Each hyperparameter runs in its own process, proposing values via binary search. An Observer coordinates all processes, tests combinations, and signals optimal values.

### Key Features

✅ **Parallel Processing** - Each hyperparameter tuned in separate process  
✅ **Observer Pattern** - Central coordinator manages all processes  
✅ **Binary Search** - Each process narrows range around optimal values  
✅ **Truth Table** - Records all tested combinations and results  
✅ **Flexible Output** - Multiple output options (console, CSV, attribute)  
✅ **Backward Compatible** - Works like standard AdamW when disabled

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     OBSERVER (Main Process)                  │
│  • Tests all hyperparameter combinations                    │
│  • Ranks by cost                                            │
│  • Signals best values to each process                      │
│  • Manages shared context and convergence                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
     ┌─────────────┼─────────────┬───────────────┐
     │             │             │               │
     ▼             ▼             ▼               ▼
┌─────────┐  ┌─────────┐  ┌──────────┐  ┌────────────┐
│ Process │  │ Process │  │ Process  │  │  Process   │
│  Beta1  │  │  Beta2  │  │ Epsilon  │  │ Weight     │
│         │  │         │  │          │  │ Decay      │
│ Propose │  │ Propose │  │ Propose  │  │ Propose    │
│ 3 values│  │ 3 values│  │ 3 values │  │ 3 values   │
│ [L,M,H] │  │ [L,M,H] │  │ [L,M,H]  │  │ [L,M,H]    │
└─────────┘  └─────────┘  └──────────┘  └────────────┘
     │             │             │               │
     └─────────────┴─────────────┴───────────────┘
                   │
                   ▼
         ┌──────────────────────┐
         │  Shared Context      │
         │  (Manager.dict())    │
         │  • Proposals         │
         │  • Results           │
         │  • Status flags      │
         └──────────────────────┘
```

---

## How It Works

### 1. Process Creation

Each enabled hyperparameter gets its own `Process`:

```python
optimizer = ObserverAdamW(
    observer_tuning=True,
    auto_tune_beta1=True,      # Process 1
    auto_tune_beta2=True,      # Process 2
    auto_tune_epsilon=True,    # Process 3
    auto_tune_weight_decay=True # Process 4
)
```

### 2. Proposal Phase

Each process proposes 3 values using binary search:

```
Iteration 1:
  Process 1 (beta1):        [0.800, 0.895, 0.990]
  Process 2 (beta2):        [0.900, 0.9499, 0.9999]
  Process 3 (epsilon):      [1e-10, 5e-8, 1e-6]
  Process 4 (weight_decay): [0.0, 0.05, 0.1]
```

### 3. Combination Testing

Observer generates **all combinations** (3^N where N = number of hyperparameters):

```
4 hyperparameters = 3^4 = 81 combinations to test
```

Each combination runs **full optimization** (e.g., 100 iterations of AdamW):

```python
# For each combination
temp_optimizer = AdamW(
    beta1=proposed_beta1,
    beta2=proposed_beta2,
    epsilon=proposed_epsilon,
    weight_decay=proposed_weight_decay,
    max_iter=100
)
theta = temp_optimizer.optimize(X, y, theta_init, cost_fn, grad_fn)
final_cost = cost_fn(theta, X, y)
```

### 4. Observer Ranking

Observer finds best combination by cost:

```
Best combination (iteration 1):
  beta1 = 0.895
  beta2 = 0.9499
  epsilon = 5e-8
  weight_decay = 0.05
  cost = 0.115682
```

### 5. Feedback & Range Narrowing

Observer signals each process with best value:

```
Process 1 receives: beta1 = 0.895
  → Narrow range to [0.85, 0.94] (±25% margin)

Process 2 receives: beta2 = 0.9499
  → Narrow range to [0.94, 0.96]

... etc
```

### 6. Iteration & Convergence

Repeat until convergence:

- **Cost stable**: Best cost change < threshold for 2+ iterations
- **Ranges converged**: All processes have narrow ranges (< 1% of original)
- **Max iterations**: Reached maximum observer iterations

---

## Usage Examples

### Example 1: Basic Usage (Beta1 + Beta2)

```python
from binary_search.observer_tuning import ObserverAdamW
import numpy as np

# Data
X = np.random.randn(100, 10)
y = X @ np.ones(10) + 0.1 * np.random.randn(100)
theta_init = np.zeros(10)

# Cost and gradient
def cost_fn(theta, X, y):
    return 0.5 * np.mean((X @ theta - y) ** 2)

def grad_fn(theta, X, y):
    return X.T @ (X @ theta - y) / len(y)

# Create optimizer with observer tuning
optimizer = ObserverAdamW(
    observer_tuning=True,        # Enable parallel tuning
    auto_tune_beta1=True,        # Tune beta1
    auto_tune_beta2=True,        # Tune beta2
    max_iter=100,                # Iterations per combination test
    max_observer_iterations=5,   # Observer search iterations
    print_truth_table=True,      # Print results
    verbose=True
)

# Optimize
theta = optimizer.optimize(X, y, theta_init, cost_fn, grad_fn)

# Access results
print(f"Tuned beta1: {optimizer.beta1:.6f}")
print(f"Tuned beta2: {optimizer.beta2:.6f}")
print(f"Truth table entries: {len(optimizer.truth_table)}")
```

### Example 2: All Hyperparameters

```python
optimizer = ObserverAdamW(
    observer_tuning=True,
    auto_tune_beta1=True,
    auto_tune_beta2=True,
    auto_tune_epsilon=True,
    auto_tune_weight_decay=True,
    max_iter=50,
    max_observer_iterations=3,  # Fewer iterations (81 combos per iteration!)
    print_truth_table=True,
    save_truth_table_csv=True,
    truth_table_filename='hyperparameter_search.csv',
    verbose=True
)

theta = optimizer.optimize(X, y, theta_init, cost_fn, grad_fn)
```

### Example 3: Custom Ranges

```python
optimizer = ObserverAdamW(
    observer_tuning=True,
    auto_tune_beta1=True,
    auto_tune_beta2=True,
    # Narrow ranges around known-good values
    beta1_range=(0.85, 0.95),
    beta2_range=(0.95, 0.999),
    max_iter=100,
    max_observer_iterations=5,
    verbose=True
)
```

### Example 4: Truth Table Output Options

```python
optimizer = ObserverAdamW(
    observer_tuning=True,
    auto_tune_beta1=True,
    
    # Truth table output (checkboxes)
    print_truth_table=True,       # ☑ Print to console
    save_truth_table_csv=True,    # ☑ Save to CSV
    store_truth_table=True,       # ☑ Store as attribute
    truth_table_filename='results.csv',
    
    verbose=True
)

theta = optimizer.optimize(X, y, theta_init, cost_fn, grad_fn)

# Access truth table
for entry in optimizer.truth_table:
    print(entry)
```

### Example 5: Backward Compatibility (Standard AdamW)

```python
# Works like normal AdamW when observer_tuning=False
optimizer = ObserverAdamW(
    observer_tuning=False,  # Disabled
    max_iter=100,
    verbose=True
)

theta = optimizer.optimize(X, y, theta_init, cost_fn, grad_fn)
# No parallel tuning, just standard AdamW
```

---

## Parameters

### Observer Tuning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observer_tuning` | bool | False | Enable observer-based parallel tuning |
| `auto_tune_beta1` | bool | False | Enable parallel tuning for beta1 |
| `auto_tune_beta2` | bool | False | Enable parallel tuning for beta2 |
| `auto_tune_epsilon` | bool | False | Enable parallel tuning for epsilon |
| `auto_tune_weight_decay` | bool | False | Enable parallel tuning for weight_decay |
| `max_observer_iterations` | int | 5 | Maximum observer search iterations |
| `convergence_threshold` | float | 0.001 | Cost improvement threshold for convergence |
| `range_convergence_threshold` | float | 0.01 | Range width threshold (fraction of original) |

### Truth Table Output Options (Checkboxes)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `print_truth_table` | bool | False | Print truth table to console |
| `save_truth_table_csv` | bool | False | Save truth table to CSV file |
| `return_truth_table_df` | bool | False | Return as DataFrame (not implemented yet) |
| `store_truth_table` | bool | True | Store as `optimizer.truth_table` attribute |
| `truth_table_filename` | str | 'truth_table.csv' | Filename for CSV output |

### Hyperparameter Ranges

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta1_range` | tuple | (0.8, 0.99) | Search range for beta1 |
| `beta2_range` | tuple | (0.9, 0.9999) | Search range for beta2 |
| `epsilon_range` | tuple | (1e-10, 1e-6) | Search range for epsilon |
| `weight_decay_range` | tuple | (0.0, 0.1) | Search range for weight_decay |

### Inherited from AdamW

All standard AdamW parameters: `max_iter`, `beta1`, `beta2`, `epsilon`, `weight_decay`, `base_lr`, `tol`, `verbose`, etc.

---

## Truth Table Format

The truth table records all tested combinations:

```python
[
    {
        'iteration': 1,
        'beta1': 0.800000,
        'beta2': 0.900000,
        'cost': 0.125432
    },
    {
        'iteration': 1,
        'beta1': 0.800000,
        'beta2': 0.9499,
        'cost': 0.118234
    },
    ...
]
```

Access via: `optimizer.truth_table`

---

## Performance Considerations

### Time Complexity

For N hyperparameters and M observer iterations:

```
Total combinations = 3^N × M
Total AdamW runs = 3^N × M

Example:
  2 hyperparameters, 5 iterations = 3^2 × 5 = 45 AdamW runs
  4 hyperparameters, 3 iterations = 3^4 × 3 = 243 AdamW runs
```

### Time Cost Analysis

| Hyperparameters | Iterations | Combinations | AdamW Runs | Est. Time* |
|----------------|------------|--------------|------------|------------|
| 1 (beta1)      | 5          | 15           | 15         | ~1 min     |
| 2 (beta1+beta2) | 5         | 45           | 45         | ~3 min     |
| 3              | 3          | 81           | 81         | ~5 min     |
| 4 (all)        | 3          | 243          | 243        | ~15 min    |

*Assuming 5 seconds per AdamW run (100 iterations)

### Recommendations

✅ **Use when:**
- Complex/non-convex problems
- No prior knowledge of good hyperparameters
- Willing to pay time cost for better results
- Tuning 1-2 hyperparameters

⚠️ **Use with caution:**
- 3+ hyperparameters (exponential growth)
- Reduce `max_observer_iterations` to 2-3
- Consider tuning one at a time

❌ **Don't use when:**
- Simple/convex problems (defaults work well)
- Time-critical applications
- Default hyperparameters already work

---

## Technical Details

### Multiprocessing Implementation

- **Process per hyperparameter**: Each runs `HyperparameterProcess.run()`
- **Shared context**: `multiprocessing.Manager().dict()` for communication
- **Feedback queues**: Each process has dedicated `Queue` for observer signals
- **Stop event**: `Event` for graceful shutdown

### Observer Pattern

Based on classic **Subject-Observer** pattern:

- **Subject**: Each hyperparameter process (proposes values)
- **Observer**: Main observer process (tests combinations, signals feedback)
- **Update mechanism**: Feedback queues with best value signals
- **One-to-many**: Observer coordinates N processes simultaneously

### Binary Search Strategy

Each process uses binary search to narrow range:

1. Initial: [low, high]
2. Propose: [low, mid, high]
3. Observer tests all combinations
4. Feedback: best_value
5. Narrow: [best - 25%, best + 25%]
6. Repeat

### Convergence Criteria

Optimization stops when ANY of these conditions met:

1. **Cost stable**: `abs(prev_cost - best_cost) < convergence_threshold` for 2+ iterations
2. **Ranges converged**: All processes have `(high - low) < range_convergence_threshold × initial_range`
3. **Max iterations**: Reached `max_observer_iterations`

---

## Comparison with Sequential Tuning

| Feature | Sequential (AdamW) | Parallel (ObserverAdamW) |
|---------|-------------------|-------------------------|
| **Hyperparameters tuned** | One at a time | All simultaneously |
| **Interactions detected** | ❌ No | ✅ Yes |
| **Time complexity** | O(N) | O(3^N) |
| **Parallelism** | None | Multiprocessing |
| **Truth table** | ❌ No | ✅ Yes |
| **When to use** | Quick tuning | Thorough search |

---

## Future Enhancements

1. **Adaptive test length** - Increase test iterations if candidates are close
2. **Validation-based selection** - Use validation set instead of training cost
3. **Smart combination sampling** - Don't test all 3^N, use sampling
4. **GPU acceleration** - Parallelize AdamW runs across GPUs
5. **Meta-learning** - Learn which hyperparameters to tune based on problem type

---

## References

- **Observer Pattern**: [Wikipedia](https://en.wikipedia.org/wiki/Observer_pattern)
- **Python Multiprocessing**: [Python Docs](https://docs.python.org/3/library/multiprocessing.html)
- **AdamW**: Loshchilov & Hutter (2019) "Decoupled Weight Decay Regularization"

---

## Status

✅ **Implemented**: Core functionality complete  
✅ **Tested**: Basic tests passing  
✅ **Documented**: Full documentation  
⚠️ **Production**: Use with caution (computationally expensive)

**Version**: 1.0.0  
**Date**: 2026-01-25
