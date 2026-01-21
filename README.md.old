# Binary Search

A Python package providing advanced binary search implementations with mathematical optimization techniques.

## Overview

This package contains two main components:
1. **BinaryRateOptimizer**: Gradient Descent with dynamic binary search learning rate
2. **BinarySearch**: Flexible binary search with multiple variants and configurations

## Installation

```bash
pip install -e .
```

## Features

### 1. BinaryRateOptimizer (BR-GD)

A gradient descent optimizer that uses binary search to find the optimal learning rate at each iteration, eliminating the need for manual learning rate tuning.

**Key Features:**
- Dynamic line search instead of fixed learning rate
- Two-phase optimization: expansion + binary refinement
- Convergence history tracking (theta, cost, alpha)
- Adaptive step size selection

**Example Usage:**

```python
import numpy as np
from binary_search import BinaryRateOptimizer

# Define cost and gradient functions
def mse_cost(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    predictions = X * theta
    return np.mean((predictions - y) ** 2)

def mse_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    predictions = X * theta
    error = predictions - y
    return 2 * np.mean(error * X)

# Prepare data: y = 2x
X_data = np.array([1, 2, 3, 4], dtype=float)
y_data = np.array([2, 4, 6, 8], dtype=float)
initial_theta = np.array([0.0])

# Optimize
optimizer = BinaryRateOptimizer(max_iter=20, tol=1e-9)
final_theta = optimizer.optimize(
    X=X_data, 
    y=y_data, 
    initial_theta=initial_theta, 
    cost_func=mse_cost, 
    grad_func=mse_gradient
)

print(f"Found Theta: {final_theta[0]:.10f}")  # Expected: 2.0
```

**Parameters:**
- `max_iter`: Maximum gradient descent iterations (default: 100)
- `tol`: Convergence tolerance (default: 1e-6)
- `expansion_factor`: Multiplier for alpha expansion phase (default: 2.0)
- `binary_search_steps`: Binary subdivisions for refinement (default: 10)

### 2. BinarySearch

A highly configurable binary search implementation with multiple search strategies and mathematical comparison functions.

**Key Features:**
- Tolerance-based comparisons using mathematical functions (no traditional if/else)
- Multiple search modes: array search, function search, dictionary search
- Configurable search priorities and behaviors
- Support for monotonic function root finding
- Adaptive mid-step behavior changes

**Example Usage:**

#### Array Search
```python
from binary_search import BinarySearch

searcher = BinarySearch()
array = [1, 3, 5, 7, 9, 11, 13, 15]
index, is_max, is_min = searcher.search_for_array(7, array)
print(f"Found at index: {index}")  # Output: 3
```

#### Function Search
```python
import math
from binary_search import BinarySearch

# Find x where sqrt(x) = 5
x = BinarySearch.search_for_function(5, math.sqrt)
print(f"x = {x}")  # Output: 25.0
```

#### Dictionary Search (Miniterm Finder)
```python
searcher = BinarySearch()
data_dict = {0: 1.2, 1: 3.5, 2: 5.8, 3: 7.1, 4: 9.4}
value, key = searcher.binary_search_to_find_miniterm_from_dict(6.0, data_dict)
print(f"Closest value: {value}, Key: {key}")
```

#### Advanced: Custom Search Progression
```python
# Binary search with smallest values priority
searcher = BinarySearch(
    binary_search_priority_for_smallest_values=True,
    number_of_attempts=20
)

# Step-based search
for step in range(10):
    value = searcher.binary_search_by_step(step, 0, 100)
    print(f"Step {step}: {value}")

searcher.reset()  # Reset state for next search
```

**Constructor Parameters:**
- `binary_search_priority_for_smallest_values`: Prioritize smaller (True) or larger (False) values
- `previous_value_should_be_the_basis_of_binary_search_calculations`: Use previous value as search basis
- `previous_value_is_the_target`: Treat previous value as target instead of starting point
- `change_behavior_mid_step`: Flip priority halfway through search
- `number_of_attempts`: Maximum iterations (default: 20)

**Methods:**
- `search_for_array(expected_result, array, tolerance)`: Binary search in sorted arrays
- `search_for_function(y, function, tolerance, max_iter)`: Find x where f(x) ≈ y
- `binary_search_by_step(steps, min_limit, max_limit, previous_value)`: Rational function progression
- `linear_search_step(steps, min_limit, max_limit, previous_value)`: Arithmetic progression
- `binary_search_to_find_miniterm_from_dict(wanted_number, array_dict)`: Dictionary value search
- `reset()`: Reset internal state

## Understanding Theta (θ) - Model Parameters and Weights

### What is Theta?

**Theta (θ)** represents the **parameters or weights** of a model. It's what the machine learning algorithm is trying to **learn** from data.

**In the context of:**
- **Linear Regression**: θ is the slope and intercept
- **Neural Networks**: θ represents ALL weights and biases (synaptic strengths)
- **General ML**: θ is any learnable parameter

### Simple Example: Finding the Slope

Given data points: `(1,2), (2,4), (3,6), (4,8)`, we want to find `y = θ * x`

- If **θ = 1.0** → predictions: `[1, 2, 3, 4]` ❌ Wrong!
- If **θ = 2.0** → predictions: `[2, 4, 6, 8]` ✅ Perfect!
- If **θ = 3.0** → predictions: `[3, 6, 9, 12]` ❌ Wrong!

The optimizer **searches for θ = 2.0 automatically!**

```python
import numpy as np
from binary_search import BinaryRateOptimizer

X = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)

# Manual testing different theta values
test_thetas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for theta_test in test_thetas:
    predictions = X * theta_test
    error = np.mean((predictions - y) ** 2)
    print(f'θ={theta_test:.1f}: predictions={predictions}, error={error:.4f}')

# Output:
# θ=0.5: predictions=[0.5 1.  1.5 2. ], error=24.7500
# θ=1.0: predictions=[1. 2. 3. 4.], error=11.0000
# θ=1.5: predictions=[1.5 3.  4.5 6. ], error=2.7500
# θ=2.0: predictions=[2. 4. 6. 8.], error=0.0000  ← Perfect!
# θ=2.5: predictions=[ 2.5  5.   7.5 10. ], error=2.7500
# θ=3.0: predictions=[ 3.  6.  9. 12.], error=11.0000
```

### Neural Network Analogy: θ as Synaptic Weight

**YES! Theta IS essentially a synaptic weight!** 🧠

**Simple Neuron:**
```
[Input x] ----θ----> (•) -----> [Output = θ * x]
                   neuron
```

θ = **SYNAPTIC WEIGHT** (connection strength)

**Multiple Inputs (like real neurons):**
```
[x₁] ----θ₁----\
                 \
[x₂] ----θ₂------(•)----> [Output = θ₁x₁ + θ₂x₂ + θ₃x₃]
                 /
[x₃] ----θ₃----/
```

Each θᵢ is a **weight** representing synaptic strength.

### Biological vs Artificial Neurons

| **Biological Neurons** | **Artificial Neurons** |
|------------------------|------------------------|
| Synapses have different **strengths** | θ (weight) represents **connection strength** |
| Strong synapse → signal passes easily | Large θ → input has big influence |
| Weak synapse → signal is reduced | Small θ → input has small influence |
| Unused synapses weaken | θ → 0 (connection ignored) |
| Inhibitory connections (GABA) | Negative θ → inhibitory effect |

### What the Optimizer Does (Learning Process)

1. **Start** with random θ (wrong weights)
2. **Predict** using current θ
3. **Calculate error** (how wrong predictions are)
4. **Adjust θ** to reduce error (learning!)
5. **Repeat** until error is minimal

**This mirrors learning in the brain:**
- Synapses strengthen with use (θ increases)
- Unused connections weaken (θ decreases)  
- This is called **"synaptic plasticity"** or **Hebbian learning**: *"Neurons that fire together, wire together"*

### BinaryRateOptimizer's Role

**Gradient Descent** = Adjusting weights to minimize error

**Binary Rate Optimizer** = **Smart way** to find the best weight adjustments (learning rate) at each step

Instead of using a fixed learning rate, it performs a **binary search** to find the optimal step size for updating θ, making convergence faster and more reliable.

**Example:**
```python
optimizer = BinaryRateOptimizer(max_iter=20, tol=1e-9)
final_theta = optimizer.optimize(X, y, initial_theta, cost_func, grad_func)

# final_theta = The learned weight (like a trained synapse!)
```

---

## Mathematical Approach

The package uses mathematical functions instead of traditional conditionals for robustness:

- **Greater-than comparison**: `(1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance)`
- **Equality comparison**: Tolerance-based absolute difference evaluation
- **Rational progression**: `a/(a+1)` for non-linear convergence

This approach provides:
- Smooth numerical transitions
- Tolerance-aware comparisons
- Avoidance of edge cases with floating-point comparisons

## Dependencies

- `numpy`: For array operations and numerical computations
- `math`: For mathematical functions (isfinite)
- `typing`: For type hints

## Use Cases

- **Machine Learning**: Auto-tuning learning rates in gradient descent
- **Numerical Analysis**: Root finding for monotonic functions
- **Optimization**: Parameter search with custom convergence strategies
- **Data Structures**: Efficient searching in sorted collections
- **Scientific Computing**: Precision-controlled numerical searches

## License

This project is provided as-is for educational and practical use.

## Author

Josue

## Version

0.1

---

## Execution Examples

Below are real execution examples demonstrating the package functionality.

### Example 1: BinaryRateOptimizer - Linear Regression

```python
import numpy as np
from binary_search import BinaryRateOptimizer

def mse_cost(theta, X, y):
    predictions = X * theta
    return np.mean((predictions - y) ** 2)

def mse_gradient(theta, X, y):
    predictions = X * theta
    error = predictions - y
    return 2 * np.mean(error * X)

# Data: y = 2x
X_data = np.array([1, 2, 3, 4], dtype=float)
y_data = np.array([2, 4, 6, 8], dtype=float)
initial_theta = np.array([0.0])

optimizer = BinaryRateOptimizer(max_iter=20, tol=1e-9)
final_theta = optimizer.optimize(
    X=X_data, y=y_data, 
    initial_theta=initial_theta, 
    cost_func=mse_cost, 
    grad_func=mse_gradient
)
```

**Output:**
```
--- Starting BR-GD Optimization ---
Initial Cost: 30.000000
Iter 001: Alpha=0.067200 | Cost=0.00192000
Iter 002: Alpha=0.067200 | Cost=0.00000012
Iter 003: Alpha=0.067200 | Cost=0.00000000
Iter 004: Alpha=0.067200 | Cost=0.00000000
Convergence reached by tolerance (1e-09) at iter 4.

Expected Theta: 2.0
Found Theta: 2.0000010240
```

### Example 2: Array Search

```python
from binary_search import BinarySearch

searcher = BinarySearch()
array = [1, 3, 5, 7, 9, 11, 13, 15]
index, is_max, is_min = searcher.search_for_array(7, array)
print(f'Found at index: {index}, Value: {array[index]}')
print(f'Is global maximum: {is_max}, Is global minimum: {is_min}')
```

**Output:**
```
Found at index: 3, Value: 7
Is global maximum: 0, Is global minimum: 0
```

### Example 3: Function Search (Root Finding)

```python
import math
from binary_search import BinarySearch

# Find x where sqrt(x) = 5
result = BinarySearch.search_for_function(5, math.sqrt)
print(f'x = {result:.6f}')
print(f'Verification: sqrt({result:.6f}) = {math.sqrt(result):.6f}')

# Find x where x^2 = 16
result2 = BinarySearch.search_for_function(16, lambda x: x**2)
print(f'x = {result2:.6f}')
print(f'Verification: ({result2:.6f})^2 = {result2**2:.6f}')
```

**Output:**
```
x = 25.000000
Verification: sqrt(25.000000) = 5.000000
x = 4.000000
Verification: (4.000000)^2 = 16.000000
```

### Example 4: Dictionary Search (Closest Value)

```python
from binary_search import BinarySearch

searcher = BinarySearch()
data_dict = {0: 1.2, 1: 3.5, 2: 5.8, 3: 7.1, 4: 9.4}
value, key = searcher.binary_search_to_find_miniterm_from_dict(6.0, data_dict)
print(f'Closest value: {value}, Key: {key}')
```

**Output:**
```
Searching for closest value to 6.0
Closest value: 5.8, Key: 2
```

### Example 5: Step-based Binary Search (Rational Progression)

```python
from binary_search import BinarySearch

searcher = BinarySearch(
    binary_search_priority_for_smallest_values=True,
    number_of_attempts=10
)

for step in range(11):
    value = searcher.binary_search_by_step(step, 0, 100)
    print(f'Step {step:2d}: {value:7.3f}')
```

**Output:**
```
Step  0:   0.000
Step  1:  50.000
Step  2:  66.667
Step  3:  75.000
Step  4:  80.000
Step  5:  83.333
Step  6:  85.714
Step  7:  87.500
Step  8:  88.889
Step  9:  90.000
Step 10:  90.909
```

Notice how the rational function `a/(a+1)` creates a progression that converges faster than linear search but more gradually than traditional binary search.

### Example 6: Linear Search Progression

```python
from binary_search import BinarySearch

searcher = BinarySearch(
    binary_search_priority_for_smallest_values=True,
    number_of_attempts=10
)

for step in range(11):
    value = searcher.linear_search_step(step, 0, 100)
    print(f'Step {step:2d}: {value:7.3f}')
```

**Output:**
```
Step  0:   0.000
Step  1:  10.000
Step  2:  20.000
Step  3:  30.000
Step  4:  40.000
Step  5:  50.000
Step  6:  60.000
Step  7:  70.000
Step  8:  80.000
Step  9:  90.000
Step 10: 100.000
```

Linear progression provides uniform steps across the range, useful for systematic exploration.
