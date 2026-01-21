# Binary Rate Optimizer

**Gradient Descent with Dynamic Learning Rate using Binary Search**

A Python optimizer that automatically finds the optimal learning rate at each iteration, eliminating the need for manual tuning.

## Features

- 🚀 **Zero Tuning Required**: No need to manually set learning rates
- 🎯 **Adaptive**: Adjusts to local cost function topology automatically
- 📊 **History Tracking**: Records theta, cost, and alpha at each iteration
- ✅ **Production Ready**: Type hints, input validation, comprehensive tests
- 🧪 **Well Tested**: 90%+ code coverage with edge case handling

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from binary_rate_optimizer import BinaryRateOptimizer

# Define your cost and gradient functions
def mse_cost(theta, X, y):
    """Mean Squared Error"""
    predictions = X * theta
    return np.mean((predictions - y) ** 2)

def mse_gradient(theta, X, y):
    """Gradient of MSE"""
    predictions = X * theta
    error = predictions - y
    return 2 * np.mean(error * X)

# Prepare data: y = 2x
X = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)

# Initialize optimizer
optimizer = BinaryRateOptimizer(
    max_iter=20,
    tol=1e-9,
    expansion_factor=2.0,
    binary_search_steps=10
)

# Optimize!
initial_theta = np.array([0.0])
final_theta = optimizer.optimize(
    X=X,
    y=y,
    initial_theta=initial_theta,
    cost_func=mse_cost,
    grad_func=mse_gradient
)

print(f"Optimal theta: {final_theta[0]:.6f}")  # Should be ~2.0
```

## How It Works

The Binary Rate Optimizer uses a two-phase strategy:

### Phase 1: Expansion
- Starts with a small learning rate (alpha = 1e-4)
- Exponentially increases alpha until cost stops decreasing
- Finds bounds [alpha_low, alpha_high] containing optimal alpha

### Phase 2: Binary Search
- Refines the search within the found bounds
- Uses binary subdivision for precision
- Performs local slope tests to guide the search

This eliminates the need for:
- Manual learning rate tuning
- Learning rate schedules
- Decay strategies
- Grid search over hyperparameters

## API Reference

### BinaryRateOptimizer

```python
BinaryRateOptimizer(
    max_iter: int = 100,
    tol: float = 1e-6,
    expansion_factor: float = 2.0,
    binary_search_steps: int = 10
)
```

**Parameters:**
- `max_iter`: Maximum gradient descent iterations
- `tol`: Convergence tolerance (stop if |cost_change| < tol)
- `expansion_factor`: Multiplier for alpha during expansion (default: 2.0)
- `binary_search_steps`: Binary subdivisions for refinement (default: 10)

### optimize()

```python
optimize(
    X: np.ndarray,
    y: np.ndarray,
    initial_theta: np.ndarray,
    cost_func: Callable,
    grad_func: Callable,
    verbose: bool = True
) -> np.ndarray
```

**Parameters:**
- `X`: Input features
- `y`: Target values
- `initial_theta`: Starting parameters
- `cost_func`: Function with signature `(theta, X, y) -> float`
- `grad_func`: Function with signature `(theta, X, y) -> np.ndarray`
- `verbose`: Print progress (default: True)

**Returns:**
- Optimized parameters (theta)

## Examples

See the `examples/` directory for:
- Linear regression
- Polynomial fitting
- Logistic regression
- Custom cost functions

## When to Use

**✅ Good for:**
- Problems where learning rate is hard to tune
- Non-stationary optimization landscapes
- When you want adaptive behavior without complexity

**❌ Not ideal for:**
- Stochastic gradient descent (use Adam/AdamW instead)
- Very large-scale problems (binary search adds overhead)
- Cases where a fixed learning rate already works well

## Comparison with Other Optimizers

| Optimizer | Learning Rate | Pros | Cons |
|-----------|---------------|------|------|
| **BR-GD** | Auto (binary search) | No tuning, adaptive | Extra cost evaluations |
| **SGD** | Manual | Simple, fast | Requires tuning |
| **Adam** | Auto (momentum) | Great for deep learning | Not deterministic |
| **Line Search** | Auto (analytical) | Optimal per step | Expensive for ML |

## Performance

- **Overhead**: ~20 extra cost evaluations per iteration
- **Convergence**: Usually faster than fixed LR in wall-clock time
- **Use Case**: Best for batch optimization on small-medium datasets

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=binary_rate_optimizer tests/
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run type checking
mypy binary_rate_optimizer/

# Format code
black binary_rate_optimizer/
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this optimizer in research, please cite:

```bibtex
@software{binary_rate_optimizer,
  author = {Josue},
  title = {Binary Rate Optimizer: Gradient Descent with Dynamic Learning Rate},
  year = {2026},
  url = {https://github.com/yourusername/binary-rate-optimizer}
}
```

## Changelog

### 1.0.0 (2026-01-21)
- Initial release
- Full refactor from combined package
- Comprehensive documentation and tests
- Type hints and input validation
- CI/CD with GitHub Actions

## See Also

- [Binary Search Algorithms](../binary_search_algorithms/) - Companion package for search algorithms
- [NumPy](https://numpy.org/) - Numerical computing library
- [SciPy Optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) - Alternative optimization methods
