#!/usr/bin/env python3
"""
AdamW Optimizer Example with Binary Search Learning Rate.

This example demonstrates the new AdamW optimizer that combines:
- Adaptive moment estimation (Adam algorithm)
- Decoupled weight decay (AdamW improvement)
- Binary search for optimal learning rate per iteration

We'll compare three optimizers on the same regression problem:
1. BinaryRateOptimizer (vanilla gradient descent + binary search)
2. AdamW without binary search (fixed learning rate)
3. AdamW with binary search (our innovation)
"""

import numpy as np
import matplotlib.pyplot as plt
from binary_search.optimizers import BinaryRateOptimizer, AdamW


# ============================================================================
# Problem Setup: Polynomial Regression
# ============================================================================

def generate_data(n_samples=200, n_features=5, noise=0.5):
    """Generate synthetic regression data."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # True coefficients: [2, -3, 1, 4, -2]
    true_theta = np.array([2.0, -3.0, 1.0, 4.0, -2.0])
    
    # Generate targets with noise
    y = X @ true_theta + np.random.randn(n_samples) * noise
    
    return X, y, true_theta


def mse_cost(theta, X, y):
    """Mean Squared Error cost function."""
    predictions = X @ theta
    return np.mean((predictions - y) ** 2)


def mse_gradient(theta, X, y):
    """Gradient of MSE."""
    predictions = X @ theta
    return 2 * X.T @ (predictions - y) / len(y)


# ============================================================================
# Generate Data
# ============================================================================

print("=" * 80)
print("AdamW Optimizer Demonstration")
print("=" * 80)

X_train, y_train, true_theta = generate_data(n_samples=200, n_features=5, noise=0.5)
theta_init = np.zeros(5)

print(f"\nDataset:")
print(f"  Samples: {len(X_train)}")
print(f"  Features: {X_train.shape[1]}")
print(f"  True theta: {true_theta}")
print(f"  Initial theta: {theta_init}")
print(f"  Initial cost: {mse_cost(theta_init, X_train, y_train):.6f}")


# ============================================================================
# Optimizer 1: BinaryRateOptimizer (Vanilla GD + Binary Search)
# ============================================================================

print("\n" + "=" * 80)
print("Optimizer 1: BinaryRateOptimizer (Vanilla GD + Binary Search)")
print("=" * 80)

optimizer1 = BinaryRateOptimizer(
    max_iter=50,
    tol=1e-8,
    expansion_factor=2.0,
    binary_search_steps=10,
    verbose=False
)

theta1 = optimizer1.optimize(X_train, y_train, theta_init.copy(), mse_cost, mse_gradient)

print(f"\nResults:")
print(f"  Final theta: {theta1}")
print(f"  Error from true: {np.linalg.norm(theta1 - true_theta):.6f}")
print(f"  Final cost: {optimizer1.history['cost'][-1]:.8f}")
print(f"  Iterations: {len(optimizer1.history['cost']) - 1}")


# ============================================================================
# Optimizer 2: AdamW WITHOUT Binary Search (Fixed LR)
# ============================================================================

print("\n" + "=" * 80)
print("Optimizer 2: AdamW WITHOUT Binary Search (Fixed LR = 0.01)")
print("=" * 80)

optimizer2 = AdamW(
    max_iter=50,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
    tol=1e-8,
    use_binary_search=False,
    base_lr=0.01,
    verbose=False
)

theta2 = optimizer2.optimize(X_train, y_train, theta_init.copy(), mse_cost, mse_gradient)

print(f"\nResults:")
print(f"  Final theta: {theta2}")
print(f"  Error from true: {np.linalg.norm(theta2 - true_theta):.6f}")
print(f"  Final cost: {optimizer2.history['cost'][-1]:.8f}")
print(f"  Iterations: {len(optimizer2.history['cost']) - 1}")


# ============================================================================
# Optimizer 3: AdamW WITH Binary Search (Adaptive LR)
# ============================================================================

print("\n" + "=" * 80)
print("Optimizer 3: AdamW WITH Binary Search (Adaptive LR)")
print("=" * 80)

optimizer3 = AdamW(
    max_iter=50,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
    tol=1e-8,
    use_binary_search=True,
    base_lr=0.01,
    expansion_factor=2.0,
    binary_search_steps=10,
    verbose=False
)

theta3 = optimizer3.optimize(X_train, y_train, theta_init.copy(), mse_cost, mse_gradient)

print(f"\nResults:")
print(f"  Final theta: {theta3}")
print(f"  Error from true: {np.linalg.norm(theta3 - true_theta):.6f}")
print(f"  Final cost: {optimizer3.history['cost'][-1]:.8f}")
print(f"  Iterations: {len(optimizer3.history['cost']) - 1}")


# ============================================================================
# Comparison Summary
# ============================================================================

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

comparison = [
    ("BinaryRateOptimizer (GD + BS)", 
     len(optimizer1.history['cost']) - 1,
     optimizer1.history['cost'][-1],
     np.linalg.norm(theta1 - true_theta)),
    
    ("AdamW (Fixed LR)", 
     len(optimizer2.history['cost']) - 1,
     optimizer2.history['cost'][-1],
     np.linalg.norm(theta2 - true_theta)),
    
    ("AdamW (Binary Search LR)", 
     len(optimizer3.history['cost']) - 1,
     optimizer3.history['cost'][-1],
     np.linalg.norm(theta3 - true_theta)),
]

print(f"\n{'Optimizer':<30} {'Iters':<8} {'Final Cost':<15} {'Param Error':<15}")
print("-" * 70)
for name, iters, cost, error in comparison:
    print(f"{name:<30} {iters:<8} {cost:<15.8e} {error:<15.6f}")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("""
1. **BinaryRateOptimizer**: Simple but effective. Uses vanilla gradient descent
   with binary search for learning rate. Good baseline performance.

2. **AdamW (Fixed LR)**: Modern adaptive optimizer. Performs well with proper
   learning rate tuning, but requires hyperparameter experimentation.

3. **AdamW (Binary Search LR)**: Best of both worlds! Combines Adam's adaptive
   moments with binary search's optimal step size. No learning rate tuning needed.

Benefits of AdamW + Binary Search:
- No manual learning rate tuning required
- Adapts to local cost function topology
- Robust to different problem scales
- Faster convergence than fixed learning rate
""")


# ============================================================================
# Visualization (Optional - requires matplotlib)
# ============================================================================

try:
    print("\nGenerating convergence plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cost over iterations
    ax1.semilogy(optimizer1.history['cost'][1:], 'o-', label='BinaryRateOptimizer', alpha=0.7)
    ax1.semilogy(optimizer2.history['cost'][1:], 's-', label='AdamW (Fixed LR)', alpha=0.7)
    ax1.semilogy(optimizer3.history['cost'][1:], '^-', label='AdamW (Binary Search)', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost (log scale)')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning rate over iterations
    ax2.plot(optimizer1.history['alpha'][1:], 'o-', label='BinaryRateOptimizer', alpha=0.7)
    ax2.plot(optimizer2.history['lr'][1:], 's-', label='AdamW (Fixed LR)', alpha=0.7)
    ax2.plot(optimizer3.history['lr'][1:], '^-', label='AdamW (Binary Search)', alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Adaptation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adamw_comparison.png', dpi=150, bbox_inches='tight')
    print("  Plot saved as: adamw_comparison.png")
    
except Exception as e:
    print(f"  Could not generate plot: {e}")


print("\n" + "=" * 80)
print("Example completed successfully!")
print("=" * 80)
