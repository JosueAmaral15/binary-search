"""
Test suite for ObserverAdamW parallel hyperparameter tuning.

Tests the multiprocessing observer architecture with different configurations.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from binary_search.observer_tuning import ObserverAdamW

# Set random seed
np.random.seed(42)

print("=" * 80)
print("ObserverAdamW PARALLEL HYPERPARAMETER TUNING TESTS")
print("=" * 80)

# Generate test data
n_samples = 100
n_features = 10
X = np.random.randn(n_samples, n_features)
true_theta = np.random.randn(n_features)
y = X @ true_theta + 0.1 * np.random.randn(n_samples)

theta_init = np.zeros(n_features)

# Cost and gradient functions
def cost_fn(theta, X, y):
    residuals = X @ theta - y
    return 0.5 * np.mean(residuals ** 2)

def grad_fn(theta, X, y):
    residuals = X @ theta - y
    return X.T @ residuals / len(y)

print(f"\nTest Problem:")
print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")
print(f"  Initial cost: {cost_fn(theta_init, X, y):.6f}")
print(f"  Optimal cost: ~{cost_fn(true_theta, X, y):.6f}")


# Test 1: Basic observer tuning with beta1 and beta2
print("\n" + "=" * 80)
print("TEST 1: Observer Tuning - Beta1 and Beta2")
print("=" * 80)

optimizer1 = ObserverAdamW(
    observer_tuning=True,
    auto_tune_beta1=True,
    auto_tune_beta2=True,
    max_iter=50,
    max_observer_iterations=3,
    print_truth_table=True,
    save_truth_table_csv=True,
    truth_table_filename='test1_truth_table.csv',
    verbose=True
)

theta1 = optimizer1.optimize(X, y, theta_init.copy(), cost_fn, grad_fn)
final_cost1 = cost_fn(theta1, X, y)

print(f"\n✓ Test 1 Complete:")
print(f"  Final cost: {final_cost1:.8f}")
print(f"  Tuned beta1: {optimizer1.beta1:.6f}")
print(f"  Tuned beta2: {optimizer1.beta2:.6f}")
print(f"  Truth table entries: {len(optimizer1.truth_table)}")


# Test 2: Observer tuning with all hyperparameters
print("\n" + "=" * 80)
print("TEST 2: Observer Tuning - All Hyperparameters")
print("=" * 80)

optimizer2 = ObserverAdamW(
    observer_tuning=True,
    auto_tune_beta1=True,
    auto_tune_beta2=True,
    auto_tune_epsilon=True,
    auto_tune_weight_decay=True,
    max_iter=50,
    max_observer_iterations=2,  # Fewer iterations (3^4=81 combos per iteration)
    print_truth_table=True,
    save_truth_table_csv=True,
    truth_table_filename='test2_truth_table.csv',
    verbose=True
)

theta2 = optimizer2.optimize(X, y, theta_init.copy(), cost_fn, grad_fn)
final_cost2 = cost_fn(theta2, X, y)

print(f"\n✓ Test 2 Complete:")
print(f"  Final cost: {final_cost2:.8f}")
print(f"  Tuned beta1: {optimizer2.beta1:.6f}")
print(f"  Tuned beta2: {optimizer2.beta2:.6f}")
print(f"  Tuned epsilon: {optimizer2.epsilon:.2e}")
print(f"  Tuned weight_decay: {optimizer2.weight_decay:.6f}")
print(f"  Truth table entries: {len(optimizer2.truth_table)}")


# Test 3: Backward compatibility - observer_tuning=False (standard AdamW)
print("\n" + "=" * 80)
print("TEST 3: Backward Compatibility - Standard AdamW")
print("=" * 80)

optimizer3 = ObserverAdamW(
    observer_tuning=False,  # Disabled
    max_iter=50,
    verbose=True
)

theta3 = optimizer3.optimize(X, y, theta_init.copy(), cost_fn, grad_fn)
final_cost3 = cost_fn(theta3, X, y)

print(f"\n✓ Test 3 Complete:")
print(f"  Final cost: {final_cost3:.8f}")
print(f"  Beta1 (default): {optimizer3.beta1:.6f}")
print(f"  Beta2 (default): {optimizer3.beta2:.6f}")


# Test 4: Single hyperparameter (beta1 only)
print("\n" + "=" * 80)
print("TEST 4: Observer Tuning - Single Hyperparameter (Beta1)")
print("=" * 80)

optimizer4 = ObserverAdamW(
    observer_tuning=True,
    auto_tune_beta1=True,  # Only beta1
    max_iter=50,
    max_observer_iterations=3,
    print_truth_table=True,
    verbose=True
)

theta4 = optimizer4.optimize(X, y, theta_init.copy(), cost_fn, grad_fn)
final_cost4 = cost_fn(theta4, X, y)

print(f"\n✓ Test 4 Complete:")
print(f"  Final cost: {final_cost4:.8f}")
print(f"  Tuned beta1: {optimizer4.beta1:.6f}")
print(f"  Beta2 (default): {optimizer4.beta2:.6f}")
print(f"  Truth table entries: {len(optimizer4.truth_table)}")


# Test 5: Custom ranges
print("\n" + "=" * 80)
print("TEST 5: Observer Tuning - Custom Ranges")
print("=" * 80)

optimizer5 = ObserverAdamW(
    observer_tuning=True,
    auto_tune_beta1=True,
    auto_tune_beta2=True,
    beta1_range=(0.85, 0.95),  # Narrow range
    beta2_range=(0.95, 0.999),  # Narrow range
    max_iter=50,
    max_observer_iterations=3,
    print_truth_table=False,
    save_truth_table_csv=False,
    verbose=True
)

theta5 = optimizer5.optimize(X, y, theta_init.copy(), cost_fn, grad_fn)
final_cost5 = cost_fn(theta5, X, y)

print(f"\n✓ Test 5 Complete:")
print(f"  Final cost: {final_cost5:.8f}")
print(f"  Tuned beta1: {optimizer5.beta1:.6f}")
print(f"  Tuned beta2: {optimizer5.beta2:.6f}")


# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

print(f"\nTest 1 (Beta1+Beta2):        Cost = {final_cost1:.8f}")
print(f"Test 2 (All 4 params):       Cost = {final_cost2:.8f}")
print(f"Test 3 (Standard AdamW):     Cost = {final_cost3:.8f}")
print(f"Test 4 (Beta1 only):         Cost = {final_cost4:.8f}")
print(f"Test 5 (Custom ranges):      Cost = {final_cost5:.8f}")

print(f"\n✓ All tests completed successfully!")
print("=" * 80)
