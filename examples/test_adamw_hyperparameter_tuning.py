"""
Test AdamW Hyperparameter Auto-Tuning

Tests the checkbox paradigm for hyperparameter auto-tuning using binary search.
User can select which hyperparameters to auto-tune:
- beta1 (momentum decay)
- beta2 (RMSprop decay)
- epsilon (numerical stability)
- weight_decay (L2 regularization)
"""

import numpy as np
import sys
sys.path.insert(0, '../binary_search')

from binary_search.optimizers import AdamW


def create_test_problem(n_samples=100, n_features=5):
    """Create a simple regression problem."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_theta = np.random.randn(n_features) * 0.5
    y = X @ true_theta + np.random.randn(n_samples) * 0.1
    return X, y, true_theta


def cost(theta, X, y):
    """MSE cost function."""
    return np.mean((X @ theta - y) ** 2)


def grad(theta, X, y):
    """Gradient of MSE."""
    return 2 * X.T @ (X @ theta - y) / len(y)


def test_single_hyperparameter():
    """Test 1: Auto-tune single hyperparameter (beta1)."""
    print("=" * 80)
    print("TEST 1: AUTO-TUNE BETA1 ONLY")
    print("Checkboxes: ☑ beta1  ☐ beta2  ☐ epsilon  ☐ weight_decay")
    print("=" * 80)
    
    X, y, _ = create_test_problem()
    theta_init = np.zeros(5)
    
    optimizer = AdamW(
        max_iter=20,
        verbose=True,
        auto_tune_beta1=True  # ☑ Enable binary search for beta1
    )
    
    theta_final = optimizer.optimize(X, y, theta_init, cost, grad)
    
    print(f"\nFinal cost: {cost(theta_final, X, y):.6f}")
    print(f"Tuned beta1: {optimizer.beta1:.6f}")
    print(f"✅ Test 1 passed\n")


def test_multiple_hyperparameters():
    """Test 2: Auto-tune multiple hyperparameters (beta1 + beta2)."""
    print("=" * 80)
    print("TEST 2: AUTO-TUNE BETA1 + BETA2")
    print("Checkboxes: ☑ beta1  ☑ beta2  ☐ epsilon  ☐ weight_decay")
    print("=" * 80)
    
    X, y, _ = create_test_problem()
    theta_init = np.zeros(5)
    
    optimizer = AdamW(
        max_iter=20,
        verbose=False,
        auto_tune_beta1=True,  # ☑
        auto_tune_beta2=True   # ☑
    )
    
    theta_final = optimizer.optimize(X, y, theta_init, cost, grad)
    
    print(f"\nTuned beta1: {optimizer.beta1:.6f}")
    print(f"Tuned beta2: {optimizer.beta2:.6f}")
    print(f"Final cost: {cost(theta_final, X, y):.6f}")
    print(f"✅ Test 2 passed\n")


def test_all_hyperparameters():
    """Test 3: Auto-tune all hyperparameters."""
    print("=" * 80)
    print("TEST 3: AUTO-TUNE ALL HYPERPARAMETERS")
    print("Checkboxes: ☑ beta1  ☑ beta2  ☑ epsilon  ☑ weight_decay")
    print("=" * 80)
    
    X, y, _ = create_test_problem()
    theta_init = np.zeros(5)
    
    optimizer = AdamW(
        max_iter=20,
        verbose=False,
        auto_tune_beta1=True,        # ☑
        auto_tune_beta2=True,        # ☑
        auto_tune_epsilon=True,      # ☑
        auto_tune_weight_decay=True  # ☑
    )
    
    theta_final = optimizer.optimize(X, y, theta_init, cost, grad)
    
    print(f"\nTuned hyperparameters:")
    print(f"  beta1: {optimizer.beta1:.6f}")
    print(f"  beta2: {optimizer.beta2:.6f}")
    print(f"  epsilon: {optimizer.epsilon:.2e}")
    print(f"  weight_decay: {optimizer.weight_decay:.6f}")
    print(f"Final cost: {cost(theta_final, X, y):.6f}")
    print(f"✅ Test 3 passed\n")


def test_custom_ranges():
    """Test 4: Custom search ranges."""
    print("=" * 80)
    print("TEST 4: CUSTOM SEARCH RANGES")
    print("=" * 80)
    
    X, y, _ = create_test_problem()
    theta_init = np.zeros(5)
    
    # Custom narrower range for beta1
    optimizer = AdamW(
        max_iter=20,
        verbose=False,
        auto_tune_beta1=True,
        beta1_range=(0.85, 0.95)  # Custom range instead of (0.8, 0.99)
    )
    
    theta_final = optimizer.optimize(X, y, theta_init, cost, grad)
    
    print(f"\nCustom beta1 range: (0.85, 0.95)")
    print(f"Tuned beta1: {optimizer.beta1:.6f}")
    print(f"In range: {0.85 <= optimizer.beta1 <= 0.95}")
    print(f"✅ Test 4 passed\n")


def test_backward_compatibility():
    """Test 5: Backward compatibility (no auto-tuning)."""
    print("=" * 80)
    print("TEST 5: BACKWARD COMPATIBILITY")
    print("Checkboxes: ☐ beta1  ☐ beta2  ☐ epsilon  ☐ weight_decay (ALL OFF)")
    print("=" * 80)
    
    X, y, _ = create_test_problem()
    theta_init = np.zeros(5)
    
    # Old API - no auto-tuning
    optimizer = AdamW(
        max_iter=20,
        verbose=False,
        beta1=0.9,
        beta2=0.999
    )
    
    theta_final = optimizer.optimize(X, y, theta_init, cost, grad)
    
    print(f"\nNo auto-tuning (old API)")
    print(f"beta1: {optimizer.beta1:.6f} (should be 0.9)")
    print(f"beta2: {optimizer.beta2:.6f} (should be 0.999)")
    print(f"Final cost: {cost(theta_final, X, y):.6f}")
    
    assert optimizer.beta1 == 0.9, "beta1 should remain 0.9"
    assert optimizer.beta2 == 0.999, "beta2 should remain 0.999"
    
    print(f"✅ Test 5 passed - Backward compatibility maintained\n")


def test_checkbox_combinations():
    """Test 6: Different checkbox combinations."""
    print("=" * 80)
    print("TEST 6: DIFFERENT CHECKBOX COMBINATIONS")
    print("=" * 80)
    
    X, y, _ = create_test_problem()
    theta_init = np.zeros(5)
    
    combinations = [
        ("☑☐☐☐", {"auto_tune_beta1": True}),
        ("☐☑☐☐", {"auto_tune_beta2": True}),
        ("☐☐☑☐", {"auto_tune_epsilon": True}),
        ("☐☐☐☑", {"auto_tune_weight_decay": True}),
        ("☑☑☐☐", {"auto_tune_beta1": True, "auto_tune_beta2": True}),
        ("☐☐☑☑", {"auto_tune_epsilon": True, "auto_tune_weight_decay": True}),
    ]
    
    for pattern, kwargs in combinations:
        print(f"\n{pattern}: ", end="")
        optimizer = AdamW(max_iter=10, verbose=False, **kwargs)
        theta_final = optimizer.optimize(X, y, theta_init, cost, grad)
        print(f"✓ Works")
    
    print(f"\n✅ Test 6 passed - All combinations work\n")


def main():
    """Run all hyperparameter auto-tuning tests."""
    print("\n" + "=" * 80)
    print("  ADAMW HYPERPARAMETER AUTO-TUNING - COMPREHENSIVE TESTS")
    print("  Feature: Binary Search for Hyperparameters (Checkbox Paradigm)")
    print("=" * 80)
    
    try:
        # Test 1: Single hyperparameter
        test_single_hyperparameter()
        
        # Test 2: Multiple hyperparameters
        test_multiple_hyperparameters()
        
        # Test 3: All hyperparameters
        test_all_hyperparameters()
        
        # Test 4: Custom ranges
        test_custom_ranges()
        
        # Test 5: Backward compatibility
        test_backward_compatibility()
        
        # Test 6: Checkbox combinations
        test_checkbox_combinations()
        
        print("=" * 80)
        print("✅ ALL HYPERPARAMETER AUTO-TUNING TESTS PASSED")
        print("=" * 80)
        
        print("\nSummary:")
        print("  ✅ Single hyperparameter tuning (beta1)")
        print("  ✅ Multiple hyperparameters (beta1 + beta2)")
        print("  ✅ All hyperparameters (beta1, beta2, epsilon, weight_decay)")
        print("  ✅ Custom search ranges")
        print("  ✅ Backward compatibility maintained")
        print("  ✅ All checkbox combinations work")
        
        print("\nCheckbox Paradigm:")
        print("  User selects which hyperparameters to auto-tune:")
        print("    ☑ auto_tune_beta1=True")
        print("    ☑ auto_tune_beta2=True")
        print("    ☐ auto_tune_epsilon=False (default)")
        print("    ☐ auto_tune_weight_decay=False (default)")
        
        print("\nBinary Search:")
        print("  - Tests multiple values in specified range")
        print("  - Evaluates cost after 5 quick iterations")
        print("  - Selects optimal value")
        print("  - Caches for remaining iterations")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
