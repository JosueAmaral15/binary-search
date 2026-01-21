"""
Example: Linear Regression with Binary Rate Optimizer

This example demonstrates using the Binary Rate Optimizer to solve
a simple linear regression problem without manually tuning the learning rate.

Problem: Find the best-fit line for data points
Dataset: y = 2x + noise
"""

import numpy as np
from binary_rate_optimizer import BinaryRateOptimizer


def main():
    print("=" * 60)
    print("Example: Linear Regression with Binary Rate Optimizer")
    print("=" * 60)
    
    # Generate synthetic data: y = 2x + noise
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y_true = 2 * X
    noise = np.random.normal(0, 0.5, size=len(X))
    y = y_true + noise
    
    print(f"\nDataset: {len(X)} points")
    print(f"True relationship: y = 2x")
    print(f"Sample points: X={X[:3]}, y={y[:3]}")
    
    # Define cost function (Mean Squared Error)
    def mse_cost(theta, X, y):
        """Mean Squared Error"""
        predictions = X * theta
        return np.mean((predictions - y) ** 2)
    
    # Define gradient function
    def mse_gradient(theta, X, y):
        """Gradient of MSE with respect to theta"""
        predictions = X * theta
        error = predictions - y
        return 2 * np.mean(error * X)
    
    # Initialize optimizer (no learning rate needed!)
    optimizer = BinaryRateOptimizer(
        max_iter=50,
        tol=1e-9,
        expansion_factor=2.0,
        binary_search_steps=10
    )
    
    # Starting guess (intentionally bad)
    initial_theta = np.array([0.5])
    print(f"\nInitial guess: theta = {initial_theta[0]:.4f}")
    print(f"Initial cost: {mse_cost(initial_theta, X, y):.6f}")
    
    # Optimize!
    print("\n" + "=" * 60)
    print("Starting Optimization...")
    print("=" * 60)
    
    final_theta = optimizer.optimize(
        X=X,
        y=y,
        initial_theta=initial_theta,
        cost_func=mse_cost,
        grad_func=mse_gradient,
        verbose=True
    )
    
    # Results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"True theta:  {2.0:.6f}")
    print(f"Found theta: {final_theta[0]:.6f}")
    print(f"Error:       {abs(final_theta[0] - 2.0):.6f}")
    print(f"Final cost:  {optimizer.history['cost'][-1]:.6f}")
    print(f"Iterations:  {len(optimizer.history['cost']) - 1}")
    
    # Show how learning rate adapted
    print("\n" + "=" * 60)
    print("Learning Rate Adaptation (last 5 iterations)")
    print("=" * 60)
    for i, alpha in enumerate(optimizer.history['alpha'][-5:], 1):
        print(f"Iteration {len(optimizer.history['alpha']) - 5 + i}: alpha = {alpha:.6f}")
    
    # Comparison with predictions
    print("\n" + "=" * 60)
    print("Prediction Quality")
    print("=" * 60)
    predictions = X * final_theta[0]
    for i in range(min(5, len(X))):
        print(f"X={X[i]:.1f}: True={y[i]:.2f}, Predicted={predictions[i]:.2f}, Error={abs(y[i] - predictions[i]):.2f}")
    
    print("\n✅ Optimization complete! No manual learning rate tuning required.")


if __name__ == "__main__":
    main()
