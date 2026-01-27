"""
Examples of NonLinearGaussSeidel Solver

Demonstrates solving various nonlinear systems using binary-enhanced
Gauss-Seidel iteration.
"""

import numpy as np
from math_toolkit.linear_systems import NonLinearGaussSeidel


def example_1_himmelblau():
    """
    Himmelblau's function system (classic test problem).
    
    System:
        x² + y - 11 = 0
        x + y² - 7 = 0
    
    Known solutions: (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
    """
    print("=" * 70)
    print("Example 1: Himmelblau's Function System")
    print("=" * 70)
    print("System:")
    print("  x² + y - 11 = 0")
    print("  x + y² - 7 = 0")
    print()
    
    f1 = lambda x, y: x**2 + y - 11
    f2 = lambda x, y: x + y**2 - 7
    
    solver = NonLinearGaussSeidel(functions=[f1, f2], verbose=False)
    
    # Try different initial guesses to find different solutions
    initial_guesses = [
        ([0, 0], "(3, 2)"),
        ([-3, 3], "(-2.8, 3.1)"),
        ([-4, -3], "(-3.8, -3.3)"),
        ([3, -2], "(3.6, -1.8)")
    ]
    
    for guess, expected in initial_guesses:
        result = solver.solve(initial_guess=guess)
        print(f"Initial guess {guess} → Solution: x={result.x[0]:.4f}, y={result.x[1]:.4f}")
        print(f"  Expected: {expected}, Residual: {result.residual:.2e}, "
              f"Iterations: {result.iterations}")
    
    print()


def example_2_3d_system():
    """
    3D nonlinear system.
    
    System:
        x² + y² + z² - 14 = 0
        x + y + z - 6 = 0
        xyz - 6 = 0
    
    One solution: (3, 2, 1)
    """
    print("=" * 70)
    print("Example 2: 3D Nonlinear System")
    print("=" * 70)
    print("System:")
    print("  x² + y² + z² - 14 = 0")
    print("  x + y + z - 6 = 0")
    print("  xyz - 6 = 0")
    print()
    
    f1 = lambda x, y, z: x**2 + y**2 + z**2 - 14
    f2 = lambda x, y, z: x + y + z - 6
    f3 = lambda x, y, z: x * y * z - 6
    
    solver = NonLinearGaussSeidel(functions=[f1, f2, f3], verbose=True)
    result = solver.solve(initial_guess=[1, 1, 1])
    
    print()
    print(f"✓ Solution: x={result.x[0]:.6f}, y={result.x[1]:.6f}, z={result.x[2]:.6f}")
    print(f"  Residual: {result.residual:.2e}")
    print(f"  Function evaluations: {result.function_evals}")
    
    # Verify solution
    print("\nVerification:")
    print(f"  x² + y² + z² = {result.x[0]**2 + result.x[1]**2 + result.x[2]**2:.6f} (should be 14)")
    print(f"  x + y + z = {np.sum(result.x):.6f} (should be 6)")
    print(f"  xyz = {np.prod(result.x):.6f} (should be 6)")
    print()


def example_3_transcendental():
    """
    System with transcendental functions.
    
    System:
        sin(x) + y - 1.5 = 0
        x + cos(y) - 1 = 0
    """
    print("=" * 70)
    print("Example 3: Transcendental System")
    print("=" * 70)
    print("System:")
    print("  sin(x) + y - 1.5 = 0")
    print("  x + cos(y) - 1 = 0")
    print()
    
    f1 = lambda x, y: np.sin(x) + y - 1.5
    f2 = lambda x, y: x + np.cos(y) - 1
    
    solver = NonLinearGaussSeidel(functions=[f1, f2], verbose=False)
    result = solver.solve(initial_guess=[0.5, 0.5])
    
    print(f"Solution: x={result.x[0]:.6f}, y={result.x[1]:.6f}")
    print(f"Residual: {result.residual:.2e}")
    print(f"Converged: {result.converged}")
    
    # Verify
    print("\nVerification:")
    print(f"  sin(x) + y = {np.sin(result.x[0]) + result.x[1]:.6f} (should be 1.5)")
    print(f"  x + cos(y) = {result.x[0] + np.cos(result.x[1]):.6f} (should be 1.0)")
    print()


def example_4_exponential():
    """
    System with exponential functions.
    
    System:
        e^x - y - 2 = 0
        x - e^(-y) - 1 = 0
    """
    print("=" * 70)
    print("Example 4: Exponential System")
    print("=" * 70)
    print("System:")
    print("  e^x - y - 2 = 0")
    print("  x - e^(-y) - 1 = 0")
    print()
    
    f1 = lambda x, y: np.exp(x) - y - 2
    f2 = lambda x, y: x - np.exp(-y) - 1
    
    solver = NonLinearGaussSeidel(
        functions=[f1, f2],
        tolerance=1e-8,
        max_iterations=50,
        verbose=False
    )
    result = solver.solve(initial_guess=[0.5, 0.5])
    
    print(f"Solution: x={result.x[0]:.6f}, y={result.x[1]:.6f}")
    print(f"Residual: {result.residual:.2e}")
    print(f"Iterations: {result.iterations}")
    
    # Verify
    print("\nVerification:")
    print(f"  e^x - y = {np.exp(result.x[0]) - result.x[1]:.6f} (should be 2)")
    print(f"  x - e^(-y) = {result.x[0] - np.exp(-result.x[1]):.6f} (should be 1)")
    print()


def example_5_single_equation():
    """
    Single nonlinear equation (1D).
    
    Equation: x³ - 2x - 5 = 0
    
    Known root: x ≈ 2.0946
    """
    print("=" * 70)
    print("Example 5: Single Nonlinear Equation")
    print("=" * 70)
    print("Equation: x³ - 2x - 5 = 0")
    print()
    
    f = lambda x: x**3 - 2*x - 5
    
    solver = NonLinearGaussSeidel(functions=[f], verbose=False)
    result = solver.solve(initial_guess=[2.0])
    
    print(f"Root: x = {result.x[0]:.6f}")
    print(f"Residual: {result.residual:.2e}")
    print(f"f(x) = {f(result.x[0]):.2e}")
    print()


def example_6_comparison_with_newton():
    """
    Compare performance with Newton's method (conceptual).
    
    This example shows that binary search doesn't need derivatives.
    """
    print("=" * 70)
    print("Example 6: Advantage Over Newton's Method")
    print("=" * 70)
    print("Binary-Enhanced Gauss-Seidel advantages:")
    print("  ✓ No derivatives needed")
    print("  ✓ Automatic hyperparameter tuning")
    print("  ✓ Robust to initial guess")
    print()
    
    # System where derivatives are tedious to compute
    f1 = lambda x, y: x**2 * np.sin(y) - 1
    f2 = lambda x, y: y**2 * np.cos(x) - 1
    
    print("System:")
    print("  x²·sin(y) - 1 = 0")
    print("  y²·cos(x) - 1 = 0")
    print()
    print("Newton's method would require:")
    print("  ∂f₁/∂x = 2x·sin(y)")
    print("  ∂f₁/∂y = x²·cos(y)")
    print("  ∂f₂/∂x = -y²·sin(x)")
    print("  ∂f₂/∂y = 2y·cos(x)")
    print()
    print("Binary search: No derivatives needed!")
    print()
    
    solver = NonLinearGaussSeidel(functions=[f1, f2], verbose=False)
    result = solver.solve(initial_guess=[1.0, 1.0])
    
    print(f"Solution: x={result.x[0]:.6f}, y={result.x[1]:.6f}")
    print(f"Residual: {result.residual:.2e}")
    print(f"Function evaluations: {result.function_evals}")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "NonLinear Gauss-Seidel Examples" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    example_1_himmelblau()
    example_2_3d_system()
    example_3_transcendental()
    example_4_exponential()
    example_5_single_equation()
    example_6_comparison_with_newton()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
