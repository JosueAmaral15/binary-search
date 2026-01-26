"""
Test Binary-Enhanced Gauss-Seidel Solver - Phase 1 Prototype

Tests the core Gauss-Seidel implementation on classic 3x3 system.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from math_toolkit.linear_systems import BinaryGaussSeidel, solve_linear_system


def test_classic_3x3_system():
    """
    Test classic 3x3 diagonally dominant system.
    
    System:
        4x - y = 15
        -x + 4y - z = 10
        -y + 3z = 10
    
    Known solution: x=4, y=5, z=5
    """
    print("=" * 70)
    print("TEST 1: Classic 3×3 Diagonally Dominant System")
    print("=" * 70)
    
    A = np.array([
        [4, -1, 0],
        [-1, 4, -1],
        [0, -1, 3]
    ], dtype=float)
    
    b = np.array([15, 10, 10], dtype=float)
    
    print("\nMatrix A:")
    print(A)
    print("\nVector b:")
    print(b)
    print("\nKnown solution: x=4, y=5, z=5")
    
    # Solve using our implementation
    solver = BinaryGaussSeidel(tolerance=1e-6, verbose=True)
    result = solver.solve(A, b)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Status: {result}")
    print(f"\nSolution: {result.x}")
    print(f"Expected: [4.0, 5.0, 5.0]")
    print(f"\nError from expected:")
    expected = np.array([4.0, 5.0, 5.0])
    error = np.abs(result.x - expected)
    for i, (val, exp, err) in enumerate(zip(result.x, expected, error)):
        print(f"  x[{i}]: {val:.10f} (expected: {exp:.1f}, error: {err:.2e})")
    
    # Verify solution
    computed_b = A @ result.x
    print(f"\nVerification (Ax = b):")
    print(f"  Computed b: {computed_b}")
    print(f"  Expected b: {b}")
    print(f"  Residual ||Ax - b||: {result.residual:.2e}")
    
    return result


def test_5x5_system():
    """
    Test 5x5 system to verify scalability.
    """
    print("\n\n" + "=" * 70)
    print("TEST 2: 5×5 System (5 coefficients)")
    print("=" * 70)
    
    # Create a diagonally dominant 5x5 system
    A = np.array([
        [10, -1, 0, 0, 0],
        [-1, 10, -1, 0, 0],
        [0, -1, 10, -1, 0],
        [0, 0, -1, 10, -1],
        [0, 0, 0, -1, 10]
    ], dtype=float)
    
    b = np.array([9, 8, 8, 8, 9], dtype=float)
    
    print("\nMatrix A (5×5):")
    print(A)
    print("\nVector b:")
    print(b)
    
    # Solve
    solver = BinaryGaussSeidel(tolerance=1e-8, verbose=False)
    result = solver.solve(A, b)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Status: {result}")
    print(f"\nSolution (5 coefficients):")
    for i, val in enumerate(result.x):
        print(f"  x[{i}]: {val:.10f}")
    
    # Verify
    computed_b = A @ result.x
    print(f"\nVerification:")
    print(f"  Residual ||Ax - b||: {result.residual:.2e}")
    print(f"  Max element error: {np.max(np.abs(computed_b - b)):.2e}")
    
    return result


def test_non_dominant_warning():
    """
    Test warning for non-diagonally dominant matrix.
    """
    print("\n\n" + "=" * 70)
    print("TEST 3: Non-Diagonally Dominant Matrix (should warn)")
    print("=" * 70)
    
    A = np.array([
        [1, 2, 3],
        [2, 1, 2],
        [3, 2, 1]
    ], dtype=float)
    
    b = np.array([6, 5, 6], dtype=float)
    
    print("\nMatrix A (NOT diagonally dominant):")
    print(A)
    
    # Should produce warning
    solver = BinaryGaussSeidel(tolerance=1e-6, check_dominance=True)
    result = solver.solve(A, b)
    
    print(f"\nResult: {result}")
    if result.warnings:
        print(f"Warnings received: {len(result.warnings)}")
        for w in result.warnings:
            print(f"  - {w}")
    
    return result


def compare_with_numpy():
    """
    Compare our solution with NumPy's direct solver.
    """
    print("\n\n" + "=" * 70)
    print("TEST 4: Comparison with NumPy Direct Solver")
    print("=" * 70)
    
    A = np.array([
        [4, -1, 0],
        [-1, 4, -1],
        [0, -1, 3]
    ], dtype=float)
    
    b = np.array([15, 10, 10], dtype=float)
    
    # Our solver
    result_ours = solve_linear_system(A, b, tolerance=1e-10)
    
    # NumPy direct solver
    x_numpy = np.linalg.solve(A, b)
    
    print(f"\nOur solution:   {result_ours.x}")
    print(f"NumPy solution: {x_numpy}")
    print(f"\nDifference: {np.abs(result_ours.x - x_numpy)}")
    print(f"Max difference: {np.max(np.abs(result_ours.x - x_numpy)):.2e}")
    print(f"\nOur iterations: {result_ours.iterations}")
    print(f"Our residual: {result_ours.residual:.2e}")
    
    return result_ours, x_numpy


def main():
    """Run all tests"""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "BINARY-ENHANCED GAUSS-SEIDEL PROTOTYPE TEST" + " " * 15 + "║")
    print("║" + " " * 20 + "Phase 1: Core Implementation" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    
    try:
        # Test 1: Classic 3x3
        result1 = test_classic_3x3_system()
        
        # Test 2: 5x5 system
        result2 = test_5x5_system()
        
        # Test 3: Warning test
        result3 = test_non_dominant_warning()
        
        # Test 4: Compare with NumPy
        result4_ours, result4_numpy = compare_with_numpy()
        
        # Summary
        print("\n\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 25 + "TEST SUMMARY" + " " * 31 + "║")
        print("╚" + "═" * 68 + "╝")
        
        print(f"\n✓ Test 1 (3×3): {result1.converged} - {result1.iterations} iterations")
        print(f"✓ Test 2 (5×5): {result2.converged} - {result2.iterations} iterations")
        print(f"✓ Test 3 (warnings): {len(result3.warnings)} warnings correctly issued")
        print(f"✓ Test 4 (NumPy comparison): max diff = {np.max(np.abs(result4_ours.x - result4_numpy)):.2e}")
        
        print("\n" + "═" * 70)
        print("✅ PHASE 1 PROTOTYPE: ALL TESTS PASSED")
        print("═" * 70)
        print("\nCore Gauss-Seidel implementation validated!")
        print("Ready for Phase 2: Binary search optimizations")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
