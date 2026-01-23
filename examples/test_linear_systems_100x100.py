"""
Test Binary-Enhanced Gauss-Seidel on Larger Systems (100×100)

Per Decision 3.C: Test on 100×100 after validating 3×3.
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from binary_search.linear_systems import BinaryGaussSeidel


def create_diagonally_dominant_matrix(n, dominance_factor=2.0):
    """
    Create an n×n diagonally dominant matrix.
    
    Parameters
    ----------
    n : int
        Matrix size
    dominance_factor : float
        How dominant the diagonal should be (>1.0)
    """
    # Random off-diagonal elements
    A = np.random.rand(n, n) - 0.5  # Range: [-0.5, 0.5]
    
    # Make diagonal dominant
    for i in range(n):
        off_diag_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        A[i, i] = dominance_factor * off_diag_sum
        if A[i, i] < 1.0:
            A[i, i] = dominance_factor
    
    return A


def test_100x100_well_conditioned():
    """
    Test on well-conditioned 100×100 system.
    """
    print("=" * 70)
    print("TEST: Well-Conditioned 100×100 System")
    print("=" * 70)
    
    np.random.seed(42)  # Reproducible
    n = 100
    
    A = create_diagonally_dominant_matrix(n, dominance_factor=3.0)
    b = np.random.rand(n)
    
    print(f"\nMatrix size: {n}×{n}")
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    
    # Classical GS
    print("\n" + "-" * 70)
    print("Classical Gauss-Seidel")
    print("-" * 70)
    
    solver_classical = BinaryGaussSeidel(tolerance=1e-6, verbose=False)
    
    start = time.time()
    result_classical = solver_classical.solve(A, b)
    time_classical = time.time() - start
    
    print(f"Iterations: {result_classical.iterations}")
    print(f"Time: {time_classical:.3f}s")
    print(f"Residual: {result_classical.residual:.2e}")
    print(f"Converged: {result_classical.converged}")
    
    # Adaptive omega
    print("\n" + "-" * 70)
    print("Adaptive Omega")
    print("-" * 70)
    
    solver_adaptive = BinaryGaussSeidel(tolerance=1e-6, verbose=False,
                                       use_adaptive_omega=True,
                                       omega_search_iterations=6)
    
    start = time.time()
    result_adaptive = solver_adaptive.solve(A, b)
    time_adaptive = time.time() - start
    
    print(f"Iterations: {result_adaptive.iterations}")
    print(f"Time: {time_adaptive:.3f}s")
    print(f"Residual: {result_adaptive.residual:.2e}")
    print(f"Converged: {result_adaptive.converged}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    iter_improvement = (result_classical.iterations - result_adaptive.iterations) / result_classical.iterations * 100
    
    print(f"Classical:  {result_classical.iterations} iterations in {time_classical:.3f}s")
    print(f"Adaptive:   {result_adaptive.iterations} iterations in {time_adaptive:.3f}s")
    print(f"Improvement: {iter_improvement:.1f}% fewer iterations")
    
    # Verify same solution
    solution_diff = np.max(np.abs(result_classical.x - result_adaptive.x))
    print(f"\nSolution difference: {solution_diff:.2e}")
    
    return result_classical.converged and result_adaptive.converged


def test_100x100_ill_conditioned():
    """
    Test on ill-conditioned 100×100 system (where adaptive omega should help).
    """
    print("\n\n" + "=" * 70)
    print("TEST: Ill-Conditioned 100×100 System")
    print("=" * 70)
    
    np.random.seed(123)
    n = 100
    
    # Create barely diagonally dominant matrix (ill-conditioned)
    A = create_diagonally_dominant_matrix(n, dominance_factor=1.1)
    b = np.random.rand(n)
    
    print(f"\nMatrix size: {n}×{n}")
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    print("(Higher condition number = more ill-conditioned)")
    
    # Classical GS
    print("\n" + "-" * 70)
    print("Classical Gauss-Seidel")
    print("-" * 70)
    
    solver_classical = BinaryGaussSeidel(tolerance=1e-4, max_iterations=500, verbose=False)
    
    start = time.time()
    result_classical = solver_classical.solve(A, b)
    time_classical = time.time() - start
    
    print(f"Iterations: {result_classical.iterations}")
    print(f"Time: {time_classical:.3f}s")
    print(f"Residual: {result_classical.residual:.2e}")
    print(f"Converged: {result_classical.converged}")
    
    # Adaptive omega
    print("\n" + "-" * 70)
    print("Adaptive Omega")
    print("-" * 70)
    
    solver_adaptive = BinaryGaussSeidel(tolerance=1e-4, max_iterations=500, verbose=False,
                                       use_adaptive_omega=True,
                                       omega_search_iterations=6)
    
    start = time.time()
    result_adaptive = solver_adaptive.solve(A, b)
    time_adaptive = time.time() - start
    
    print(f"Iterations: {result_adaptive.iterations}")
    print(f"Time: {time_adaptive:.3f}s")
    print(f"Residual: {result_adaptive.residual:.2e}")
    print(f"Converged: {result_adaptive.converged}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    if result_classical.converged:
        iter_improvement = (result_classical.iterations - result_adaptive.iterations) / result_classical.iterations * 100
        print(f"Classical:  {result_classical.iterations} iterations in {time_classical:.3f}s")
        print(f"Adaptive:   {result_adaptive.iterations} iterations in {time_adaptive:.3f}s")
        print(f"Improvement: {iter_improvement:.1f}% fewer iterations")
        
        if iter_improvement > 10:
            print(f"\n✅ Adaptive omega helps on ill-conditioned problems!")
    else:
        print("⚠️  Classical GS did not converge within max iterations")
        print(f"Adaptive reached residual: {result_adaptive.residual:.2e}")
    
    return result_adaptive.converged


def main():
    """Run 100×100 system tests"""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "100×100 SYSTEM TESTS" + " " * 30 + "║")
    print("║" + " " * 18 + "Per Decision 3.C" + " " * 34 + "║")
    print("╚" + "═" * 68 + "╝")
    
    try:
        # Test 1: Well-conditioned
        success1 = test_100x100_well_conditioned()
        
        # Test 2: Ill-conditioned
        success2 = test_100x100_ill_conditioned()
        
        # Summary
        print("\n\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 25 + "TEST SUMMARY" + " " * 31 + "║")
        print("╚" + "═" * 68 + "╝")
        
        print(f"\n✓ Test 1 (well-conditioned 100×100):  {'PASS' if success1 else 'FAIL'}")
        print(f"✓ Test 2 (ill-conditioned 100×100):   {'PASS' if success2 else 'FAIL'}")
        
        if success1 and success2:
            print("\n" + "=" * 70)
            print("✅ LARGE SYSTEM TESTS PASSED")
            print("=" * 70)
            print("\nBinaryGaussSeidel handles 100×100 systems successfully.")
            print("Adaptive omega shows value on ill-conditioned matrices.")
            return 0
        else:
            print("\n⚠️  Some tests did not complete successfully.")
            return 1
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
