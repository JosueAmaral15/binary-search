"""
Profile WeightCombinationSearch to identify hot functions for Numba optimization.
"""

import sys
import cProfile
import pstats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from math_toolkit.binary_search.combinatorial import WeightCombinationSearch


def profile_search():
    """Profile a typical search operation."""
    coeffs = list(range(1, 16))  # 15 parameters
    target = 60
    
    search = WeightCombinationSearch(
        tolerance=2.0,
        max_iter=20,
        adaptive_sampling=False,  # Disable to see exhaustive performance
        verbose=False
    )
    
    # Run the search
    weights = search.find_optimal_weights(coeffs, target)
    print(f"Result weights: {weights}")


if __name__ == '__main__':
    print("Profiling WeightCombinationSearch...")
    print("=" * 80)
    
    # Profile the search
    profiler = cProfile.Profile()
    profiler.enable()
    profile_search()
    profiler.disable()
    
    # Print statistics
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    
    print("\n" + "=" * 80)
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME:")
    print("=" * 80)
    stats.print_stats(20)
    
    print("\n" + "=" * 80)
    print("FUNCTIONS WITH MOST TIME (self time):")
    print("=" * 80)
    stats.sort_stats('time')
    stats.print_stats(20)
