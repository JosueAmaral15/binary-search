"""
Visualization Tools Package

Provides plotting utilities for optimization and solver convergence analysis.
"""

from .plots import (
    OptimizationVisualizer,
    plot_convergence,
    plot_learning_rate,
    plot_cost_landscape,
    plot_parameter_trajectory,
    compare_optimizers
)

__all__ = [
    'OptimizationVisualizer',
    'plot_convergence',
    'plot_learning_rate',
    'plot_cost_landscape',
    'plot_parameter_trajectory',
    'compare_optimizers'
]
