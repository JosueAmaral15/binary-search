"""
Optimization Algorithms Module.

Provides gradient-based optimizers using binary search concepts for
adaptive hyperparameter selection.

Classes:
    BinaryRateOptimizer: Gradient descent with binary search learning rate
    AdamW: Adaptive moment estimation with weight decay and binary search
    ObserverAdamW: Parallel hyperparameter tuning with observer pattern
"""

from math_toolkit.optimization.gradient_descent import BinaryRateOptimizer
from math_toolkit.optimization.adaptive_optimizer import AdamW
from math_toolkit.optimization.observer_tuning import ObserverAdamW

__all__ = ['BinaryRateOptimizer', 'AdamW', 'ObserverAdamW']
