"""
Visualization utilities for optimization and solver convergence analysis.

Provides plotting functions for:
- Convergence plots (cost vs iterations)
- Learning rate evolution
- 2D cost landscapes with optimization paths
- Parameter trajectories
- Optimizer comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, List, Dict, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class OptimizationVisualizer:
    """
    Comprehensive visualization toolkit for optimization analysis.
    
    Provides static methods for creating various plots from optimizer history.
    All methods return matplotlib Figure and Axes for further customization.
    
    Examples
    --------
    >>> from math_toolkit.optimization import BinaryRateOptimizer
    >>> from math_toolkit.visualization import OptimizationVisualizer
    >>> 
    >>> optimizer = BinaryRateOptimizer()
    >>> theta = optimizer.optimize(X, y, initial_theta, cost_fn, grad_fn)
    >>> 
    >>> # Plot convergence
    >>> fig, ax = OptimizationVisualizer.plot_convergence(optimizer)
    >>> plt.show()
    >>> 
    >>> # Plot learning rate evolution
    >>> fig, ax = OptimizationVisualizer.plot_learning_rate(optimizer)
    >>> plt.show()
    """
    
    @staticmethod
    def plot_convergence(optimizer, title: str = "Convergence", 
                        figsize: Tuple[int, int] = (10, 6),
                        show_log_scale: bool = False) -> Tuple[Figure, Axes]:
        """
        Plot cost function convergence over iterations.
        
        Parameters
        ----------
        optimizer : Optimizer object
            Optimizer with history dict containing 'cost' key
        title : str, default="Convergence"
            Plot title
        figsize : tuple, default=(10, 6)
            Figure size (width, height) in inches
        show_log_scale : bool, default=False
            Use logarithmic scale for y-axis
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        ax : matplotlib.axes.Axes
            Axes object
        
        Examples
        --------
        >>> fig, ax = OptimizationVisualizer.plot_convergence(optimizer)
        >>> ax.set_ylabel('Loss')
        >>> plt.savefig('convergence.png')
        """
        if not hasattr(optimizer, 'history') or 'cost' not in optimizer.history:
            raise ValueError("Optimizer must have history dict with 'cost' key")
        
        cost_history = optimizer.history['cost']
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(cost_history, linewidth=2, color='#2E86AB', marker='o', 
                markersize=3, markevery=max(1, len(cost_history) // 20))
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Cost', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if show_log_scale and len(cost_history) > 1:
            ax.set_yscale('log')
        
        # Add final cost annotation
        final_cost = cost_history[-1]
        ax.annotate(f'Final: {final_cost:.6f}',
                   xy=(len(cost_history) - 1, final_cost),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_learning_rate(optimizer, title: str = "Learning Rate Evolution",
                          figsize: Tuple[int, int] = (10, 6),
                          use_log_scale: bool = True) -> Tuple[Figure, Axes]:
        """
        Plot learning rate evolution (for optimizers that track it).
        
        Parameters
        ----------
        optimizer : Optimizer object
            Optimizer with history dict containing 'learning_rate' key
        title : str, default="Learning Rate Evolution"
            Plot title
        figsize : tuple, default=(10, 6)
            Figure size
        use_log_scale : bool, default=True
            Use logarithmic scale for y-axis
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        ax : matplotlib.axes.Axes
            Axes object
        """
        if not hasattr(optimizer, 'history') or 'learning_rate' not in optimizer.history:
            raise ValueError("Optimizer must have history dict with 'learning_rate' key")
        
        lr_history = optimizer.history['learning_rate']
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(lr_history, linewidth=2, color='#A23B72', marker='s',
                markersize=3, markevery=max(1, len(lr_history) // 20))
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if use_log_scale:
            ax.set_yscale('log')
        
        # Add statistics
        mean_lr = np.mean(lr_history)
        ax.axhline(mean_lr, color='red', linestyle='--', alpha=0.5, 
                   label=f'Mean: {mean_lr:.6f}')
        ax.legend()
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_cost_landscape(cost_func, theta_history: np.ndarray,
                           param_ranges: Tuple[Tuple[float, float], Tuple[float, float]],
                           title: str = "Cost Landscape",
                           figsize: Tuple[int, int] = (12, 8),
                           resolution: int = 50) -> Tuple[Figure, Axes]:
        """
        Plot 2D cost landscape with optimization trajectory (for 2-parameter problems).
        
        Parameters
        ----------
        cost_func : callable
            Cost function f(theta) returning scalar cost
        theta_history : np.ndarray, shape (n_iterations, 2)
            Parameter trajectory through optimization
        param_ranges : tuple of tuples
            ((theta0_min, theta0_max), (theta1_min, theta1_max))
        title : str, default="Cost Landscape"
            Plot title
        figsize : tuple, default=(12, 8)
            Figure size
        resolution : int, default=50
            Grid resolution for contour plot
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        ax : matplotlib.axes.Axes
            Axes object
        
        Examples
        --------
        >>> def cost_fn(theta):
        ...     return (theta[0] - 3)**2 + (theta[1] - 2)**2
        >>> fig, ax = OptimizationVisualizer.plot_cost_landscape(
        ...     cost_fn, theta_history, param_ranges=((0, 6), (0, 4))
        ... )
        """
        if theta_history.ndim != 2 or theta_history.shape[1] != 2:
            raise ValueError("theta_history must have shape (n_iterations, 2)")
        
        (theta0_range, theta1_range) = param_ranges
        
        # Create grid
        theta0 = np.linspace(theta0_range[0], theta0_range[1], resolution)
        theta1 = np.linspace(theta1_range[0], theta1_range[1], resolution)
        Theta0, Theta1 = np.meshgrid(theta0, theta1)
        
        # Compute cost on grid
        Z = np.zeros_like(Theta0)
        for i in range(resolution):
            for j in range(resolution):
                theta = np.array([Theta0[i, j], Theta1[i, j]])
                try:
                    Z[i, j] = cost_func(theta)
                except:
                    Z[i, j] = np.nan
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Contour plot
        levels = np.logspace(np.log10(np.nanmin(Z) + 1e-10), 
                            np.log10(np.nanmax(Z) + 1e-10), 20)
        contour = ax.contour(Theta0, Theta1, Z, levels=levels, cmap='viridis', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.2e')
        
        # Filled contour
        ax.contourf(Theta0, Theta1, Z, levels=levels, cmap='viridis', alpha=0.3)
        
        # Plot optimization path
        ax.plot(theta_history[:, 0], theta_history[:, 1], 
                'r-', linewidth=2, label='Optimization Path', alpha=0.8)
        ax.plot(theta_history[0, 0], theta_history[0, 1], 
                'go', markersize=10, label='Start')
        ax.plot(theta_history[-1, 0], theta_history[-1, 1], 
                'r*', markersize=15, label='End')
        
        ax.set_xlabel('θ₀', fontsize=12)
        ax.set_ylabel('θ₁', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(contour, ax=ax, label='Cost')
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_parameter_trajectory(optimizer, param_names: Optional[List[str]] = None,
                                 title: str = "Parameter Trajectory",
                                 figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, Axes]:
        """
        Plot evolution of parameters over iterations.
        
        Parameters
        ----------
        optimizer : Optimizer object
            Optimizer with history dict containing 'theta' key
        param_names : list of str, optional
            Names for parameters (e.g., ['θ₀', 'θ₁'])
        title : str, default="Parameter Trajectory"
            Plot title
        figsize : tuple, default=(10, 6)
            Figure size
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        ax : matplotlib.axes.Axes
            Axes object
        """
        if not hasattr(optimizer, 'history') or 'theta' not in optimizer.history:
            raise ValueError("Optimizer must have history dict with 'theta' key")
        
        theta_history = optimizer.history['theta']
        n_params = theta_history.shape[1] if theta_history.ndim > 1 else 1
        
        if param_names is None:
            param_names = [f'θ{i}' for i in range(n_params)]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if theta_history.ndim == 1:
            theta_history = theta_history.reshape(-1, 1)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_params))
        
        for i in range(n_params):
            ax.plot(theta_history[:, i], linewidth=2, color=colors[i],
                   label=param_names[i], marker='o', markersize=3,
                   markevery=max(1, len(theta_history) // 20))
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Parameter Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def compare_optimizers(optimizers: Dict[str, Any],
                          title: str = "Optimizer Comparison",
                          figsize: Tuple[int, int] = (12, 5)) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """
        Compare multiple optimizers side-by-side.
        
        Parameters
        ----------
        optimizers : dict
            Dictionary mapping optimizer names to optimizer objects
            e.g., {'BinaryRate': opt1, 'AdamW': opt2}
        title : str, default="Optimizer Comparison"
            Plot title
        figsize : tuple, default=(12, 5)
            Figure size
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        axes : tuple of matplotlib.axes.Axes
            (ax_cost, ax_lr) axes objects
        
        Examples
        --------
        >>> opt1 = BinaryRateOptimizer()
        >>> opt2 = AdamW()
        >>> # ... optimize both ...
        >>> fig, (ax1, ax2) = OptimizationVisualizer.compare_optimizers({
        ...     'Binary': opt1,
        ...     'AdamW': opt2
        ... })
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(optimizers)))
        
        # Plot convergence comparison
        for (name, optimizer), color in zip(optimizers.items(), colors):
            if hasattr(optimizer, 'history') and 'cost' in optimizer.history:
                cost = optimizer.history['cost']
                ax1.plot(cost, label=name, linewidth=2, color=color)
        
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Cost', fontsize=11)
        ax1.set_title('Convergence Comparison', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot learning rate comparison (if available)
        has_lr = False
        for (name, optimizer), color in zip(optimizers.items(), colors):
            if hasattr(optimizer, 'history') and 'learning_rate' in optimizer.history:
                lr = optimizer.history['learning_rate']
                ax2.plot(lr, label=name, linewidth=2, color=color)
                has_lr = True
        
        if has_lr:
            ax2.set_xlabel('Iteration', fontsize=11)
            ax2.set_ylabel('Learning Rate', fontsize=11)
            ax2.set_title('Learning Rate Comparison', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'Learning rate data\nnot available',
                    ha='center', va='center', fontsize=12, color='gray')
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig, (ax1, ax2)


# Convenience functions (wrappers around OptimizationVisualizer)

def plot_convergence(optimizer, **kwargs) -> Tuple[Figure, Axes]:
    """Plot convergence. See OptimizationVisualizer.plot_convergence for details."""
    return OptimizationVisualizer.plot_convergence(optimizer, **kwargs)


def plot_learning_rate(optimizer, **kwargs) -> Tuple[Figure, Axes]:
    """Plot learning rate. See OptimizationVisualizer.plot_learning_rate for details."""
    return OptimizationVisualizer.plot_learning_rate(optimizer, **kwargs)


def plot_cost_landscape(cost_func, theta_history, param_ranges, **kwargs) -> Tuple[Figure, Axes]:
    """Plot cost landscape. See OptimizationVisualizer.plot_cost_landscape for details."""
    return OptimizationVisualizer.plot_cost_landscape(
        cost_func, theta_history, param_ranges, **kwargs
    )


def plot_parameter_trajectory(optimizer, **kwargs) -> Tuple[Figure, Axes]:
    """Plot parameter trajectory. See OptimizationVisualizer.plot_parameter_trajectory for details."""
    return OptimizationVisualizer.plot_parameter_trajectory(optimizer, **kwargs)


def compare_optimizers(optimizers, **kwargs) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Compare optimizers. See OptimizationVisualizer.compare_optimizers for details."""
    return OptimizationVisualizer.compare_optimizers(optimizers, **kwargs)
