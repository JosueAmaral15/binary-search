"""
Tests for visualization module.

Phase 3, Task 3.3: Verify plotting functionality.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from math_toolkit.visualization import (
    OptimizationVisualizer,
    plot_convergence,
    plot_learning_rate,
    plot_cost_landscape,
    plot_parameter_trajectory,
    compare_optimizers
)
from math_toolkit.optimization import BinaryRateOptimizer, AdamW


class MockOptimizer:
    """Mock optimizer for testing visualization"""
    
    def __init__(self, cost_history, lr_history=None, theta_history=None):
        self.history = {'cost': cost_history}
        if lr_history is not None:
            self.history['learning_rate'] = lr_history
        if theta_history is not None:
            self.history['theta'] = theta_history


class TestPlotConvergence:
    """Test convergence plotting"""
    
    def test_basic_convergence_plot(self):
        """Test basic convergence plotting"""
        cost = [100, 50, 25, 12, 6, 3, 1.5]
        opt = MockOptimizer(cost)
        
        fig, ax = plot_convergence(opt)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == 'Iteration'
        assert ax.get_ylabel() == 'Cost'
        plt.close(fig)
    
    def test_convergence_log_scale(self):
        """Test convergence plot with log scale"""
        cost = [1000, 100, 10, 1, 0.1, 0.01]
        opt = MockOptimizer(cost)
        
        fig, ax = plot_convergence(opt, show_log_scale=True)
        
        assert ax.get_yscale() == 'log'
        plt.close(fig)
    
    def test_convergence_custom_title(self):
        """Test custom title"""
        cost = [10, 5, 2, 1]
        opt = MockOptimizer(cost)
        
        fig, ax = plot_convergence(opt, title="My Custom Title")
        
        assert "My Custom Title" in ax.get_title()
        plt.close(fig)
    
    def test_convergence_no_history_raises(self):
        """Test error when optimizer has no history"""
        class BadOptimizer:
            pass
        
        opt = BadOptimizer()
        
        with pytest.raises(ValueError, match="history"):
            plot_convergence(opt)
    
    def test_convergence_missing_cost_raises(self):
        """Test error when history missing 'cost'"""
        class BadOptimizer:
            history = {'other': [1, 2, 3]}
        
        opt = BadOptimizer()
        
        with pytest.raises(ValueError, match="cost"):
            plot_convergence(opt)


class TestPlotLearningRate:
    """Test learning rate plotting"""
    
    def test_basic_lr_plot(self):
        """Test basic learning rate plot"""
        lr = [0.1, 0.05, 0.025, 0.0125, 0.00625]
        opt = MockOptimizer([10, 5, 2, 1, 0.5], lr_history=lr)
        
        fig, ax = plot_learning_rate(opt)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == 'Iteration'
        assert ax.get_ylabel() == 'Learning Rate'
        plt.close(fig)
    
    def test_lr_log_scale_default(self):
        """Test that log scale is used by default"""
        lr = [1.0, 0.1, 0.01, 0.001]
        opt = MockOptimizer([10, 5, 2, 1], lr_history=lr)
        
        fig, ax = plot_learning_rate(opt)
        
        assert ax.get_yscale() == 'log'
        plt.close(fig)
    
    def test_lr_no_log_scale(self):
        """Test disabling log scale"""
        lr = [0.1, 0.09, 0.08, 0.07]
        opt = MockOptimizer([10, 5, 2, 1], lr_history=lr)
        
        fig, ax = plot_learning_rate(opt, use_log_scale=False)
        
        assert ax.get_yscale() == 'linear'
        plt.close(fig)
    
    def test_lr_missing_raises(self):
        """Test error when learning_rate not in history"""
        opt = MockOptimizer([10, 5, 2, 1])  # No lr_history
        
        with pytest.raises(ValueError, match="learning_rate"):
            plot_learning_rate(opt)


class TestPlotCostLandscape:
    """Test cost landscape plotting"""
    
    def test_basic_landscape(self):
        """Test basic 2D cost landscape"""
        def cost_fn(theta):
            return (theta[0] - 2)**2 + (theta[1] - 3)**2
        
        theta_history = np.array([
            [0, 0],
            [1, 1],
            [1.5, 2],
            [1.8, 2.7],
            [2, 3]
        ])
        
        fig, ax = plot_cost_landscape(
            cost_fn, theta_history,
            param_ranges=((0, 4), (0, 6)),
            resolution=20  # Low resolution for speed
        )
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)
    
    def test_landscape_with_nan_handling(self):
        """Test landscape handles NaN from cost function"""
        def cost_fn(theta):
            if theta[0] < 0:
                raise ValueError("Invalid")
            return theta[0]**2 + theta[1]**2
        
        theta_history = np.array([[0.5, 0.5], [1, 1]])
        
        fig, ax = plot_cost_landscape(
            cost_fn, theta_history,
            param_ranges=((-1, 2), (-1, 2)),
            resolution=10
        )
        
        plt.close(fig)
    
    def test_landscape_wrong_dimensions_raises(self):
        """Test error when theta_history wrong shape"""
        def cost_fn(theta):
            return theta[0]**2
        
        theta_history = np.array([1, 2, 3])  # 1D instead of 2D
        
        with pytest.raises(ValueError, match="shape"):
            plot_cost_landscape(cost_fn, theta_history, ((0, 5), (0, 5)))


class TestPlotParameterTrajectory:
    """Test parameter trajectory plotting"""
    
    def test_single_parameter(self):
        """Test trajectory for single parameter"""
        theta = np.array([0, 1, 2, 3, 4])
        opt = MockOptimizer([10, 5, 2, 1, 0.5], theta_history=theta)
        
        fig, ax = plot_parameter_trajectory(opt)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_multiple_parameters(self):
        """Test trajectory for multiple parameters"""
        theta = np.array([
            [0, 0, 0],
            [1, 0.5, 0.2],
            [2, 1.0, 0.4],
            [3, 1.5, 0.6]
        ])
        opt = MockOptimizer([10, 5, 2, 1], theta_history=theta)
        
        fig, ax = plot_parameter_trajectory(opt)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_custom_param_names(self):
        """Test custom parameter names"""
        theta = np.array([[0, 0], [1, 1]])
        opt = MockOptimizer([10, 5], theta_history=theta)
        
        fig, ax = plot_parameter_trajectory(opt, param_names=['α', 'β'])
        
        # Check legend contains custom names
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'α' in legend_texts
        assert 'β' in legend_texts
        plt.close(fig)
    
    def test_trajectory_missing_raises(self):
        """Test error when theta not in history"""
        opt = MockOptimizer([10, 5, 2, 1])
        
        with pytest.raises(ValueError, match="theta"):
            plot_parameter_trajectory(opt)


class TestCompareOptimizers:
    """Test optimizer comparison"""
    
    def test_compare_two_optimizers(self):
        """Test comparing two optimizers"""
        opt1 = MockOptimizer([100, 50, 25, 12], lr_history=[0.1, 0.05, 0.025, 0.0125])
        opt2 = MockOptimizer([100, 40, 15, 5], lr_history=[0.2, 0.1, 0.05, 0.025])
        
        fig, (ax1, ax2) = compare_optimizers({
            'Optimizer 1': opt1,
            'Optimizer 2': opt2
        })
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)
        plt.close(fig)
    
    def test_compare_without_lr(self):
        """Test comparison when optimizers don't have learning rate"""
        opt1 = MockOptimizer([100, 50, 25])
        opt2 = MockOptimizer([100, 40, 20])
        
        fig, (ax1, ax2) = compare_optimizers({
            'Opt1': opt1,
            'Opt2': opt2
        })
        
        # Second axes should show "not available" message
        plt.close(fig)
    
    def test_compare_multiple_optimizers(self):
        """Test comparing multiple optimizers"""
        opts = {
            f'Opt{i}': MockOptimizer([100 / (i+1), 50 / (i+1), 25 / (i+1)])
            for i in range(4)
        }
        
        fig, axes = compare_optimizers(opts)
        
        plt.close(fig)


class TestWithRealOptimizers:
    """Test visualization with real optimizers"""
    
    def test_with_binary_rate_optimizer(self):
        """Test plotting with BinaryRateOptimizer"""
        # Simple linear regression
        X = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        
        def cost_fn(theta, X, y):
            return np.mean((X * theta - y) ** 2)
        
        def grad_fn(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)
        
        opt = BinaryRateOptimizer(max_iter=20, verbose=False)
        theta = opt.optimize(X, y, np.array([0.0]), cost_fn, grad_fn)
        
        # Test convergence plot
        fig, ax = plot_convergence(opt)
        assert len(ax.lines) > 0  # Has plot lines
        plt.close(fig)
        
        # Test learning rate plot (if available)
        if 'learning_rate' in opt.history:
            fig, ax = plot_learning_rate(opt)
            assert len(ax.lines) > 0
            plt.close(fig)
    
    def test_with_adamw(self):
        """Test plotting with AdamW optimizer"""
        X = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        
        def cost_fn(theta, X, y):
            return np.mean((X * theta - y) ** 2)
        
        def grad_fn(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)
        
        opt = AdamW(max_iter=20, verbose=False)
        theta = opt.optimize(X, y, np.array([0.0]), cost_fn, grad_fn)
        
        fig, ax = plot_convergence(opt)
        assert len(ax.lines) > 0
        plt.close(fig)
    
    def test_compare_real_optimizers(self):
        """Test comparing BinaryRateOptimizer and AdamW"""
        X = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        
        def cost_fn(theta, X, y):
            return np.mean((X * theta - y) ** 2)
        
        def grad_fn(theta, X, y):
            return 2 * np.mean((X * theta - y) * X)
        
        opt1 = BinaryRateOptimizer(max_iter=15, verbose=False)
        opt1.optimize(X, y, np.array([0.0]), cost_fn, grad_fn)
        
        opt2 = AdamW(max_iter=15, verbose=False)
        opt2.optimize(X, y, np.array([0.0]), cost_fn, grad_fn)
        
        fig, axes = compare_optimizers({
            'BinaryRate': opt1,
            'AdamW': opt2
        })
        
        plt.close(fig)


class TestConvenienceFunctions:
    """Test convenience function wrappers"""
    
    def test_convenience_functions_exist(self):
        """Test that all convenience functions are accessible"""
        from math_toolkit.visualization import (
            plot_convergence,
            plot_learning_rate,
            plot_cost_landscape,
            plot_parameter_trajectory,
            compare_optimizers
        )
        
        assert callable(plot_convergence)
        assert callable(plot_learning_rate)
        assert callable(plot_cost_landscape)
        assert callable(plot_parameter_trajectory)
        assert callable(compare_optimizers)
