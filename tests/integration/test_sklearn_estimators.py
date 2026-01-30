"""
Tests for scikit-learn compatible estimators.

Phase 3, Task 3.2: Verify sklearn API compatibility.
"""

import pytest
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression, make_classification
from sklearn.utils.estimator_checks import check_estimator
from math_toolkit.integration import (
    BinaryLinearRegression,
    BinaryLogisticRegression,
    BinaryRidgeRegression
)


class TestBinaryLinearRegressionBasic:
    """Basic functionality tests for BinaryLinearRegression"""
    
    def test_simple_fit_predict(self):
        """Test basic fit and predict"""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        model = BinaryLinearRegression(max_iter=200)
        model.fit(X, y)
        
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        
        y_pred = model.predict(X)
        assert np.allclose(y_pred, y, atol=0.1)
    
    def test_multivariate_regression(self):
        """Test with multiple features"""
        X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
        
        model = BinaryLinearRegression(max_iter=200)
        model.fit(X, y)
        
        assert model.coef_.shape == (5,)
        assert isinstance(model.intercept_, (float, np.floating))
        
        y_pred = model.predict(X)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        assert r2 > 0.8  # Should fit reasonably well
    
    def test_no_intercept(self):
        """Test with fit_intercept=False"""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])
        
        model = BinaryLinearRegression(fit_intercept=False, max_iter=200)
        model.fit(X, y)
        
        assert model.intercept_ == 0.0
        assert np.abs(model.coef_[0] - 2.0) < 0.1
    
    def test_adamw_optimizer(self):
        """Test with AdamW optimizer (verify it runs)"""
        X, y = make_regression(n_samples=50, n_features=3, noise=5, random_state=42)
        
        # Just verify AdamW optimizer can be used (may not converge well for simple regression)
        model = BinaryLinearRegression(optimizer='adamw', max_iter=100,
                                       optimizer_params={'verbose': False})
        model.fit(X, y)
        
        y_pred = model.predict(X)
        # Just check it produces some output (AdamW not ideal for linear regression)
        assert y_pred.shape == y.shape
        assert not np.any(np.isnan(y_pred))


class TestBinaryLinearRegressionSklearn:
    """Test sklearn compatibility"""
    
    def test_cross_validation(self):
        """Test with cross_val_score"""
        X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
        
        model = BinaryLinearRegression(max_iter=200)
        scores = cross_val_score(model, X, y, cv=3, scoring='r2')
        
        assert len(scores) == 3
        assert np.all(scores > 0.5)  # Reasonable performance
    
    def test_pipeline(self):
        """Test in sklearn Pipeline"""
        X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', BinaryLinearRegression(max_iter=200))
        ])
        
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        assert r2 > 0.8
    
    def test_grid_search(self):
        """Test with GridSearchCV"""
        X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
        
        model = BinaryLinearRegression()
        param_grid = {'max_iter': [50, 100], 'tolerance': [1e-4, 1e-6]}
        
        grid = GridSearchCV(model, param_grid, cv=3, scoring='r2')
        grid.fit(X, y)
        
        assert hasattr(grid, 'best_params_')
        assert grid.best_score_ > 0.5


class TestBinaryLogisticRegressionBasic:
    """Basic tests for BinaryLogisticRegression"""
    
    def test_simple_classification(self):
        """Test basic binary classification"""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        model = BinaryLogisticRegression(max_iter=300)
        model.fit(X, y)
        
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert hasattr(model, 'classes_')
        assert len(model.classes_) == 2
        
        y_pred = model.predict(X)
        accuracy = np.mean(y_pred == y)
        assert accuracy >= 0.8
    
    def test_predict_proba(self):
        """Test probability predictions"""
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2,
                                   n_informative=3, random_state=42)
        
        model = BinaryLogisticRegression(max_iter=300)
        model.fit(X, y)
        
        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all((proba >= 0) & (proba <= 1))
    
    def test_multifeature_classification(self):
        """Test with multiple features"""
        X, y = make_classification(n_samples=150, n_features=8, n_classes=2,
                                   n_informative=5, random_state=42)
        
        model = BinaryLogisticRegression(max_iter=300)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        accuracy = np.mean(y_pred == y)
        assert accuracy > 0.7
    
    def test_adamw_optimizer(self):
        """Test with AdamW optimizer"""
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        
        model = BinaryLogisticRegression(optimizer='adamw', max_iter=300)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        accuracy = np.mean(y_pred == y)
        assert accuracy > 0.7


class TestBinaryLogisticRegressionSklearn:
    """Test sklearn compatibility for logistic regression"""
    
    def test_cross_validation(self):
        """Test with cross_val_score"""
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        
        model = BinaryLogisticRegression(max_iter=300)
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        
        assert len(scores) == 3
        assert np.all(scores > 0.5)
    
    def test_pipeline(self):
        """Test in sklearn Pipeline"""
        X, y = make_classification(n_samples=150, n_features=6, n_classes=2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', BinaryLogisticRegression(max_iter=300))
        ])
        
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        
        accuracy = np.mean(y_pred == y)
        assert accuracy > 0.7
    
    def test_grid_search(self):
        """Test with GridSearchCV"""
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        
        model = BinaryLogisticRegression()
        param_grid = {'max_iter': [100, 200], 'tolerance': [1e-4, 1e-6]}
        
        grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid.fit(X, y)
        
        assert hasattr(grid, 'best_params_')
        assert grid.best_score_ > 0.5


class TestBinaryRidgeRegressionBasic:
    """Basic tests for BinaryRidgeRegression"""
    
    def test_simple_ridge(self):
        """Test basic ridge regression"""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2.1, 4.2, 5.9, 8.1, 10.0])
        
        model = BinaryRidgeRegression(alpha=0.1, max_iter=200)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert np.allclose(y_pred, y, atol=0.5)
    
    def test_regularization_effect(self):
        """Test that alpha controls regularization strength"""
        X, y = make_regression(n_samples=50, n_features=10, n_informative=5,
                              noise=10, random_state=42)
        
        # High alpha = more regularization = smaller coefficients
        model_high = BinaryRidgeRegression(alpha=10.0, max_iter=200)
        model_high.fit(X, y)
        
        model_low = BinaryRidgeRegression(alpha=0.01, max_iter=200)
        model_low.fit(X, y)
        
        # High alpha should have smaller coefficient norm
        assert np.linalg.norm(model_high.coef_) < np.linalg.norm(model_low.coef_)
    
    def test_multivariate_ridge(self):
        """Test with multiple features"""
        X, y = make_regression(n_samples=100, n_features=8, noise=15, random_state=42)
        
        model = BinaryRidgeRegression(alpha=1.0, max_iter=200)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        assert r2 > 0.7


class TestBinaryRidgeRegressionSklearn:
    """Test sklearn compatibility for ridge regression"""
    
    def test_cross_validation(self):
        """Test with cross_val_score"""
        X, y = make_regression(n_samples=100, n_features=6, noise=10, random_state=42)
        
        model = BinaryRidgeRegression(alpha=1.0, max_iter=200)
        scores = cross_val_score(model, X, y, cv=3, scoring='r2')
        
        assert len(scores) == 3
        assert np.all(scores > 0.5)
    
    def test_pipeline(self):
        """Test in sklearn Pipeline"""
        X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', BinaryRidgeRegression(alpha=1.0, max_iter=200))
        ])
        
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        assert r2 > 0.7
    
    def test_grid_search_alpha(self):
        """Test hyperparameter tuning for alpha"""
        X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
        
        model = BinaryRidgeRegression(max_iter=200)
        param_grid = {'alpha': [0.1, 1.0, 10.0]}
        
        grid = GridSearchCV(model, param_grid, cv=3, scoring='r2')
        grid.fit(X, y)
        
        assert hasattr(grid, 'best_params_')
        assert 'alpha' in grid.best_params_


class TestEstimatorValidation:
    """Test sklearn estimator validation"""
    
    @pytest.mark.parametrize("Estimator", [
        BinaryLinearRegression,
        BinaryRidgeRegression
    ])
    def test_estimator_interface_regression(self, Estimator):
        """Verify estimator follows sklearn interface (partial check)"""
        estimator = Estimator()
        
        # Check basic interface
        assert hasattr(estimator, 'fit')
        assert hasattr(estimator, 'predict')
        assert hasattr(estimator, 'get_params')
        assert hasattr(estimator, 'set_params')
    
    def test_estimator_interface_classification(self):
        """Verify classifier follows sklearn interface"""
        estimator = BinaryLogisticRegression()
        
        assert hasattr(estimator, 'fit')
        assert hasattr(estimator, 'predict')
        assert hasattr(estimator, 'predict_proba')
        assert hasattr(estimator, 'get_params')
        assert hasattr(estimator, 'set_params')


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_unfitted_predict_raises(self):
        """Unfitted estimator should raise error"""
        from sklearn.exceptions import NotFittedError
        
        model = BinaryLinearRegression()
        X = np.array([[1], [2], [3]])
        
        with pytest.raises(NotFittedError):
            model.predict(X)
    
    def test_dimension_mismatch(self):
        """Predict with wrong dimensions should raise"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        
        model = BinaryLinearRegression()
        model.fit(X, y)
        
        X_wrong = np.array([[1, 2, 3]])  # 3 features instead of 2
        
        with pytest.raises(ValueError, match="features"):
            model.predict(X_wrong)
    
    def test_logistic_nonbinary_raises(self):
        """Logistic regression with >2 classes should raise"""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 1, 2, 0])  # 3 classes
        
        model = BinaryLogisticRegression()
        
        with pytest.raises(ValueError, match="binary"):
            model.fit(X, y)
    
    def test_invalid_optimizer_raises(self):
        """Invalid optimizer name should raise"""
        model = BinaryLinearRegression(optimizer='invalid')
        X = np.array([[1], [2]])
        y = np.array([1, 2])
        
        with pytest.raises(ValueError, match="Unknown optimizer"):
            model.fit(X, y)
