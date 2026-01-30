"""
Scikit-learn Compatible Estimators

Provides drop-in replacements for sklearn estimators using math_toolkit optimizers.
These estimators follow sklearn's API (fit/predict) and work with:
- Pipelines
- GridSearchCV/RandomizedSearchCV
- Cross-validation
- Other sklearn utilities
"""

import numpy as np
import logging
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import add_dummy_feature
from math_toolkit.optimization import BinaryRateOptimizer, AdamW

logger = logging.getLogger(__name__)


class BinaryLinearRegression(BaseEstimator, RegressorMixin):
    """
    Linear regression using BinaryRateOptimizer with binary search learning rate.
    
    Compatible with scikit-learn API (fit/predict, pipelines, cross-validation).
    
    Parameters
    ----------
    optimizer : str, default='binary'
        Optimizer to use: 'binary' (BinaryRateOptimizer) or 'adamw' (AdamW)
    max_iter : int, default=100
        Maximum number of iterations
    tolerance : float, default=1e-6
        Convergence tolerance
    fit_intercept : bool, default=True
        Whether to calculate intercept (adds bias term)
    verbose : bool, default=False
        Print optimization progress
    optimizer_params : dict, optional
        Additional parameters for the optimizer
    
    Attributes
    ----------
    coef_ : np.ndarray, shape (n_features,)
        Coefficients of the linear model
    intercept_ : float
        Intercept term (if fit_intercept=True)
    n_features_in_ : int
        Number of features seen during fit
    n_iter_ : int
        Number of iterations performed
    
    Examples
    --------
    >>> from math_toolkit.integration import BinaryLinearRegression
    >>> from sklearn.model_selection import cross_val_score
    >>> 
    >>> # Basic usage
    >>> model = BinaryLinearRegression()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> 
    >>> # Cross-validation
    >>> scores = cross_val_score(model, X, y, cv=5)
    >>> 
    >>> # Pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('regressor', BinaryLinearRegression())
    ... ])
    >>> pipe.fit(X_train, y_train)
    """
    
    def __init__(self, optimizer='binary', max_iter=100, tolerance=1e-6,
                 fit_intercept=True, verbose=False, optimizer_params=None):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
    
    def fit(self, X, y):
        """
        Fit linear regression model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Store number of features
        self.n_features_in_ = X.shape[1]
        
        # Add intercept column if needed
        if self.fit_intercept:
            X_design = add_dummy_feature(X)  # Adds column of 1s
        else:
            X_design = X
        
        # Initialize parameters
        n_params = X_design.shape[1]
        initial_theta = np.zeros(n_params)
        
        # Define cost and gradient functions (MSE)
        def cost_func(theta, X, y):
            predictions = X @ theta
            residuals = predictions - y
            return np.mean(residuals ** 2)
        
        def grad_func(theta, X, y):
            predictions = X @ theta
            residuals = predictions - y
            return (2 / len(y)) * (X.T @ residuals)
        
        # Create optimizer
        # Merge optimizer_params with our standard params (ours take precedence if conflict)
        opt_params = self.optimizer_params.copy()
        opt_params.update({
            'max_iter': self.max_iter,
            'tol': self.tolerance,
            'verbose': self.verbose
        })
        
        if self.optimizer == 'binary':
            opt = BinaryRateOptimizer(**opt_params)
        elif self.optimizer == 'adamw':
            opt = AdamW(**opt_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        # Optimize
        theta_opt = opt.optimize(X_design, y, initial_theta, cost_func, grad_func)
        
        # Extract coefficients and intercept
        if self.fit_intercept:
            self.coef_ = theta_opt[1:]  # Skip first (intercept)
            self.intercept_ = theta_opt[0]
        else:
            self.coef_ = theta_opt
            self.intercept_ = 0.0
        
        # Store iteration count
        self.n_iter_ = len(opt.history.get('cost', []))
        
        return self
    
    def predict(self, X):
        """
        Predict using linear model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
        
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted values
        """
        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X, accept_sparse=False)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        
        return X @ self.coef_ + self.intercept_


class BinaryLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic regression using BinaryRateOptimizer with binary search learning rate.
    
    Compatible with scikit-learn API (fit/predict/predict_proba).
    
    Parameters
    ----------
    optimizer : str, default='binary'
        Optimizer to use: 'binary' or 'adamw'
    max_iter : int, default=100
        Maximum number of iterations
    tolerance : float, default=1e-6
        Convergence tolerance
    fit_intercept : bool, default=True
        Whether to calculate intercept
    verbose : bool, default=False
        Print optimization progress
    optimizer_params : dict, optional
        Additional parameters for the optimizer
    
    Attributes
    ----------
    coef_ : np.ndarray, shape (n_features,)
        Coefficients of the logistic model
    intercept_ : float
        Intercept term
    classes_ : np.ndarray
        Classes seen during fit
    n_features_in_ : int
        Number of features
    n_iter_ : int
        Number of iterations
    
    Examples
    --------
    >>> from math_toolkit.integration import BinaryLogisticRegression
    >>> 
    >>> model = BinaryLogisticRegression()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> y_proba = model.predict_proba(X_test)
    """
    
    def __init__(self, optimizer='binary', max_iter=100, tolerance=1e-6,
                 fit_intercept=True, verbose=False, optimizer_params=None):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
    
    def fit(self, X, y):
        """
        Fit logistic regression model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (binary: 0 or 1)
        
        Returns
        -------
        self : object
            Fitted estimator
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Store classes and validate binary
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("BinaryLogisticRegression only supports binary classification")
        
        # Ensure y is 0/1
        y_binary = (y == self.classes_[1]).astype(float)
        
        self.n_features_in_ = X.shape[1]
        
        # Add intercept
        if self.fit_intercept:
            X_design = add_dummy_feature(X)
        else:
            X_design = X
        
        initial_theta = np.zeros(X_design.shape[1])
        
        # Logistic regression cost and gradient
        def sigmoid(z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Prevent overflow
        
        def cost_func(theta, X, y):
            predictions = sigmoid(X @ theta)
            epsilon = 1e-15  # Prevent log(0)
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        def grad_func(theta, X, y):
            predictions = sigmoid(X @ theta)
            return (1 / len(y)) * (X.T @ (predictions - y))
        
        # Create optimizer with bounds [−10, 10] to prevent overflow
        opt_params = self.optimizer_params.copy()
        if 'bounds' not in opt_params:
            opt_params['bounds'] = (-10, 10)
        
        if self.optimizer == 'binary':
            opt = BinaryRateOptimizer(
                max_iter=self.max_iter,
                tol=self.tolerance,
                verbose=self.verbose,
                **opt_params
            )
        elif self.optimizer == 'adamw':
            opt = AdamW(
                max_iter=self.max_iter,
                tol=self.tolerance,
                verbose=self.verbose,
                **opt_params
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        theta_opt = opt.optimize(X_design, y_binary, initial_theta, cost_func, grad_func)
        
        if self.fit_intercept:
            self.coef_ = theta_opt[1:]
            self.intercept_ = theta_opt[0]
        else:
            self.coef_ = theta_opt
            self.intercept_ = 0.0
        
        self.n_iter_ = len(opt.history.get('cost', []))
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples
        
        Returns
        -------
        proba : np.ndarray, shape (n_samples, 2)
            Class probabilities [P(y=0), P(y=1)]
        """
        check_is_fitted(self, ['coef_', 'intercept_', 'classes_'])
        X = check_array(X, accept_sparse=False)
        
        def sigmoid(z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        
        z = X @ self.coef_ + self.intercept_
        prob_class_1 = sigmoid(z)
        prob_class_0 = 1 - prob_class_1
        
        return np.column_stack([prob_class_0, prob_class_1])
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples
        
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]


class BinaryRidgeRegression(BaseEstimator, RegressorMixin):
    """
    Ridge regression (L2 regularization) using BinaryRateOptimizer.
    
    Adds L2 penalty to prevent overfitting: cost = MSE + alpha * ||theta||^2
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength (higher = more regularization)
    optimizer : str, default='binary'
        Optimizer to use
    max_iter : int, default=100
        Maximum iterations
    tolerance : float, default=1e-6
        Convergence tolerance
    fit_intercept : bool, default=True
        Whether to fit intercept
    verbose : bool, default=False
        Print progress
    optimizer_params : dict, optional
        Additional optimizer parameters
    
    Attributes
    ----------
    coef_ : np.ndarray
        Coefficients
    intercept_ : float
        Intercept
    n_features_in_ : int
        Number of features
    n_iter_ : int
        Iterations performed
    
    Examples
    --------
    >>> from math_toolkit.integration import BinaryRidgeRegression
    >>> 
    >>> # Stronger regularization for high-dimensional data
    >>> model = BinaryRidgeRegression(alpha=10.0)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    
    def __init__(self, alpha=1.0, optimizer='binary', max_iter=100,
                 tolerance=1e-6, fit_intercept=True, verbose=False,
                 optimizer_params=None):
        self.alpha = alpha
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
    
    def fit(self, X, y):
        """Fit ridge regression model."""
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        self.n_features_in_ = X.shape[1]
        
        if self.fit_intercept:
            X_design = add_dummy_feature(X)
        else:
            X_design = X
        
        initial_theta = np.zeros(X_design.shape[1])
        
        # Ridge cost and gradient (don't regularize intercept)
        def cost_func(theta, X, y):
            predictions = X @ theta
            residuals = predictions - y
            mse = np.mean(residuals ** 2)
            
            # L2 penalty (exclude intercept if fit_intercept=True)
            if self.fit_intercept:
                l2_penalty = self.alpha * np.sum(theta[1:] ** 2)
            else:
                l2_penalty = self.alpha * np.sum(theta ** 2)
            
            return mse + l2_penalty
        
        def grad_func(theta, X, y):
            predictions = X @ theta
            residuals = predictions - y
            grad = (2 / len(y)) * (X.T @ residuals)
            
            # Add L2 gradient (exclude intercept)
            if self.fit_intercept:
                l2_grad = np.zeros_like(theta)
                l2_grad[1:] = 2 * self.alpha * theta[1:]
                grad += l2_grad
            else:
                grad += 2 * self.alpha * theta
            
            return grad
        
        if self.optimizer == 'binary':
            opt = BinaryRateOptimizer(
                max_iter=self.max_iter,
                tol=self.tolerance,
                verbose=self.verbose,
                **self.optimizer_params
            )
        elif self.optimizer == 'adamw':
            opt = AdamW(
                max_iter=self.max_iter,
                tol=self.tolerance,
                verbose=self.verbose,
                **self.optimizer_params
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        theta_opt = opt.optimize(X_design, y, initial_theta, cost_func, grad_func)
        
        if self.fit_intercept:
            self.coef_ = theta_opt[1:]
            self.intercept_ = theta_opt[0]
        else:
            self.coef_ = theta_opt
            self.intercept_ = 0.0
        
        self.n_iter_ = len(opt.history.get('cost', []))
        
        return self
    
    def predict(self, X):
        """Predict using ridge regression model."""
        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X, accept_sparse=False)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        
        return X @ self.coef_ + self.intercept_
