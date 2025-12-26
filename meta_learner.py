"""
Bayesian meta-learner for GME trading strategy.
Implements Bayesian Ridge Regression stacking layer with uncertainty quantification.
"""

import numpy as np
from sklearn.linear_model import BayesianRidge
from typing import Tuple
import config


class BayesianMetaLearner:
    """
    Bayesian Ridge Regression stacking layer.
    
    Takes predictions from base models as inputs and outputs:
    1. Final price prediction for t+1
    2. Predictive variance (confidence measure)
    
    The Bayesian approach provides:
    - Automatic regularization via prior
    - Uncertainty quantification for confidence-based trading
    - Robustness to base model overfitting
    """
    
    def __init__(
        self,
        max_iter: int = 300,
        tol: float = 1e-3,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        compute_score: bool = True
    ):
        """
        Initialize Bayesian meta-learner.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            alpha_1, alpha_2: Hyperparameters for Gamma prior over alpha (noise precision)
            lambda_1, lambda_2: Hyperparameters for Gamma prior over lambda (weight precision)
            compute_score: Whether to compute log marginal likelihood
        """
        self.model = BayesianRidge(
            max_iter=max_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            compute_score=compute_score
        )
        self._is_fitted = False
    
    def fit(self, base_predictions: np.ndarray, y: np.ndarray) -> 'BayesianMetaLearner':
        """
        Fit the meta-learner on base model predictions.
        
        Args:
            base_predictions: Predictions from base models, shape (n_samples, n_models)
            y: True target values, shape (n_samples,)
            
        Returns:
            self
        """
        self.model.fit(base_predictions, y)
        self._is_fitted = True
        return self
    
    def predict(self, base_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate final predictions with uncertainty.
        
        Args:
            base_predictions: Predictions from base models, shape (n_samples, n_models)
            
        Returns:
            Tuple of:
            - predictions: Final predictions, shape (n_samples,)
            - variances: Predictive variances, shape (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Meta-learner must be fitted before predicting")
        
        predictions, std = self.model.predict(base_predictions, return_std=True)
        variances = std ** 2
        
        return predictions, variances
    
    def get_confidence(self, variances: np.ndarray) -> np.ndarray:
        """
        Convert predictive variance to confidence score.
        
        Confidence is defined as the reciprocal of variance:
        confidence = 1 / variance
        
        Higher confidence means lower uncertainty in the prediction.
        
        Args:
            variances: Predictive variances, shape (n_samples,)
            
        Returns:
            Confidence scores, shape (n_samples,)
        """
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        return 1.0 / (variances + eps)
    
    def get_model_weights(self) -> np.ndarray:
        """
        Get the learned weights for each base model.
        
        Returns:
            Weight coefficients, shape (n_models,)
        """
        if not self._is_fitted:
            raise RuntimeError("Meta-learner must be fitted before getting weights")
        
        return self.model.coef_
    
    def get_noise_precision(self) -> float:
        """
        Get the estimated noise precision (alpha).
        
        Returns:
            Noise precision value
        """
        if not self._is_fitted:
            raise RuntimeError("Meta-learner must be fitted")
        
        return self.model.alpha_
    
    def get_weight_precision(self) -> float:
        """
        Get the estimated weight precision (lambda).
        
        Returns:
            Weight precision value
        """
        if not self._is_fitted:
            raise RuntimeError("Meta-learner must be fitted")
        
        return self.model.lambda_


if __name__ == "__main__":
    # Test meta-learner
    np.random.seed(42)
    
    # Generate synthetic base model predictions
    n_samples = 500
    n_models = 3
    
    # True signal
    true_signal = np.sin(np.linspace(0, 4 * np.pi, n_samples)) * 0.02
    
    # Base model predictions with different biases/noise
    base_preds = np.column_stack([
        true_signal + np.random.randn(n_samples) * 0.005,  # Low noise
        true_signal * 0.8 + np.random.randn(n_samples) * 0.008,  # Biased
        true_signal + np.random.randn(n_samples) * 0.01,  # Higher noise
    ])
    
    y = true_signal + np.random.randn(n_samples) * 0.003
    
    # Train/test split
    train_size = int(0.8 * n_samples)
    base_train, base_test = base_preds[:train_size], base_preds[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print("Testing BayesianMetaLearner...")
    meta = BayesianMetaLearner()
    meta.fit(base_train, y_train)
    
    predictions, variances = meta.predict(base_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Variances shape: {variances.shape}")
    
    # Confidence scores
    confidence = meta.get_confidence(variances)
    print(f"\nConfidence stats:")
    print(f"  Min: {confidence.min():.2f}")
    print(f"  Max: {confidence.max():.2f}")
    print(f"  Mean: {confidence.mean():.2f}")
    
    # Model weights
    weights = meta.get_model_weights()
    print(f"\nBase model weights:")
    for i, w in enumerate(weights):
        print(f"  Model {i}: {w:.4f}")
    
    # MSE comparison
    from sklearn.metrics import mean_squared_error
    meta_mse = mean_squared_error(y_test, predictions)
    print(f"\nMeta-learner MSE: {meta_mse:.6f}")
    for i in range(n_models):
        base_mse = mean_squared_error(y_test, base_test[:, i])
        print(f"Base model {i} MSE: {base_mse:.6f}")
