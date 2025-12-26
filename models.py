"""
Base model ensemble for GME trading strategy.
Implements Logistic Regression, LightGBM, and Extra Trees CLASSIFIERS.

Changed from regressors to classifiers to support Triple Barrier labeling:
- Class 1: Long (upper barrier hit first)
- Class -1: Short (lower barrier hit first)  
- Class 0: Flat (time barrier hit first)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from typing import List, Optional, Tuple
import config


class BaseEnsemble:
    """
    Three-component classifier ensemble for signature-based trading.
    
    Models:
    1. Logistic Regression: Captures linear separability in signature space
    2. LightGBM: Identifies non-linear decision boundaries
    3. Extra Trees: Provides robust, low-variance baseline
    
    All models output class probabilities for Long/Short/Flat.
    """
    
    def __init__(
        self,
        logistic_C: float = 1.0,
        lgb_params: dict = None,
        et_params: dict = None,
        random_state: int = config.RANDOM_SEED
    ):
        """
        Initialize the classifier ensemble.
        
        Args:
            logistic_C: Regularization strength for LogisticRegression
            lgb_params: LightGBM parameters (uses config defaults if None)
            et_params: Extra Trees parameters (uses config defaults if None)
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.classes_ = np.array([-1, 0, 1])  # Short, Flat, Long
        
        # Logistic Regression - linear baseline
        # Note: multi_class parameter removed in sklearn 1.2+ (auto-inferred)
        self.logistic = LogisticRegression(
            C=logistic_C,
            solver='lbfgs',
            max_iter=500,
            random_state=random_state
        )
        
        # LightGBM Classifier - gradient boosting
        lgb_params = lgb_params or {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_jobs': -1,
            'verbosity': -1
        }
        lgb_params['random_state'] = random_state
        lgb_params['objective'] = 'multiclass'
        lgb_params['num_class'] = 3
        self.lgb = lgb.LGBMClassifier(**lgb_params)
        
        # Extra Trees Classifier - bagging for robustness
        et_params = et_params or {
            'n_estimators': 100,
            'max_depth': 10,
            'n_jobs': -1
        }
        et_params['random_state'] = random_state
        self.extra_trees = ExtraTreesClassifier(**et_params)
        
        self.models: List = [self.logistic, self.lgb, self.extra_trees]
        self.model_names = ['logistic', 'lightgbm', 'extra_trees']
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseEnsemble':
        """
        Fit all base models on the training data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target class labels of shape (n_samples,)
               Labels should be -1 (Short), 0 (Flat), or 1 (Long)
            
        Returns:
            self
        """
        # Store unique classes
        self.classes_ = np.unique(y)
        
        for model in self.models:
            model.fit(X, y)
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate class predictions from all base models.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples, 3) - one column per model
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before predicting")
        
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate class probability predictions from all base models.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Probabilities of shape (n_samples, n_models, n_classes)
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before predicting")
        
        # Stack probabilities from each model
        probas = [model.predict_proba(X) for model in self.models]
        
        return np.stack(probas, axis=1)  # (n_samples, n_models, n_classes)
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble prediction via soft voting.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, confidence)
            - predictions: Class labels (n_samples,)
            - confidence: Max probability (n_samples,)
        """
        # Average probabilities across models
        probas = self.predict_proba(X)  # (n_samples, n_models, n_classes)
        avg_proba = probas.mean(axis=1)  # (n_samples, n_classes)
        
        # Get predicted class and confidence
        predictions = self.classes_[np.argmax(avg_proba, axis=1)]
        confidence = np.max(avg_proba, axis=1)
        
        return predictions, confidence
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> dict:
        """
        Get feature importance from tree-based models.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with model names as keys and importance arrays as values
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before getting importance")
        
        importance = {}
        
        # LightGBM importance
        lgb_imp = self.lgb.feature_importances_
        importance['lightgbm'] = dict(zip(
            feature_names or range(len(lgb_imp)),
            lgb_imp
        ))
        
        # Extra Trees importance
        et_imp = self.extra_trees.feature_importances_
        importance['extra_trees'] = dict(zip(
            feature_names or range(len(et_imp)),
            et_imp
        ))
        
        # Logistic coefficients (mean absolute value across classes as importance proxy)
        log_coef = np.mean(np.abs(self.logistic.coef_), axis=0)
        importance['logistic'] = dict(zip(
            feature_names or range(len(log_coef)),
            log_coef
        ))
        
        return importance


if __name__ == "__main__":
    # Test base classifier ensemble
    np.random.seed(42)
    
    # Generate synthetic classification data
    n_samples = 500
    n_features = 30  # Signature dimensions
    
    X = np.random.randn(n_samples, n_features)
    
    # Create labels with some structure
    scores = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.5
    y = np.where(scores > 0.3, 1, np.where(scores < -0.3, -1, 0))
    
    # Train/test split
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print("Testing BaseEnsemble (Classifiers)")
    print("=" * 50)
    print(f"\nLabel distribution in training:")
    for label in [-1, 0, 1]:
        print(f"  Class {label}: {(y_train == label).sum()}")
    
    ensemble = BaseEnsemble()
    ensemble.fit(X_train, y_train)
    
    predictions = ensemble.predict(X_test)
    print(f"\nIndividual predictions shape: {predictions.shape}")
    
    # Ensemble prediction
    ensemble_pred, confidence = ensemble.predict_ensemble(X_test)
    print(f"Ensemble predictions shape: {ensemble_pred.shape}")
    print(f"Mean confidence: {confidence.mean():.3f}")
    
    # Accuracy
    from sklearn.metrics import accuracy_score, classification_report
    
    for i, name in enumerate(ensemble.model_names):
        acc = accuracy_score(y_test, predictions[:, i])
        print(f"{name} accuracy: {acc:.3f}")
    
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble accuracy: {ensemble_acc:.3f}")
    
    print("\nClassification Report (Ensemble):")
    print(classification_report(y_test, ensemble_pred, target_names=['Short', 'Flat', 'Long']))
