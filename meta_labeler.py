"""
XGBoost Meta-Labeler for trade gating.

The meta-labeler decides whether to ACT on the base ensemble's signal.
This is fundamentally different from the base models:
- Base ensemble: What direction should we trade? (Long/Short/Flat)
- Meta-labeler: Should we trade at all? (Yes/No)

The meta-labeler learns to identify when the base model's predictions
are trustworthy based on market conditions (H, Vol, signature features).
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
import config

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not installed. Run: pip install xgboost")


class MetaLabeler:
    """
    XGBoost-based meta-labeler for trade gating.
    
    Features used:
    - Hurst exponent (regime indicator)
    - Rolling volatility (risk environment)
    - Signature features (market geometry)
    - Base model prediction confidence
    
    Target: Binary (1 = trade was correct, 0 = trade was wrong)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        scale_pos_weight: float = None,
        random_state: int = config.RANDOM_SEED
    ):
        """
        Initialize XGBoost meta-labeler.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (keep low to prevent overfit)
            learning_rate: Boosting learning rate
            scale_pos_weight: Weight for positive class (handles imbalance)
            random_state: Random seed
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': random_state,
            'n_jobs': -1
        }
        
        if scale_pos_weight is not None:
            self.params['scale_pos_weight'] = scale_pos_weight
        
        self.model = xgb.XGBClassifier(**self.params)
        self._is_fitted = False
        self.feature_names = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
        auto_balance: bool = True
    ) -> 'MetaLabeler':
        """
        Fit the meta-labeler on historical trades.
        
        Args:
            X: Feature matrix (n_samples, n_features)
               Features should include: [H, volatility, signature_features...]
            y: Binary labels (1 = trade correct, 0 = trade wrong)
            feature_names: Optional list of feature names for importance
            auto_balance: Whether to auto-compute class weight for imbalance
            
        Returns:
            self
        """
        if auto_balance and 'scale_pos_weight' not in self.params:
            # Compute class imbalance ratio
            n_neg = np.sum(y == 0)
            n_pos = np.sum(y == 1)
            if n_pos > 0:
                scale = n_neg / n_pos
                self.model.set_params(scale_pos_weight=scale)
        
        self.model.fit(X, y)
        self.feature_names = feature_names
        self._is_fitted = True
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of trade being correct.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability of positive class (trade correct)
        """
        if not self._is_fitted:
            raise RuntimeError("Meta-labeler must be fitted before predicting")
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict whether to take the trade.
        
        Args:
            X: Feature matrix
            threshold: Probability threshold for acting
            
        Returns:
            Binary predictions (1 = take trade, 0 = skip)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.
        
        Returns:
            Series with feature names and importance scores
        """
        if not self._is_fitted:
            raise RuntimeError("Meta-labeler must be fitted")
        
        importance = self.model.feature_importances_
        
        if self.feature_names is not None:
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        else:
            return pd.Series(importance).sort_values(ascending=False)


def create_meta_features(
    base_predictions: np.ndarray,
    hurst: np.ndarray,
    volatility: np.ndarray,
    signature_features: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create feature matrix for meta-labeler.
    
    Args:
        base_predictions: Base ensemble predictions (n_samples, n_classes)
        hurst: Hurst exponent values (n_samples,)
        volatility: Rolling volatility (n_samples,)
        signature_features: Optional signature features (n_samples, n_sig_features)
        
    Returns:
        Combined feature matrix
    """
    # Ensure 2D
    if base_predictions.ndim == 1:
        base_predictions = base_predictions.reshape(-1, 1)
    
    hurst = np.array(hurst).reshape(-1, 1)
    volatility = np.array(volatility).reshape(-1, 1)
    
    features = [base_predictions, hurst, volatility]
    
    if signature_features is not None:
        features.append(signature_features)
    
    return np.hstack(features)


def compute_meta_targets(
    predictions: np.ndarray,
    labels: np.ndarray,
    directional_only: bool = True
) -> np.ndarray:
    """
    Compute binary meta-labels: Was the base prediction correct?
    
    For classification:
    - Correct if predicted direction matches actual direction
    - If directional_only=True, only considers non-flat predictions
    
    Args:
        predictions: Base model predictions (class labels)
        labels: True triple barrier labels
        directional_only: Whether to only evaluate directional trades
        
    Returns:
        Binary array (1 = correct, 0 = wrong)
    """
    if directional_only:
        # Only evaluate when prediction was directional
        mask = (predictions != 0)
        correct = (predictions == labels) & mask
    else:
        correct = (predictions == labels)
    
    return correct.astype(int)


if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing Meta-Labeler")
    print("=" * 50)
    
    # Generate synthetic data
    n_samples = 1000
    n_sig_features = 10
    
    # Simulated base predictions
    base_preds = np.random.choice([-1, 0, 1], size=n_samples, p=[0.3, 0.2, 0.5])
    
    # Market conditions
    hurst = np.random.uniform(0.3, 0.7, n_samples)
    volatility = np.random.uniform(0.01, 0.05, n_samples)
    sig_features = np.random.randn(n_samples, n_sig_features)
    
    # True labels with some noise
    # Predictions are more likely correct when H is high
    prob_correct = 0.3 + 0.4 * (hurst - 0.3) / 0.4
    labels = (np.random.random(n_samples) < prob_correct).astype(int)
    
    # Create features
    X = create_meta_features(base_preds.reshape(-1, 1), hurst, volatility, sig_features)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Positive class ratio: {labels.mean():.2%}")
    
    # Train/test split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = labels[:split], labels[split:]
    
    # Fit meta-labeler
    meta = MetaLabeler(n_estimators=50, max_depth=3)
    meta.fit(X_train, y_train)
    
    # Evaluate
    proba = meta.predict_proba(X_test)
    preds = meta.predict(X_test, threshold=0.5)
    
    accuracy = (preds == y_test).mean()
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # Feature importance
    print("\nTop 5 Feature Importances:")
    importance = meta.get_feature_importance()
    for i, (idx, imp) in enumerate(importance.head().items()):
        print(f"  Feature {idx}: {imp:.4f}")
