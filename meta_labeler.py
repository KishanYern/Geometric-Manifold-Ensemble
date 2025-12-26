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
from typing import Optional, Tuple, List
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
        auto_balance: bool = True,
        early_stopping_rounds: Optional[int] = 10,
        eval_set: Optional[list] = None,
        validation_split: Optional[float] = 0.1
    ) -> 'MetaLabeler':
        """
        Fit the meta-labeler on historical trades with early stopping.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (1 = trade correct, 0 = trade wrong)
            feature_names: Optional list of feature names for importance
            auto_balance: Whether to auto-compute class weight for imbalance
            early_stopping_rounds: Rounds without improvement to stop
            eval_set: Explicit validation set [(X_val, y_val)]
            validation_split: If eval_set is None, fraction of X to use for validation (time-series split)
            
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
        
        # Prepare evaluation set for early stopping
        if early_stopping_rounds is not None:
            if eval_set is None and validation_split is not None:
                # Create time-series split for validation
                # Always split strictly chronologically
                split_idx = int(len(X) * (1 - validation_split))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                eval_set = [(X_val, y_val)]
                X_fit, y_fit = X_train, y_train
            else:
                X_fit, y_fit = X, y
            
            self.model.fit(
                X_fit, y_fit,
                eval_set=eval_set,
                verbose=False,
                early_stopping_rounds=early_stopping_rounds
            )
        else:
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
        base_predictions: Base ensemble predictions (n_samples, n_classes) or probabilities
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
