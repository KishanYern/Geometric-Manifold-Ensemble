"""
Validation framework for GME trading strategy.
Implements Purged Walk-Forward Cross-Validation and Nested CV for Meta-Labeling.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List, Optional, Dict, Any
from scipy.stats import norm
from sklearn.metrics import accuracy_score, classification_report
import config
from features import SignaturePreprocessor
from meta_labeler import create_meta_features, compute_meta_targets, MetaLabeler


class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation.
    """
    
    def __init__(
        self,
        n_splits: int = config.N_SPLITS,
        purge_window: int = config.PURGE_WINDOW,
        expanding: bool = True,
        min_train_size: Optional[int] = None
    ):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.expanding = expanding
        self.min_train_size = min_train_size
    
    def split(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        
        if self.min_train_size is None:
            min_train = max(50, n_samples // 5)
        else:
            min_train = self.min_train_size
        
        available = n_samples - min_train - self.purge_window * self.n_splits
        test_size = max(20, available // self.n_splits)
        
        for split_idx in range(self.n_splits):
            if self.expanding:
                train_end = min_train + split_idx * (test_size + self.purge_window)
                train_start = 0
            else:
                train_start = split_idx * (test_size + self.purge_window)
                train_end = train_start + min_train
            
            test_start = train_end + self.purge_window
            test_end = min(test_start + test_size, n_samples)
            
            if test_end <= test_start:
                continue
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def validate_no_leakage(train_indices, test_indices, purge_window):
    if len(train_indices) == 0 or len(test_indices) == 0:
        return True
    return test_indices.min() - train_indices.max() - 1 >= purge_window


def nested_walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    side_info: Dict[str, np.ndarray],
    model_factory: Any,
    cv: PurgedWalkForwardCV,
    meta_labeler_params: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Nested Purged Walk-Forward CV.
    
    Args:
        X: Raw features (n_samples, n_features)
        y: Targets
        side_info: Dict with 'hurst', 'volatility' arrays
        model_factory: Callable returning BaseEnsemble instance
        cv: Cross-validator for outer loop
        meta_labeler_params: Params for XGBoost
        verbose: Print progress
        
    Returns:
        Dict with aggregated predictions, meta-predictions, and metrics.
    """
    
    all_base_preds = []
    all_meta_probs = []
    all_targets = []
    all_test_indices = []
    
    fold_metrics = []
    
    # Preprocessor factory
    def get_preprocessor():
        return SignaturePreprocessor(variance_threshold=0.95)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        if verbose:
            print(f"\nOuter Fold {fold_idx + 1}/{cv.n_splits}")
            print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")
            
        # 1. Inner Loop: Generate Meta-Training Data
        X_train_outer = X[train_idx]
        y_train_outer = y[train_idx]
        
        # Create inner CV
        inner_cv = PurgedWalkForwardCV(n_splits=3, purge_window=cv.purge_window, min_train_size=100)
        
        inner_meta_features = []
        inner_meta_targets = []
        
        if len(train_idx) < 200:
             if verbose: print("  Warning: Train set too small for nested CV. Skipping meta-training.")
             has_meta = False
        else:
             has_meta = True
             if verbose: print("  Running Inner CV for Meta-Labeler...")
             
             for sub_train, sub_test in inner_cv.split(X_train_outer):
                 global_sub_test = train_idx[sub_test]
                 
                 # Preprocess
                 prep_inner = get_preprocessor()
                 X_sub_tr = prep_inner.fit_transform(X_train_outer[sub_train])
                 X_sub_te = prep_inner.transform(X_train_outer[sub_test])
                 
                 # Instantiate and Fit Base
                 base_ensemble = model_factory()
                 base_ensemble.fit(X_sub_tr, y_train_outer[sub_train])
                 
                 # Predict OOS (Ensemble)
                 sub_preds, _ = base_ensemble.predict_ensemble(X_sub_te)
                 
                 # Meta Targets
                 correctness = compute_meta_targets(sub_preds, y_train_outer[sub_test])
                 
                 # Meta Features
                 current_hurst = side_info['hurst'][global_sub_test]
                 current_vol = side_info['volatility'][global_sub_test]
                 
                 m_feats = create_meta_features(
                     sub_preds, 
                     current_hurst, 
                     current_vol, 
                     X_train_outer[sub_test][:, :10] 
                 )
                 
                 inner_meta_features.append(m_feats)
                 inner_meta_targets.append(correctness)
                 
        # 2. Train Meta-Model
        meta_learner = None
        if has_meta and inner_meta_features:
            meta_X = np.vstack(inner_meta_features)
            meta_y = np.concatenate(inner_meta_targets)
            
            meta_learner = MetaLabeler(**meta_labeler_params)
            
            if len(meta_X) > 50:
                meta_learner.fit(
                    meta_X, meta_y, 
                    early_stopping_rounds=10,
                    validation_split=0.2
                )
            else:
                meta_learner.fit(meta_X, meta_y)
            
        # 3. Outer Loop Execution
        prep_outer = get_preprocessor()
        X_train_trans = prep_outer.fit_transform(X_train_outer)
        X_test_trans = prep_outer.transform(X[test_idx])
        
        # Instantiate and Fit Base
        base_ensemble = model_factory()
        base_ensemble.fit(X_train_trans, y_train_outer)
        
        # Predict Test
        base_preds, _ = base_ensemble.predict_ensemble(X_test_trans)
        
        # Meta Predict
        meta_probs = np.zeros(len(base_preds))
        if meta_learner:
            test_hurst = side_info['hurst'][test_idx]
            test_vol = side_info['volatility'][test_idx]
            
            meta_feats_test = create_meta_features(
                base_preds,
                test_hurst,
                test_vol,
                X[test_idx][:, :10]
            )
            meta_probs = meta_learner.predict_proba(meta_feats_test)
        else:
            meta_probs = np.ones(len(base_preds)) * 0.5
            
        # Store
        all_base_preds.extend(base_preds)
        all_meta_probs.extend(meta_probs)
        all_targets.extend(y[test_idx])
        all_test_indices.extend(test_idx)
        
        acc = accuracy_score(y[test_idx], base_preds)
        fold_metrics.append({'fold': fold_idx+1, 'base_accuracy': acc, 'meta_available': has_meta})
        
        if verbose:
            print(f"  Base Accuracy: {acc:.4f}")
            if has_meta:
                meta_threshold = 0.5
                is_correct = (base_preds == y[test_idx]).astype(int)
                meta_decisions = (meta_probs > meta_threshold).astype(int)
                meta_acc = accuracy_score(is_correct, meta_decisions)
                print(f"  Meta Accuracy: {meta_acc:.4f}")

    return {
        'predictions': np.array(all_base_preds),
        'meta_probabilities': np.array(all_meta_probs),
        'targets': np.array(all_targets),
        'test_indices': np.array(all_test_indices),
        'fold_metrics': fold_metrics
    }
