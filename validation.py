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


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_benchmark: float = 0.0,
    n_trials: int = 1,
    n_observations: int = 252,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    verbose: bool = True
) -> dict:
    """
    Calculate Deflated Sharpe Ratio (DSR) using Bailey & López de Prado (2014) methodology.
    
    DSR adjusts the Sharpe Ratio for the effects of multiple testing by estimating 
    the probability that the observed SR is not just the maximum of random trials.
    
    The key insight: When you try N strategies and pick the best one, the "best" 
    Sharpe Ratio is biased upward. DSR corrects for this selection bias.
    
    Formula:
    1. Expected maximum SR from N trials: E[max(SR)] = SR_benchmark + σ_SR * E[Z_max]
       where E[Z_max] ≈ (1 - γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N*e))
       and γ ≈ 0.5772 (Euler-Mascheroni constant)
    
    2. Standard deviation of SR: σ_SR = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt-3)/4*SR^2) / (n-1))
    
    3. DSR = Φ((SR_observed - E[max(SR)]) / σ_SR)
    
    Args:
        observed_sr: The observed Sharpe Ratio from backtest
        sr_benchmark: Benchmark SR (typically 0 for null hypothesis)
        n_trials: Number of independent strategies/parameter sets tested
        n_observations: Number of return observations (e.g., 252 for daily over 1 year)
        skewness: Skewness of returns (0 for normal)
        kurtosis: Kurtosis of returns (3 for normal)
        verbose: Print detailed output
        
    Returns:
        Dict with DSR probability and adjusted metrics
    """
    from scipy.stats import norm
    
    # Euler-Mascheroni constant
    gamma = 0.5772156649
    
    # Standard deviation of SR estimator (Mertens 2002)
    # Accounts for non-normality of returns
    sr_std = np.sqrt(
        (1 + 0.5 * observed_sr**2 - skewness * observed_sr + 
         (kurtosis - 3) / 4 * observed_sr**2) / (n_observations - 1)
    )
    
    # Expected maximum SR from N trials (Extreme Value Theory)
    if n_trials > 1:
        # Approximate E[max(Z)] for N standard normals
        # Using Gumbel approximation
        z_max_approx = (
            (1 - gamma) * norm.ppf(1 - 1 / n_trials) + 
            gamma * norm.ppf(1 - 1 / (n_trials * np.e))
        )
        expected_max_sr = sr_benchmark + sr_std * z_max_approx
    else:
        expected_max_sr = sr_benchmark
    
    # Deflated probability: P(observed SR is genuine, not from selection bias)
    # This is the probability that a genuinely skillful strategy would achieve this SR
    if sr_std > 0:
        z_score = (observed_sr - expected_max_sr) / sr_std
        dsr_probability = norm.cdf(z_score)
    else:
        dsr_probability = 0.5
    
    # "Haircut" SR: what SR would be expected from random selection
    haircut_sr = observed_sr - expected_max_sr
    
    result = {
        'observed_sr': observed_sr,
        'expected_max_sr': expected_max_sr,
        'sr_std': sr_std,
        'dsr_probability': dsr_probability,
        'haircut_sr': haircut_sr,
        'n_trials': n_trials,
        'n_observations': n_observations
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("DEFLATED SHARPE RATIO (DSR) ANALYSIS")
        print("Bailey & Lopez de Prado (2014) Multiple Testing Correction")
        print("=" * 60)
        print(f"\nInput Parameters:")
        print(f"  Observed Sharpe Ratio:    {observed_sr:>10.4f}")
        print(f"  Number of Trials (N):     {n_trials:>10d}")
        print(f"  Number of Observations:   {n_observations:>10d}")
        print(f"  SR Benchmark (H0):        {sr_benchmark:>10.4f}")
        
        print(f"\nDSR Calculation:")
        print(f"  SR Standard Deviation:    {sr_std:>10.4f}")
        print(f"  Expected Max SR (bias):   {expected_max_sr:>10.4f}")
        print(f"  Haircut SR:               {haircut_sr:>10.4f}")
        
        print(f"\nResult:")
        print(f"  DSR Probability:          {dsr_probability:>10.2%}")
        
        if dsr_probability > 0.95:
            print("  Interpretation: [STRONG] - Highly likely to be genuine skill")
        elif dsr_probability > 0.80:
            print("  Interpretation: [MODERATE] - Likely genuine, but some selection bias possible")
        elif dsr_probability > 0.50:
            print("  Interpretation: [WEAK] - May be partially due to selection bias")
        else:
            print("  Interpretation: [FAIL] - Likely due to selection bias / overfitting")
        
        print("=" * 60)
    
    return result


def calculate_meta_labeler_performance(fold_metrics: list, verbose: bool = True) -> dict:
    """
    Analyze Meta-Labeler performance across folds.
    
    The Meta-Labeler's job is to predict whether the base model's trade will be correct.
    Good meta-accuracy (significantly above 50%) indicates it identifies trade quality.
    
    Args:
        fold_metrics: List of fold metric dicts from nested_walk_forward_validation
        verbose: Print detailed output
        
    Returns:
        Dict with aggregated meta-labeler statistics
    """
    folds_with_meta = [f for f in fold_metrics if f.get('meta_available', False)]
    
    if not folds_with_meta:
        return {'error': 'No meta-labeler results available'}
    
    base_accuracies = [f['base_accuracy'] for f in fold_metrics]
    meta_accuracies = [f.get('meta_accuracy', 0.5) for f in folds_with_meta if 'meta_accuracy' in f or hasattr(f, 'meta_accuracy')]
    
    result = {
        'n_folds': len(fold_metrics),
        'n_folds_with_meta': len(folds_with_meta),
        'mean_base_accuracy': np.mean(base_accuracies),
        'std_base_accuracy': np.std(base_accuracies),
        'mean_meta_accuracy': np.mean(meta_accuracies) if meta_accuracies else 0.5,
        'std_meta_accuracy': np.std(meta_accuracies) if meta_accuracies else 0.0,
        'meta_above_random': np.mean(meta_accuracies) > 0.5 if meta_accuracies else False,
        'fold_details': fold_metrics
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("META-LABELER PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        print(f"\nFold-by-Fold Results:")
        print("-" * 50)
        for fold in fold_metrics:
            meta_str = f"{fold.get('meta_accuracy', 'N/A'):.4f}" if 'meta_accuracy' in fold else "N/A"
            print(f"  Fold {fold['fold']}: Base Acc = {fold['base_accuracy']:.4f}, "
                  f"Meta Acc = {meta_str}")
        
        print(f"\nAggregated Statistics:")
        print(f"  Mean Base Accuracy:  {result['mean_base_accuracy']:.4f} (+/- {result['std_base_accuracy']:.4f})")
        print(f"  Mean Meta Accuracy:  {result['mean_meta_accuracy']:.4f} (+/- {result['std_meta_accuracy']:.4f})")
        
        print(f"\nMeta-Labeler Evaluation:")
        meta_edge = result['mean_meta_accuracy'] - 0.5
        if meta_edge > 0.15:
            print(f"  [STRONG] Meta-Labeler has significant edge ({meta_edge:.1%} above random)")
            print("  -> Can effectively filter out bad trades")
        elif meta_edge > 0.05:
            print(f"  [MODERATE] Meta-Labeler has some edge ({meta_edge:.1%} above random)")
            print("  -> Provides value but with noise")
        elif meta_edge > 0:
            print(f"  [WEAK] Meta-Labeler has minimal edge ({meta_edge:.1%} above random)")
            print("  -> Limited ability to identify trade failures")
        else:
            print(f"  [FAIL] Meta-Labeler performs at or below random ({meta_edge:.1%})")
            print("  -> Not adding value to position sizing")
        
        print("=" * 60)
    
    return result

