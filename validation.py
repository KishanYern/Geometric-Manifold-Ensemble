"""
Validation framework for GME trading strategy.
Implements Purged Walk-Forward Cross-Validation for classification.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List, Optional
from scipy.stats import norm
from itertools import combinations
import scipy.special as sc
from sklearn.metrics import accuracy_score, classification_report
import config


class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation for time series classification.
    
    Key features:
    1. Strictly chronological train/test splits
    2. Purge window to eliminate overlap leakage
    3. Expanding or sliding window options
    """
    
    def __init__(
        self,
        n_splits: int = config.N_SPLITS,
        purge_window: int = config.PURGE_WINDOW,
        expanding: bool = True,
        min_train_size: Optional[int] = None
    ):
        """
        Initialize walk-forward cross-validator.
        
        Args:
            n_splits: Number of train/test splits
            purge_window: Number of samples to purge between train and test
            expanding: If True, training set grows. If False, sliding window.
            min_train_size: Minimum training set size (default: 20% of data)
        """
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
        """
        Generate train/test indices for each fold.
        
        The data is split as follows:
        - Fold 1: Train [0:t1], Purge [t1:t1+p], Test [t1+p:t2]
        - Fold 2: Train [0:t2] (or [t1:t2]), Purge [t2:t2+p], Test [t2+p:t3]
        - ...
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Calculate minimum training size
        if self.min_train_size is None:
            min_train = max(50, n_samples // 5)
        else:
            min_train = self.min_train_size
        
        # Calculate test size (roughly equal for each split)
        available = n_samples - min_train - self.purge_window * self.n_splits
        test_size = max(20, available // self.n_splits)
        
        for split_idx in range(self.n_splits):
            if self.expanding:
                # Expanding window: train set grows with each fold
                train_end = min_train + split_idx * (test_size + self.purge_window)
            else:
                # Sliding window: fixed train size
                train_start = split_idx * (test_size + self.purge_window)
                train_end = train_start + min_train
            
            # Test indices (after purge)
            test_start = train_end + self.purge_window
            test_end = min(test_start + test_size, n_samples)
            
            if test_end <= test_start:
                continue
            
            if self.expanding:
                train_indices = np.arange(0, train_end)
            else:
                train_indices = np.arange(train_start, train_end)
            
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def get_n_splits(
        self, 
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return the number of splits."""
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    As described in Advances in Financial Machine Learning (Ch. 12).
    Generates N choose k splits, where:
    - N is the number of groups
    - k is the number of groups in the test set
    """
    
    def __init__(
        self,
        n_splits: int = config.N_SPLITS,
        n_test_splits: int = 2,
        purge_window: int = config.PURGE_WINDOW,
        embargo_pct: float = 0.01,
        samples_info_sets: Optional[pd.Series] = None
    ):
        """
        Initialize CPCV.
        
        Args:
            n_splits (N): Number of total groups
            n_test_splits (k): Number of groups in test set
            purge_window: Number of samples to exclude around test set
            embargo_pct: Percentage of samples to embargo after test set
            samples_info_sets: Series with index=evaluation_time, value=t1 (optional)
                             Used for more precise purging
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct
        self.samples_info_sets = samples_info_sets
        
    def split(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate combinatorial train/test splits with purging and embargoing.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        # 1. Split indices into N groups
        # Using simple time-based chunks for now
        group_size = n_samples // self.n_splits
        group_indices = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            group_indices.append(indices[start:end])
            
        # 2. Generate combinations of k test groups
        for test_groups_idx in combinations(range(self.n_splits), self.n_test_splits):
            # Form test set
            test_indices = np.concatenate([group_indices[i] for i in test_groups_idx])
            test_indices.sort()
            
            # Form initial train set (complement of test chunks)
            train_groups_idx = [i for i in range(self.n_splits) if i not in test_groups_idx]
            if not train_groups_idx:
                continue
            
            train_indices = np.concatenate([group_indices[i] for i in train_groups_idx])
            train_indices.sort()
            
            # 3. Apply Purging
            # Remove train samples that overlap with test samples
            # Simple implementation: remove 'purge_window' before and after test blocks
            # For more robust purging, we would need t1 info (event end times)
            
            # Find boundaries of test blocks
            test_blocks = []
            if len(test_indices) > 0:
                diffs = np.diff(test_indices)
                breaks = np.where(diffs > 1)[0]
                starts = [test_indices[0]] + [test_indices[i+1] for i in breaks]
                ends = [test_indices[i] for i in breaks] + [test_indices[-1]]
                test_blocks = list(zip(starts, ends))
            
            purge_mask = np.zeros(n_samples, dtype=bool)
            
            # Embargo calculation
            embargo = int(n_samples * self.embargo_pct)
            
            for start, end in test_blocks:
                # Purge before
                p_start = max(0, start - self.purge_window)
                purge_mask[p_start:start] = True
                
                # Purge after (includes embargo)
                p_end = min(n_samples, end + 1 + self.purge_window + embargo)
                purge_mask[end+1:p_end] = True
            
            # Remove purged samples from train_indices
            valid_train = ~purge_mask[train_indices]
            train_indices = train_indices[valid_train]
            
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of combinatorial splits: N choose k"""
        return int(sc.comb(self.n_splits, self.n_test_splits))


def validate_no_leakage(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    purge_window: int
) -> bool:
    """
    Validate that there is no information leakage between train and test sets.
    
    Returns:
        True if no leakage, False otherwise
    """
    if len(train_indices) == 0 or len(test_indices) == 0:
        return True
    
    train_max = train_indices.max()
    test_min = test_indices.min()
    gap = test_min - train_max - 1
    
    return gap >= purge_window


def walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    base_ensemble,
    cv: Optional[PurgedWalkForwardCV] = None,
    verbose: bool = True
) -> dict:
    """
    Perform walk-forward validation on the classifier ensemble.
    
    For each fold:
    1. Train base ensemble on training data
    2. Predict on test data
    3. Collect predictions for meta-labeling
    
    Args:
        X: Feature matrix
        y: Class labels
        base_ensemble: BaseEnsemble classifier instance
        cv: Cross-validator instance
        verbose: Whether to print progress
        
    Returns:
        Dictionary with validation results
    """
    if cv is None:
        cv = PurgedWalkForwardCV()
    
    all_predictions = []
    all_probas = []
    all_targets = []
    all_test_indices = []
    all_confidences = []
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        if verbose:
            print(f"\nFold {fold_idx + 1}/{cv.n_splits}")
            print(f"  Train: {len(train_idx)} samples [{train_idx[0]}-{train_idx[-1]}]")
            print(f"  Test:  {len(test_idx)} samples [{test_idx[0]}-{test_idx[-1]}]")
        
        # Verify no leakage
        assert validate_no_leakage(train_idx, test_idx, cv.purge_window), \
            "Information leakage detected!"
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit base ensemble
        base_ensemble.fit(X_train, y_train)
        
        # Get predictions and probabilities
        preds, confidence = base_ensemble.predict_ensemble(X_test)
        probas = base_ensemble.predict_proba(X_test)
        
        # Collect results
        all_predictions.extend(preds)
        all_probas.extend(probas)
        all_targets.extend(y_test)
        all_test_indices.extend(test_idx)
        all_confidences.extend(confidence)
        
        # Calculate fold accuracy
        fold_acc = accuracy_score(y_test, preds)
        fold_metrics.append({
            'fold': fold_idx + 1,
            'accuracy': fold_acc,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })
        
        if verbose:
            print(f"  Accuracy: {fold_acc:.4f}")
            
            # Class distribution
            for cls in [-1, 0, 1]:
                pred_pct = (preds == cls).mean() * 100
                true_pct = (y_test == cls).mean() * 100
                print(f"    Class {cls}: Pred {pred_pct:.1f}% | True {true_pct:.1f}%")
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_test_indices = np.array(all_test_indices)
    all_confidences = np.array(all_confidences)
    all_probas = np.array(all_probas)
    
    # Overall metrics
    overall_acc = accuracy_score(all_targets, all_predictions)
    
    if verbose:
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Overall Accuracy: {overall_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            all_targets, all_predictions,
            target_names=['Short', 'Flat', 'Long'],
            zero_division=0
        ))
        
        # Mean fold metrics
        mean_acc = np.mean([m['accuracy'] for m in fold_metrics])
        std_acc = np.std([m['accuracy'] for m in fold_metrics])
        print(f"\nMean Fold Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return {
        'predictions': all_predictions,
        'probabilities': all_probas,
        'targets': all_targets,
        'test_indices': all_test_indices,
        'confidence': all_confidences,
        'fold_metrics': fold_metrics,
        'overall_accuracy': overall_acc
    }


def estimated_sharpe_ratio(returns: np.ndarray) -> float:
    """Calculate annualized Sharpe Ratio."""
    if len(returns) < 2:
        return 0.0
    return np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(365)


def deflated_sharpe_ratio(
    sharpe_ratios: List[float], 
    n_trials: int, 
    expected_sr: float = 0.0
) -> float:
    """
    Calculate the Deflated Sharpe Ratio (DSR).
    
    Args:
        sharpe_ratios: List of SRs from CV folds/paths
        n_trials: Number of total trials/experiments run (N)
        expected_sr: Expected SR (usually 0)
        
    Returns:
        Probability that the true SR is greater than expected_sr
    """
    # 1. Calculate variance of the SRs
    sr_std = np.std(sharpe_ratios, ddof=1)
    
    # 2. Expected maximum SR under the null hypothesis
    # Euler-Mascheroni constant
    gamma = 0.5772156649
    exp_max_sr = expected_sr + sr_std * ((1 - gamma) * norm.ppf(1 - 1/n_trials) + 
                                         gamma * norm.ppf(1 - 1/n_trials * np.exp(-1)))
    
    # Simple approximation for max Z score from n_trials independent Gaussian
    # expected_max = norm.ppf(1 - 1/n_trials) # simplified
    
    # 3. Calculate DSR (Probabilistic SR adjusted for multiple testing)
    # Using the observed best SR (or mean of paths if checking strategy robustness)
    # Here we typically check if the STRATEGY's SR is statistically significant 
    # given we tried n_trials variations.
    
    # Assuming 'sharpe_ratios' represents the distribution of the final strategy's performance
    # across CV folds. We want to know if the Mean SR is significant given n_trials.
    
    current_sr = np.mean(sharpe_ratios)
    t = len(sharpe_ratios) # Number of observations (folds)
    
    # Skewness and Kurtosis of returns could be added for better precision (Probabilistic SR)
    # For now using standard formulation
    
    # DSR formula for a single strategy selected from N trials:
    # We adjust the benchmark. Instead of 0, we compare against the Expected Max SR.
    
    numerator = (current_sr - exp_max_sr) * np.sqrt(t - 1)
    # Denominator is 1 for SR distribution if normalized, else sample skew/kurtosis adjustment
    # Simplified DSR:
    dsr = norm.cdf(numerator / (1 - np.mean(sharpe_ratios) * np.mean(np.abs(np.array(sharpe_ratios, dtype=float))**3))**0.5) # This formula is complex
    
    # Let's use the explicit DSR formulation from Lopez de Prado (2018)
    # DSR = PSR(SR_est, SR_benchmark=E[max_SR])
    
    # If we treat the list of sharpe_ratios as the distribution of the strategy across folds
    # We can calculate PSR against the exp_max_sr
    
    skew = 0 # Assuming 0 for now
    kurt = 3 # Gaussian
    
    benchmark = exp_max_sr
    stat = (current_sr - benchmark) * np.sqrt(t - 1)
    stat /= np.sqrt(1 - skew * current_sr + (kurt - 1) / 4 * current_sr**2)
    
    return norm.cdf(stat)


if __name__ == "__main__":
    # Test validation framework
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 500
    n_features = config.SIGNATURE_DIMENSIONS
    
    X = np.random.randn(n_samples, n_features)
    
    # Create labels with some structure
    scores = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.5
    y = np.where(scores > 0.3, 1, np.where(scores < -0.3, -1, 0))
    
    print("Testing Walk-Forward Validation")
    print("=" * 50)
    print(f"Data shape: {X.shape}")
    print(f"Label distribution:")
    for label in [-1, 0, 1]:
        print(f"  Class {label}: {(y == label).sum()}")
    
    cv = PurgedWalkForwardCV(n_splits=5, purge_window=24, expanding=True)
    
    print(f"\nSplit configuration:")
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"  Fold {fold_idx + 1}:")
        print(f"    Train: {len(train_idx)} samples [{train_idx[0]}-{train_idx[-1]}]")
        print(f"    Test: {len(test_idx)} samples [{test_idx[0]}-{test_idx[-1]}]")
        print(f"    Gap: {test_idx[0] - train_idx[-1] - 1} samples")
        print(f"    No leakage: {validate_no_leakage(train_idx, test_idx, 24)}")
    
    print("\nTesting Combinatorial Purged CV")
    print("=" * 50)
    # N=6, k=2 -> 15 combinations
    cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, purge_window=10, embargo_pct=0.01)
    
    print(f"Total splits (6 choose 2): {cpcv.get_n_splits()}")
    
    for i, (train_idx, test_idx) in enumerate(cpcv.split(X)):
        if i < 3: # Print first 3
            print(f"  Split {i+1}:")
            print(f"    Train: {len(train_idx)} samples")
            print(f"    Test: {len(test_idx)} samples")
            
            # Check for overlap
            overlap = np.intersect1d(train_idx, test_idx)
            print(f"    Overlap: {len(overlap)} samples")
            
    # Test DSR
    print("\nTesting Deflated Sharpe Ratio")
    print("=" * 50)
    # Simulate 100 trials with mean SR 0.5 and std 0.5
    srs = np.random.normal(0.5, 0.5, 100)
    dsr_prob = deflated_sharpe_ratio(list(srs), n_trials=100)
    print(f"Mean SR: {np.mean(srs):.2f}")
    print(f"DSR Probability: {dsr_prob:.4f}")
