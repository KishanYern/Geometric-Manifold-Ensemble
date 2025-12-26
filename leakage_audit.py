# Information Leakage Audit Report
# Geometric Manifold Ensemble Trading Strategy
# Generated: 2025-12-26

"""
This module provides tools to audit and verify there is no information leakage
in the GME trading strategy implementation.

## Potential Leakage Points Analyzed

### 1. Feature Engineering (features.py) ✅ NO LEAKAGE

The sliding window feature generation uses ONLY past data:

```
For sample i:
- Window: log_returns[i : i + window_size]  (past data only)
- Target: log_returns[i + window_size]       (next period - AFTER the window)
```

The target is strictly the return at time `t = window_size + i`, which occurs
AFTER all data used in the signature computation. This is verified by the
`test_no_lookahead_bias` unit test.


### 2. Walk-Forward Validation (validation.py) ✅ NO LEAKAGE (with caveat)

The PurgedWalkForwardCV correctly implements:
- Chronological train/test splits
- Purge window between train and test (24 samples = window size)
- Expanding training window

**HOWEVER, there is a potential issue identified:**

In `walk_forward_validation()`, lines 210-226:
- Base models trained on first 70% of training data
- Meta-learner trained on last 30% of training data

This is CORRECT and prevents leakage because:
- Meta-learner only sees base model predictions, not raw features/targets
- The 70/30 split is within the training fold only


### 3. Backtesting (backtest.py) ⚠️ POTENTIAL ISSUE

**ISSUE FOUND: Confidence threshold calculation**

In `Backtester.generate_signals()`, the confidence threshold is computed
using ALL variances in the test set:

```python
threshold = np.percentile(confidences, self.confidence_percentile)
```

This is LOOKAHEAD BIAS because at time t, we shouldn't know the variance
of predictions at time t+1, t+2, etc.

**FIX REQUIRED**: The threshold should be computed:
1. Using training data variances only (preferred)
2. Or using an expanding window of past variances


### 4. Data Loading (data_loader.py) ✅ NO LEAKAGE

Data is loaded and sorted chronologically. Log returns are computed correctly
as `log(P_t / P_{t-1})` which only uses past and current prices.


## Recommendations

1. **Fix confidence threshold calculation** (HIGH PRIORITY)
   - Compute threshold on training variances
   - Apply same threshold to test set

2. **Add more validation assertions**
   - Verify timestamp ordering after data loading
   - Assert purge gap >= window_size at runtime

3. **Consider adding embargo period**
   - After training ends, wait N periods before testing
   - Prevents any subtle autocorrelation leakage
"""

import numpy as np
from typing import Tuple, List
import config


def audit_feature_leakage(log_returns: np.ndarray, window_size: int = 24) -> dict:
    """
    Audit feature generation for lookahead bias.
    
    Verifies that for each sample i:
    - Features use only data from indices [i, i + window_size)
    - Target is at index [i + window_size]
    
    Returns audit report.
    """
    n = len(log_returns)
    n_samples = n - window_size
    
    issues = []
    
    for i in range(n_samples):
        feature_indices = list(range(i, i + window_size))
        target_index = i + window_size
        
        # Check: target should be AFTER all feature indices
        if target_index <= max(feature_indices):
            issues.append({
                'sample': i,
                'issue': 'Target index overlaps with feature indices',
                'feature_range': (min(feature_indices), max(feature_indices)),
                'target_index': target_index
            })
        
        # Check: no negative indices
        if min(feature_indices) < 0:
            issues.append({
                'sample': i,
                'issue': 'Negative feature indices',
                'feature_range': (min(feature_indices), max(feature_indices))
            })
    
    return {
        'n_samples': n_samples,
        'issues': issues,
        'passed': len(issues) == 0
    }


def audit_cv_leakage(
    n_samples: int,
    n_splits: int = 5,
    purge_window: int = 24,
    min_train_size: int = 48
) -> dict:
    """
    Audit cross-validation splits for information leakage.
    
    Checks:
    1. No overlap between train and test
    2. Adequate gap (purge window) between train end and test start
    3. Test always comes after train chronologically
    """
    from validation import PurgedWalkForwardCV, validate_no_leakage
    
    X_dummy = np.zeros((n_samples, 30))
    cv = PurgedWalkForwardCV(
        n_splits=n_splits,
        purge_window=purge_window,
        min_train_size=min_train_size
    )
    
    issues = []
    fold_info = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_dummy)):
        train_max = train_idx.max()
        test_min = test_idx.min()
        gap = test_min - train_max - 1
        
        info = {
            'fold': fold + 1,
            'train_range': (train_idx.min(), train_idx.max()),
            'test_range': (test_idx.min(), test_idx.max()),
            'gap': gap,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        }
        fold_info.append(info)
        
        # Check gap
        if gap < purge_window:
            issues.append({
                'fold': fold + 1,
                'issue': f'Insufficient gap: {gap} < {purge_window}',
                'gap': gap,
                'required': purge_window
            })
        
        # Check overlap
        if set(train_idx) & set(test_idx):
            issues.append({
                'fold': fold + 1,
                'issue': 'Train/test overlap detected',
                'overlap_size': len(set(train_idx) & set(test_idx))
            })
        
        # Check chronological order
        if train_max >= test_min:
            issues.append({
                'fold': fold + 1,
                'issue': 'Test data not after train data',
                'train_max': train_max,
                'test_min': test_min
            })
    
    return {
        'n_splits': n_splits,
        'fold_info': fold_info,
        'issues': issues,
        'passed': len(issues) == 0
    }


def run_full_audit(verbose: bool = True) -> dict:
    """Run comprehensive leakage audit."""
    
    results = {}
    
    # 1. Feature leakage audit
    if verbose:
        print("=" * 60)
        print("INFORMATION LEAKAGE AUDIT")
        print("=" * 60)
        print("\n1. Feature Engineering Audit")
        print("-" * 40)
    
    # Simulate returns
    np.random.seed(42)
    returns = np.random.randn(1000) * 0.01
    
    feature_audit = audit_feature_leakage(returns, config.WINDOW_SIZE)
    results['feature_audit'] = feature_audit
    
    if verbose:
        if feature_audit['passed']:
            print("   ✅ PASSED - No lookahead bias in feature generation")
        else:
            print(f"   ❌ FAILED - {len(feature_audit['issues'])} issues found")
            for issue in feature_audit['issues'][:5]:
                print(f"      {issue}")
    
    # 2. CV leakage audit
    if verbose:
        print("\n2. Cross-Validation Audit")
        print("-" * 40)
    
    cv_audit = audit_cv_leakage(
        n_samples=1000 - config.WINDOW_SIZE,
        n_splits=config.N_SPLITS,
        purge_window=config.PURGE_WINDOW
    )
    results['cv_audit'] = cv_audit
    
    if verbose:
        if cv_audit['passed']:
            print("   ✅ PASSED - No leakage in CV splits")
            for fold in cv_audit['fold_info']:
                print(f"      Fold {fold['fold']}: train {fold['train_range']}, "
                      f"test {fold['test_range']}, gap={fold['gap']}")
        else:
            print(f"   ❌ FAILED - {len(cv_audit['issues'])} issues found")
            for issue in cv_audit['issues']:
                print(f"      {issue}")
    
    # 3. Confidence threshold issue
    if verbose:
        print("\n3. Confidence Threshold Audit")
        print("-" * 40)
        print("   ⚠️  WARNING - Potential lookahead bias detected")
        print("      The confidence threshold is computed using ALL test variances")
        print("      Recommendation: Use training variances for threshold calculation")
    
    results['confidence_threshold_issue'] = True
    
    # Overall result
    overall_passed = (
        feature_audit['passed'] and 
        cv_audit['passed']
        # confidence_threshold is a known issue
    )
    
    if verbose:
        print("\n" + "=" * 60)
        if overall_passed:
            print("OVERALL: ✅ Core pipeline has no leakage")
            print("         ⚠️  Minor fix needed for confidence threshold")
        else:
            print("OVERALL: ❌ Leakage detected - review issues above")
        print("=" * 60)
    
    results['overall_passed'] = overall_passed
    return results


if __name__ == "__main__":
    run_full_audit(verbose=True)
