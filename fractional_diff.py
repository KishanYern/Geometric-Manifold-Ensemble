"""
Fractional Differentiation for memory-preserving stationarity.
Implements Fixed-Width Window Fractional Differentiation (FFD).

The key insight: Standard log-returns (d=1) destroy long-term memory.
By finding the minimum d (usually 0.3-0.6) that achieves stationarity,
we retain maximum historical memory for path signature computation.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Optional


def get_ffd_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute Fixed-Width Fractional Differentiation weights.
    
    The weights follow the formula:
    w_k = -w_{k-1} * (d - k + 1) / k
    
    Args:
        d: Fractional differentiation order (0 < d < 1)
        threshold: Minimum weight magnitude to include
        
    Returns:
        Array of FFD weights
    """
    weights = [1.0]
    k = 1
    while True:
        w_next = -weights[-1] * (d - k + 1) / k
        if abs(w_next) < threshold:
            break
        weights.append(w_next)
        k += 1
    return np.array(weights[::-1])  # Reverse for convolution


def fractional_diff(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """
    Apply fractional differentiation to a time series.
    
    Uses Fixed-Width Window (FFD) method which:
    - Has O(T) memory complexity vs O(T^2) for expanding window
    - Produces consistent weights across all observations
    
    Args:
        series: Input price/return series (must be a pandas Series with index)
        d: Fractional differentiation order
        threshold: Weight cutoff threshold
        
    Returns:
        Fractionally differentiated series (shorter due to convolution)
    """
    weights = get_ffd_weights(d, threshold)
    width = len(weights)
    
    # Apply convolution (dot product of weights with rolling window)
    result = []
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1:i + 1].values
        result.append(np.dot(weights, window))
    
    # Return with aligned index
    return pd.Series(result, index=series.index[width - 1:])


def adf_test(series: pd.Series, max_lag: Optional[int] = None) -> Tuple[float, float]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series to test
        max_lag: Maximum lag for test (None = auto)
        
    Returns:
        Tuple of (ADF statistic, p-value)
    """
    result = adfuller(series.dropna(), maxlag=max_lag, autolag='AIC')
    return result[0], result[1]  # statistic, p-value


def find_optimal_d(
    series: pd.Series,
    d_range: Tuple[float, float] = (0.0, 1.0),
    target_pvalue: float = 0.05,
    tolerance: float = 0.01,
    max_iterations: int = 20
) -> Tuple[float, pd.Series]:
    """
    Find minimum fractional differentiation order that achieves stationarity.
    
    Uses binary search to find the smallest d where ADF p-value < target.
    This maximizes memory retention while ensuring stationarity.
    
    Args:
        series: Input price series
        d_range: Search range for d (default 0.0 to 1.0)
        target_pvalue: Target ADF p-value (default 0.05)
        tolerance: Convergence tolerance for d
        max_iterations: Maximum binary search iterations
        
    Returns:
        Tuple of (optimal_d, differentiated_series)
    """
    d_low, d_high = d_range
    optimal_d = d_high
    optimal_series = None
    
    for _ in range(max_iterations):
        d_mid = (d_low + d_high) / 2
        
        try:
            diff_series = fractional_diff(series, d_mid)
            if len(diff_series) < 20:  # Need enough points for ADF
                d_low = d_mid
                continue
                
            _, pvalue = adf_test(diff_series)
            
            if pvalue < target_pvalue:
                # Stationary - try lower d
                optimal_d = d_mid
                optimal_series = diff_series
                d_high = d_mid
            else:
                # Not stationary - need higher d
                d_low = d_mid
                
            if d_high - d_low < tolerance:
                break
                
        except Exception:
            # If test fails, try higher d
            d_low = d_mid
    
    # If no optimal found, use maximum d
    if optimal_series is None:
        optimal_series = fractional_diff(series, optimal_d)
    
    return optimal_d, optimal_series


if __name__ == "__main__":
    # Test fractional differentiation
    np.random.seed(42)
    
    # Create a random walk (non-stationary)
    n = 500
    random_walk = pd.Series(np.cumsum(np.random.randn(n)))
    
    print("Testing Fractional Differentiation")
    print("=" * 50)
    
    # Test ADF on original series
    stat, pval = adf_test(random_walk)
    print(f"\nOriginal series ADF:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  P-value: {pval:.4f}")
    
    # Find optimal d
    print("\nFinding optimal d...")
    optimal_d, diff_series = find_optimal_d(random_walk)
    
    stat, pval = adf_test(diff_series)
    print(f"\nOptimal d: {optimal_d:.4f}")
    print(f"Differentiated series ADF:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  P-value: {pval:.4f}")
    print(f"  Series length: {len(diff_series)} (from {len(random_walk)})")
    
    # Compare with d=1 (log returns equivalent)
    diff_1 = fractional_diff(random_walk, d=1.0)
    print(f"\nFor comparison, d=1.0 series length: {len(diff_1)}")
