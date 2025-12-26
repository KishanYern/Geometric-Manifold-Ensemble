"""
Regime detection using the Hurst Exponent.

The Hurst Exponent H classifies market regimes:
- H > 0.55: Trending (persistent) - Best for momentum strategies
- H ≈ 0.50: Random walk - No edge, stay flat
- H < 0.45: Mean-reverting - Opposite of momentum

By only trading in trending regimes, we filter out periods
where the signature-based model has no structural edge.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_hurst_rs(series: np.ndarray) -> float:
    """
    Compute Hurst exponent using the Rescaled Range (R/S) method.
    
    The R/S method:
    1. Divide series into chunks of varying sizes
    2. For each chunk size, compute mean R/S
    3. Regress log(R/S) vs log(size) to get H
    
    Args:
        series: 1D numpy array of returns or prices
        
    Returns:
        Hurst exponent H ∈ (0, 1)
    """
    n = len(series)
    if n < 20:
        return 0.5  # Default to random walk for insufficient data
    
    # Convert to returns if not already
    if np.std(series) > 1:  # Likely prices, not returns
        series = np.diff(np.log(series))
        n = len(series)
    
    max_chunk = n // 4
    min_chunk = 10
    
    if max_chunk < min_chunk:
        return 0.5
    
    # Chunk sizes to analyze
    chunk_sizes = []
    rs_values = []
    
    size = min_chunk
    while size <= max_chunk:
        n_chunks = n // size
        if n_chunks < 1:
            break
            
        chunk_rs = []
        for i in range(n_chunks):
            chunk = series[i * size:(i + 1) * size]
            
            # Mean-adjusted cumulative deviation
            mean = np.mean(chunk)
            deviations = np.cumsum(chunk - mean)
            
            # Range
            R = np.max(deviations) - np.min(deviations)
            
            # Standard deviation
            S = np.std(chunk, ddof=1)
            
            if S > 0:
                chunk_rs.append(R / S)
        
        if chunk_rs:
            chunk_sizes.append(size)
            rs_values.append(np.mean(chunk_rs))
        
        size = int(size * 1.5)  # Exponential growth of chunk sizes
    
    if len(chunk_sizes) < 3:
        return 0.5
    
    # Linear regression of log(R/S) vs log(size)
    log_sizes = np.log(chunk_sizes)
    log_rs = np.log(rs_values)
    
    # Least squares fit
    slope, _ = np.polyfit(log_sizes, log_rs, 1)
    
    # Clamp to valid range
    return np.clip(slope, 0.0, 1.0)


def rolling_hurst(
    series: pd.Series,
    window: int = 30,
    min_periods: int = 20
) -> pd.Series:
    """
    Compute rolling Hurst exponent.
    
    Args:
        series: Price or return series
        window: Rolling window size (default 30 days)
        min_periods: Minimum periods needed to compute
        
    Returns:
        Series of rolling Hurst exponents
    """
    hurst_values = []
    
    for i in range(len(series)):
        if i < min_periods - 1:
            hurst_values.append(np.nan)
        else:
            start = max(0, i - window + 1)
            window_data = series.iloc[start:i + 1].values
            h = compute_hurst_rs(window_data)
            hurst_values.append(h)
    
    return pd.Series(hurst_values, index=series.index)


def get_regime(
    hurst_value: float,
    trending_threshold: float = 0.55,
    mean_revert_threshold: float = 0.45
) -> str:
    """
    Classify market regime based on Hurst exponent.
    
    Args:
        hurst_value: Current Hurst exponent
        trending_threshold: H above this = trending
        mean_revert_threshold: H below this = mean-reverting
        
    Returns:
        'trending', 'mean_reverting', or 'random_walk'
    """
    if np.isnan(hurst_value):
        return 'unknown'
    elif hurst_value > trending_threshold:
        return 'trending'
    elif hurst_value < mean_revert_threshold:
        return 'mean_reverting'
    else:
        return 'random_walk'


def regime_filter(
    hurst_series: pd.Series,
    trending_threshold: float = 0.55
) -> pd.Series:
    """
    Create binary filter for tradeable regimes.
    
    Returns True only when market is in trending regime.
    
    Args:
        hurst_series: Series of Hurst exponents
        trending_threshold: H above this to allow trading
        
    Returns:
        Boolean series (True = trade allowed)
    """
    return hurst_series > trending_threshold


if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing Hurst Exponent Calculation")
    print("=" * 50)
    
    # Test 1: Random walk (H ≈ 0.5)
    n = 500
    random_walk = np.cumsum(np.random.randn(n))
    h_rw = compute_hurst_rs(random_walk)
    print(f"\nRandom Walk H: {h_rw:.4f} (expected ≈ 0.5)")
    
    # Test 2: Trending series (H > 0.5)
    # Create momentum by accumulating biased increments
    trend = np.zeros(n)
    for i in range(1, n):
        # Each step is biased in the direction of previous step
        bias = 0.3 * np.sign(trend[i-1] - trend[max(0, i-2)])
        trend[i] = trend[i-1] + bias + np.random.randn() * 0.5
    h_trend = compute_hurst_rs(trend)
    print(f"Trending Series H: {h_trend:.4f} (expected > 0.5)")
    
    # Test 3: Mean-reverting series (H < 0.5)
    mean_revert = np.zeros(n)
    for i in range(1, n):
        mean_revert[i] = -0.3 * mean_revert[i-1] + np.random.randn()
    mean_revert = np.cumsum(mean_revert)
    h_mr = compute_hurst_rs(mean_revert)
    print(f"Mean-Reverting H: {h_mr:.4f} (expected < 0.5)")
    
    # Test rolling Hurst
    print("\nTesting Rolling Hurst...")
    series = pd.Series(random_walk)
    rolling_h = rolling_hurst(series, window=30)
    
    print(f"Rolling Hurst stats:")
    print(f"  Mean: {rolling_h.mean():.4f}")
    print(f"  Std: {rolling_h.std():.4f}")
    print(f"  Min: {rolling_h.min():.4f}")
    print(f"  Max: {rolling_h.max():.4f}")
    
    # Test regime filter
    filter_mask = regime_filter(rolling_h, trending_threshold=0.55)
    pct_tradeable = filter_mask.sum() / len(filter_mask) * 100
    print(f"\nTradeable periods (H > 0.55): {pct_tradeable:.1f}%")
