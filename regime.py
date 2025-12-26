"""
Regime detection using Detrended Fluctuation Analysis (DFA).

DFA is used to identify the long-memory properties of the time series.
- alpha > 1.0: Trending (Persistent)
- alpha < 1.0: Mean-reverting (Anti-persistent) or Noise
- alpha approx 1.5: Random Walk (if input is price)

By only trading in trending regimes, we filter out periods
where the signature-based model has no structural edge.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List


def detrended_fluctuation_analysis(series: np.ndarray, scales: Optional[List[int]] = None) -> float:
    """
    Compute the scaling exponent alpha using Detrended Fluctuation Analysis (DFA).
    
    DFA measures the long-range correlations in a time series.
    
    Args:
        series: 1D numpy array (Time series data, e.g. Prices)
        scales: List of window sizes (scales) to compute fluctuation over.
        
    Returns:
        Scaling exponent alpha.
    """
    series = np.array(series)
    n = len(series)
    
    if n < 50:
        return 0.5 # Default to uncorrelated if insufficient data
        
    # 1. Integrate the series (Profile)
    # We remove the mean and integrate to get the profile Y(k)
    y = np.cumsum(series - np.mean(series))
    
    if scales is None:
        # Generate logarithmic scales
        min_scale = 10
        max_scale = n // 4
        if max_scale < min_scale:
             return 0.5
        # 20 scales log-spaced
        scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), num=20).astype(int))
        scales = scales[scales > 5] # Ensure minimal window size
        
    if len(scales) < 2:
        return 0.5

    fluctuations = []
    
    for scale in scales:
        n_segments = n // scale
        if n_segments < 1:
            continue
            
        rmses = []
        for i in range(n_segments):
            # Segment
            seg = y[i*scale : (i+1)*scale]
            x = np.arange(scale)
            
            # Detrend (Polynomial fit degree 1)
            coef = np.polyfit(x, seg, 1)
            trend = np.polyval(coef, x)
            
            # RMS fluctuation
            rms = np.sqrt(np.mean((seg - trend)**2))
            rmses.append(rms)
            
        # F(n) is sqrt(mean(RMS^2)) over segments
        f_n = np.sqrt(np.mean(np.array(rmses)**2))
        fluctuations.append(f_n)
        
    if len(fluctuations) < 3:
        return 0.5
        
    # Linear regression log(F(n)) vs log(n)
    # Filter out zeros if any
    scales_arr = np.array(scales[:len(fluctuations)])
    flucs_arr = np.array(fluctuations)
    
    valid = (flucs_arr > 0)
    if not np.any(valid):
        return 0.5
        
    log_scales = np.log(scales_arr[valid])
    log_flucs = np.log(flucs_arr[valid])
    
    alpha, _ = np.polyfit(log_scales, log_flucs, 1)
    return float(alpha)


def rolling_dfa(
    series: pd.Series,
    window: int = 100,
    min_periods: int = 50
) -> pd.Series:
    """
    Compute rolling DFA alpha.
    
    Args:
        series: Price series
        window: Rolling window size
        min_periods: Minimum periods
        
    Returns:
        Series of rolling alpha values
    """
    dfa_values = []
    
    for i in range(len(series)):
        if i < min_periods - 1:
            dfa_values.append(np.nan)
        else:
            start = max(0, i - window + 1)
            window_data = series.iloc[start:i + 1].values
            try:
                alpha = detrended_fluctuation_analysis(window_data)
                dfa_values.append(alpha)
            except Exception:
                dfa_values.append(np.nan)
    
    return pd.Series(dfa_values, index=series.index)


def get_regime(
    alpha: float,
    trending_threshold: float = 1.0,
    mean_revert_threshold: float = 1.0
) -> str:
    """
    Classify market regime based on DFA alpha.
    """
    if np.isnan(alpha):
        return 'unknown'
    elif alpha > trending_threshold:
        return 'trending'
    elif alpha < mean_revert_threshold:
        return 'mean_reverting'
    else:
        return 'random_walk'


def regime_filter(
    alpha_series: pd.Series,
    trending_threshold: float = 1.0
) -> pd.Series:
    """
    Create binary filter for tradeable regimes (Trending).
    """
    return alpha_series > trending_threshold


if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing DFA Calculation")
    print("=" * 50)
    
    # Test 1: Random walk (alpha approx 1.5 for Price, 0.5 for Returns)
    # DFA on Price -> Integrated RW -> alpha = 1.5
    n = 1000
    random_walk = np.cumsum(np.random.randn(n))
    alpha_rw = detrended_fluctuation_analysis(random_walk)
    print(f"\nRandom Walk (Price) Alpha: {alpha_rw:.4f} (expected approx 1.5)")
    
    # Test 2: White Noise (Price input = just noise) -> alpha = 0.5
    noise = np.random.randn(n)
    alpha_noise = detrended_fluctuation_analysis(noise)
    print(f"White Noise Alpha: {alpha_noise:.4f} (expected approx 0.5)")
    
    # Test 3: Trending (Super-diffusive)
    # Construct a fractional Brownian motion proxy or just a trend
    t = np.linspace(0, 10, n)
    trend = t**2 + np.cumsum(np.random.randn(n)) # Strong quadratic trend
    alpha_trend = detrended_fluctuation_analysis(trend)
    print(f"Trending Alpha: {alpha_trend:.4f} (expected > 1.5)")

    # Test Rolling
    print("\nTesting Rolling DFA...")
    series = pd.Series(random_walk)
    rolling_a = rolling_dfa(series, window=100)
    
    print(f"Rolling Alpha stats:")
    print(f"  Mean: {rolling_a.mean():.4f}")
    print(f"  Std: {rolling_a.std():.4f}")
    
    # Filter
    mask = regime_filter(rolling_a, trending_threshold=1.0)
    print(f"Tradeable (Alpha > 1.0): {mask.mean()*100:.1f}%")
