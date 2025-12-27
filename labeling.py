"""
Triple Barrier Labeling for classification-based trading.

Instead of predicting next-period returns (noisy regression),
we classify market state into Long/Short/Flat based on which
barrier the price hits first:
- Upper barrier (profit take): Label = 1 (Long was correct)
- Lower barrier (stop loss): Label = -1 (Short was correct)  
- Time barrier (timeout): Label = 0 (Flat/no clear trend)

Barriers are dynamic, scaled by rolling volatility σ.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import config


def compute_daily_volatility(
    close: pd.Series,
    window: int = 20,
    annualize: bool = False
) -> pd.Series:
    """
    Compute rolling daily volatility using log returns.
    
    Args:
        close: Close price series
        window: Rolling window size (default 20 days)
        annualize: Whether to annualize (multiply by sqrt(365))
        
    Returns:
        Rolling volatility series
    """
    log_returns = np.log(close / close.shift(1))
    volatility = log_returns.rolling(window=window).std()
    
    if annualize:
        volatility = volatility * np.sqrt(365)
    
    return volatility


def get_barrier_timestamps(
    close: pd.Series,
    start_idx: int,
    upper_barrier: float,
    lower_barrier: float,
    max_holding_period: int,
    alpha_series: Optional[pd.Series] = None,
    persistence_threshold: float = config.PERSISTENCE_ALPHA_THRESHOLD,
    max_extension: int = config.PERSISTENCE_EXTENSION_CANDLES
) -> Tuple[int, int]:
    """
    Find which barrier is hit first and when.
    
    Enhanced Persistence Logic:
    1. If DFA Alpha at entry > persistence_threshold, DOUBLE the holding period upfront
       to let winners run in persistent trending regimes.
    2. During extended period, continue checking barriers.
    3. If Alpha drops below threshold during extension, exit early.
    
    Args:
        close: Price series
        start_idx: Starting index for the trade
        upper_barrier: Upper price barrier (absolute)
        lower_barrier: Lower price barrier (absolute)
        max_holding_period: Maximum bars to hold (base)
        alpha_series: Optional DFA Alpha series for persistence check
        persistence_threshold: Alpha threshold for extension (default 1.45)
        max_extension: Maximum additional bars to extend (default from config)
        
    Returns:
        Tuple of (barrier_type, bars_held)
        barrier_type: 1 (upper), -1 (lower), 0 (time)
    """
    entry_price = close.iloc[start_idx]
    
    # Determine holding period based on entry-time Alpha (persistence at entry)
    current_holding = max_holding_period
    if alpha_series is not None and start_idx < len(alpha_series):
        entry_alpha = alpha_series.iloc[start_idx]
        if not np.isnan(entry_alpha) and entry_alpha > persistence_threshold:
            # Double the holding window for persistent trends at entry
            current_holding = max_holding_period * 2
    
    end_idx = min(start_idx + current_holding, len(close) - 1)
    
    # Phase 1: Check barriers during (potentially extended) holding period
    for i in range(start_idx + 1, end_idx + 1):
        price = close.iloc[i]
        
        # Check upper barrier first (profit take)
        if price >= upper_barrier:
            return 1, i - start_idx
        
        # Check lower barrier (stop loss)
        if price <= lower_barrier:
            return -1, i - start_idx
        
        # If in extended period (beyond base max_holding), check if Alpha drops
        if i > start_idx + max_holding_period and alpha_series is not None:
            if i < len(alpha_series):
                alpha_at_i = alpha_series.iloc[i]
                if np.isnan(alpha_at_i) or alpha_at_i <= persistence_threshold:
                    # Persistence broken during extension - exit with time barrier
                    return 0, i - start_idx
    
    # Phase 2: Additional extension check (if not already extended)
    # Only applies if we didn't extend upfront and Alpha is still high at end
    if current_holding == max_holding_period:
        if alpha_series is not None and end_idx < len(alpha_series):
            current_alpha = alpha_series.iloc[end_idx]
            
            if not np.isnan(current_alpha) and current_alpha > persistence_threshold:
                # Extend holding period while Alpha remains high
                extended_end = min(end_idx + max_extension, len(close) - 1)
                
                for i in range(end_idx + 1, extended_end + 1):
                    price = close.iloc[i]
                    
                    # Check if Alpha drops below threshold (persistence break)
                    if i < len(alpha_series):
                        alpha_at_i = alpha_series.iloc[i]
                        if np.isnan(alpha_at_i) or alpha_at_i <= persistence_threshold:
                            return 0, i - start_idx
                    
                    # Check upper barrier
                    if price >= upper_barrier:
                        return 1, i - start_idx
                    
                    # Check lower barrier
                    if price <= lower_barrier:
                        return -1, i - start_idx
                
                # Extended period exhausted
                return 0, extended_end - start_idx
    
    # Time barrier hit
    return 0, current_holding


def triple_barrier_labels(
    close: pd.Series,
    volatility: pd.Series,
    upper_mult: float = 2.0,
    lower_mult: float = 1.0,
    max_holding_period: int = 5,
    alpha_series: Optional[pd.Series] = None,
    persistence_threshold: float = config.PERSISTENCE_ALPHA_THRESHOLD,
    max_extension: int = config.PERSISTENCE_EXTENSION_CANDLES
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate Triple Barrier labels for each time point.
    
    For a LONG position at time t:
    - Upper barrier: close[t] + upper_mult * volatility[t]
    - Lower barrier: close[t] - lower_mult * volatility[t]
    - Time barrier: t + max_holding_period
    
    Supports structural persistence: if time barrier would be hit but
    DFA Alpha remains high (> persistence_threshold), the holding period
    is extended up to max_extension additional candles.
    
    Labels:
    - 1: Upper barrier hit first (Long correct, bullish trend)
    - -1: Lower barrier hit first (Short correct, bearish trend)
    - 0: Time barrier hit (No clear trend, flat)
    
    Args:
        close: Close price series
        volatility: Rolling volatility series
        upper_mult: Upper barrier multiplier (default 2x volatility)
        lower_mult: Lower barrier multiplier (default 1x volatility)
        max_holding_period: Maximum holding period in bars
        alpha_series: Optional DFA Alpha series for persistence extension
        persistence_threshold: Alpha threshold for extending holding period
        max_extension: Maximum additional bars to extend
        
    Returns:
        Tuple of (labels, holding_periods)
    """
    n = len(close)
    labels = pd.Series(index=close.index, dtype=float)
    holding_periods = pd.Series(index=close.index, dtype=float)
    
    # Account for potential extensions in loop range
    total_max_hold = max_holding_period + (max_extension if alpha_series is not None else 0)
    
    for i in range(len(close) - total_max_hold):
        idx = close.index[i]
        
        # Skip if volatility not available
        if pd.isna(volatility.iloc[i]) or volatility.iloc[i] <= 0:
            labels.iloc[i] = np.nan
            holding_periods.iloc[i] = np.nan
            continue
        
        # Compute barriers based on current volatility
        vol = volatility.iloc[i]
        entry_price = close.iloc[i]
        upper_barrier = entry_price * (1 + upper_mult * vol)
        lower_barrier = entry_price * (1 - lower_mult * vol)
        
        # Find which barrier is hit first (with optional persistence extension)
        barrier_type, bars = get_barrier_timestamps(
            close, i, upper_barrier, lower_barrier, max_holding_period,
            alpha_series=alpha_series,
            persistence_threshold=persistence_threshold,
            max_extension=max_extension
        )
        
        labels.iloc[i] = barrier_type
        holding_periods.iloc[i] = bars
    
    return labels, holding_periods


def compute_meta_labels(
    labels: pd.Series,
    predictions: pd.Series
) -> pd.Series:
    """
    Compute meta-labels for whether base prediction was correct.
    
    A prediction is "correct" if:
    - Predicted Long (1) and label is Long (1)
    - Predicted Short (-1) and label is Short (-1)
    
    Args:
        labels: True triple barrier labels
        predictions: Base model predictions (1, -1, or 0)
        
    Returns:
        Binary series: 1 if prediction was correct, 0 otherwise
    """
    # Correct if signs match and neither is 0
    correct = ((labels == predictions) & (labels != 0) & (predictions != 0)).astype(int)
    
    # Also consider flat predictions correct if label is flat
    flat_correct = ((predictions == 0) & (labels == 0)).astype(int)
    
    return correct | flat_correct


if __name__ == "__main__":
    # Test triple barrier labeling
    np.random.seed(42)
    
    # Create synthetic price series
    n = 200
    returns = np.random.randn(n) * 0.02  # 2% daily volatility
    returns += 0.001  # Small drift
    prices = 100 * np.exp(np.cumsum(returns))
    close = pd.Series(prices, index=pd.date_range('2023-01-01', periods=n, freq='D'))
    
    print("Testing Triple Barrier Labeling")
    print("=" * 50)
    
    # Compute volatility
    volatility = compute_daily_volatility(close, window=20)
    print(f"\nVolatility stats:")
    print(f"  Mean: {volatility.mean():.4f}")
    print(f"  Std: {volatility.std():.4f}")
    
    # Generate labels
    labels, holding = triple_barrier_labels(
        close, volatility,
        upper_mult=2.0, lower_mult=1.0, max_holding_period=5
    )
    
    # Drop NaN and count
    valid_labels = labels.dropna()
    print(f"\nLabel distribution:")
    print(f"  Long (1):  {(valid_labels == 1).sum()} ({(valid_labels == 1).mean()*100:.1f}%)")
    print(f"  Short (-1): {(valid_labels == -1).sum()} ({(valid_labels == -1).mean()*100:.1f}%)")
    print(f"  Flat (0):  {(valid_labels == 0).sum()} ({(valid_labels == 0).mean()*100:.1f}%)")
    
    print(f"\nAverage holding period: {holding.dropna().mean():.1f} bars")
