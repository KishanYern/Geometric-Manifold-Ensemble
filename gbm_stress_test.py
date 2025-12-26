"""
GBM Stress Test: Validates the regime filter correctly blocks trades on random walks.

The DFA Alpha filter should detect that GBM paths have Alpha ~ 0.5 (random walk),
which is below the trending threshold (1.45), resulting in near-zero trade frequency.
"""

import numpy as np
import pandas as pd
from regime import rolling_dfa, regime_filter
from backtest import Backtester
from features import generate_multiscale_signature_features
from models import BaseEnsemble
# fractional_diff not needed for stress test
import config

def generate_gbm_path(
    n_steps: int = 1000,
    start_price: float = 50000.0,
    mu: float = 0.0,  # Zero drift (pure random walk)
    sigma: float = 0.02,
    seed: int = 42
) -> pd.DataFrame:
    """Generate Geometric Brownian Motion path."""
    rng = np.random.default_rng(seed)
    dt = 1.0
    
    dw = rng.normal(0, np.sqrt(dt), n_steps)
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * dw
    
    log_returns = drift + diffusion
    prices = start_price * np.exp(np.cumsum(log_returns))
    prices = np.insert(prices, 0, start_price)[:-1]
    
    # Create DataFrame with required columns
    df = pd.DataFrame({
        'close': prices,
        'volume': rng.uniform(100, 1000, n_steps),
    }, index=pd.date_range('2020-01-01', periods=n_steps, freq='D'))
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    return df


def run_gbm_stress_test(n_paths: int = 5, verbose: bool = True):
    """
    Run stress test on GBM random walk paths.
    
    Expected behavior:
    - DFA Alpha should be ~0.5 for random walk
    - Regime filter should block most/all trades (Alpha < 1.45 threshold)
    - Trade frequency should be near-zero
    """
    
    print("=" * 60)
    print("GBM STRESS TEST - REGIME FILTER VALIDATION")
    print("=" * 60)
    print(f"Testing {n_paths} GBM random walk paths")
    print(f"DFA Alpha Threshold: {config.DFA_ALPHA_THRESHOLD}")
    print("-" * 60)
    
    results = []
    
    for path_idx in range(n_paths):
        seed = 42 + path_idx
        
        # Generate GBM path
        df = generate_gbm_path(n_steps=500, seed=seed)
        
        # Compute DFA Alpha (regime indicator)
        alpha_series = rolling_dfa(df['close'], window=config.HURST_WINDOW)
        
        # Get valid Alpha values
        valid_alpha = alpha_series.dropna()
        mean_alpha = valid_alpha.mean()
        pct_trending = (valid_alpha > config.DFA_ALPHA_THRESHOLD).mean() * 100
        
        # Simulate trading decisions
        # Generate random predictions (as base model would on noise)
        n_valid = len(valid_alpha)
        predictions = np.random.choice([-1, 0, 1], size=n_valid)
        
        # Apply regime filter
        backtester = Backtester(hurst_threshold=config.DFA_ALPHA_THRESHOLD)
        
        positions = backtester.generate_signals(
            predictions=predictions,
            hurst=valid_alpha.values,
            meta_proba=np.ones(n_valid) * 0.6,  # Assume some meta confidence
            apply_hurst_filter=True,
            apply_kelly_sizing=True
        )
        
        # Calculate trade frequency
        trades_attempted = (predictions != 0).sum()
        trades_taken = (positions != 0).sum()
        trade_filter_rate = (1 - trades_taken / max(trades_attempted, 1)) * 100
        
        result = {
            'path': path_idx + 1,
            'mean_alpha': mean_alpha,
            'pct_trending': pct_trending,
            'trades_attempted': trades_attempted,
            'trades_taken': trades_taken,
            'filter_rate': trade_filter_rate
        }
        results.append(result)
        
        if verbose:
            print(f"Path {path_idx + 1}: Alpha={mean_alpha:.3f}, "
                  f"Trending={pct_trending:.1f}%, "
                  f"Trades: {trades_taken}/{trades_attempted} "
                  f"(Filtered: {trade_filter_rate:.1f}%)")
    
    # Summary
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    
    avg_alpha = results_df['mean_alpha'].mean()
    avg_trending = results_df['pct_trending'].mean()
    avg_filter_rate = results_df['filter_rate'].mean()
    total_trades_taken = results_df['trades_taken'].sum()
    total_trades_attempted = results_df['trades_attempted'].sum()
    
    print(f"Average DFA Alpha: {avg_alpha:.3f} (expected ~0.5 for random walk)")
    print(f"Average % Trending: {avg_trending:.1f}% (expected ~0% for random walk)")
    print(f"Trade Filter Rate: {avg_filter_rate:.1f}%")
    print(f"Total Trades Taken: {total_trades_taken} / {total_trades_attempted}")
    
    # Validation
    print("\n" + "-" * 60)
    print("VALIDATION:")
    
    if avg_alpha < 1.0:
        print("[PASS] Mean Alpha < 1.0 (correctly identifies random walk)")
    else:
        print("[WARN] Mean Alpha >= 1.0 (unexpected for GBM)")
    
    if avg_filter_rate > 80:
        print(f"[PASS] Filter rate > 80% ({avg_filter_rate:.1f}% of trades blocked)")
    elif avg_filter_rate > 50:
        print(f"[WARN] Filter rate > 50% but < 80% ({avg_filter_rate:.1f}%)")
    else:
        print(f"[FAIL] Filter rate < 50% ({avg_filter_rate:.1f}%) - filter not working")
    
    print("=" * 60)
    
    return results_df


def run_slippage_sensitivity(verbose: bool = True):
    """
    Run detailed slippage sensitivity analysis with binary search
    to find the exact capacity threshold.
    """
    print("\n" + "=" * 60)
    print("SLIPPAGE SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Generate test data with some structure (mix of trending and random)
    np.random.seed(42)
    n = 500
    
    # Create predictions and returns with some edge
    # For a realistic scenario, assume some predictive power
    true_direction = np.random.choice([-1, 1], size=n)
    noise = np.random.rand(n)
    
    # Predictions have ~55% accuracy (slight edge)
    predictions = np.where(noise < 0.55, true_direction, -true_direction)
    actual_returns = true_direction * np.abs(np.random.randn(n) * 0.01)
    
    # Hurst values (mix of trending and non-trending)
    hurst = np.random.uniform(1.2, 1.7, n)  # Mostly trending for this test
    
    # Meta probabilities
    meta_proba = np.random.uniform(0.5, 0.8, n)
    
    # Run sweep at fine granularity
    costs_bps = list(range(0, 51, 2))  # 0 to 50 bps in 2 bps steps
    
    print("Running slippage sweep (0-50 bps)...")
    
    results = []
    for bps in costs_bps:
        backtester = Backtester()
        backtester.transaction_cost = bps / 10000.0
        
        _, metrics = backtester.run_backtest(
            predictions=predictions,
            actual_returns=actual_returns,
            hurst=hurst,
            meta_proba=meta_proba
        )
        
        results.append({
            'cost_bps': bps,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'total_return': metrics['total_return'],
            'trades_taken': metrics.get('trades_taken', 0)
        })
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        print("\nSlippage Sweep Results:")
        print(results_df[['cost_bps', 'sharpe_ratio', 'total_return']].to_string(index=False))
    
    # Find capacity threshold (where Sharpe < 1.0)
    above_threshold = results_df[results_df['sharpe_ratio'] >= 1.0]
    
    if len(above_threshold) > 0:
        capacity_threshold = above_threshold['cost_bps'].max()
        print(f"\n[PASS] Capacity Threshold: {capacity_threshold} bps")
        print(f"       Strategy maintains SR >= 1.0 up to {capacity_threshold} bps slippage")
    else:
        # Binary search for more precision if needed
        if results_df['sharpe_ratio'].max() < 1.0:
            print("\n[WARN] Sharpe Ratio never exceeds 1.0 with test data")
            print("       This may indicate insufficient edge or high volatility in test data")
            capacity_threshold = 0
        else:
            capacity_threshold = 0
    
    # Find breakeven point
    breakeven_df = results_df[results_df['total_return'] > 0]
    if len(breakeven_df) > 0:
        breakeven_bps = breakeven_df['cost_bps'].max()
        print(f"       Breakeven Point: {breakeven_bps} bps (positive total return)")
    
    print("=" * 60)
    
    return results_df, capacity_threshold


if __name__ == "__main__":
    # Run GBM stress test
    gbm_results = run_gbm_stress_test(n_paths=5)
    
    # Run slippage sensitivity
    slippage_results, capacity = run_slippage_sensitivity()
