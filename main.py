"""
Main orchestrator for Geometric Manifold Ensemble (GME) trading strategy.
Runs the complete classification-based pipeline from data loading to backtesting.

Pipeline:
1. Load data (Binance via CCXT or yfinance fallback)
2. Apply fractional differentiation for memory-preserving stationarity
3. Generate triple barrier labels (Long/Short/Flat)
4. Compute multi-dimensional path signatures
5. Nested Walk-forward validation (Base Ensemble + Meta-Learner)
6. Regime Filter (DFA)
7. Backtest with filtering
"""

import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import config
from data_loader import load_btc_data, prepare_dataset
from features import generate_signature_features, generate_multiscale_signature_features
from models import BaseEnsemble
from validation import PurgedWalkForwardCV, nested_walk_forward_validation
from backtest import Backtester
from fractional_diff import find_optimal_d
from labeling import compute_daily_volatility, triple_barrier_labels
from regime import rolling_dfa, regime_filter

# Optional imports
try:
    from binance_loader import load_binance_data
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False


def run_pipeline(
    data_path: str = None,
    use_binance: bool = True,
    verbose: bool = True,
    save_results: bool = True
) -> dict:
    """
    Run the complete GME classification pipeline with Nested CV.
    """
    if verbose:
        print("=" * 60)
        print("GEOMETRIC MANIFOLD ENSEMBLE (GME) - CLASSIFICATION FRAMEWORK")
        print("=" * 60)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================
    # Step 1: Data Loading
    # ========================================
    if verbose:
        print("\n" + "-" * 40)
        print("STEP 1: Loading Data")
        print("-" * 40)
    
    # Try Binance first for OI and Funding Rates
    df = None
    if use_binance and BINANCE_AVAILABLE and data_path is None:
        try:
            if verbose:
                print("Attempting to load from Binance (CCXT)...")
            df = load_binance_data(
                symbol=config.SYMBOL,
                timeframe=config.TIMEFRAME,
                lookback_days=config.LOOKBACK_DAYS,
                include_oi=True,
                include_funding=True,
                include_liquidations=True  # 5th dimension
            )
            data_source = "Binance"
        except Exception as e:
            if verbose:
                print(f"Binance load failed: {e}")
                print("Falling back to yfinance...")
            df = None
    
    # Fall back to yfinance
    if df is None:
        df_raw = load_btc_data(filepath=data_path)
        df = prepare_dataset(df_raw)
        data_source = "yfinance (fallback)"
    
    if verbose:
        print(f"Data source: {data_source}")
        print(f"Loaded {len(df)} records")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # ========================================
    # Step 2: Fractional Differentiation
    # ========================================
    if verbose:
        print("\n" + "-" * 40)
        print("STEP 2: Fractional Differentiation")
        print("-" * 40)
    
    # Get price series
    price_col = 'close' if 'close' in df.columns else 'price'
    prices = df[price_col]
    
    # Find optimal d for stationarity
    optimal_d, price_fracdiff = find_optimal_d(
        prices,
        d_range=config.FRACDIFF_D_RANGE,
        target_pvalue=config.FRACDIFF_ADF_PVALUE,
        threshold=config.FRACDIFF_THRESHOLD
    )
    
    if verbose:
        print(f"Optimal d: {optimal_d:.4f}")
    
    # Align other columns with fracdiff output
    align_start = len(prices) - len(price_fracdiff)
    df_aligned = df.iloc[align_start:].copy()
    df_aligned['price_fracdiff'] = price_fracdiff.values
    
    # Normalize additional dimensions if available
    if 'cvd' in df_aligned.columns:
        df_aligned['cvd_normalized'] = (df_aligned['cvd'] - df_aligned['cvd'].mean()) / df_aligned['cvd'].std()
    elif 'volume' in df_aligned.columns:
        # Use cumulative volume as proxy for CVD
        cvd = df_aligned['volume'].cumsum()
        df_aligned['cvd_normalized'] = (cvd - cvd.mean()) / cvd.std()
    
    if 'open_interest' in df_aligned.columns:
        # Re-normalize just in case
        df_aligned['oi_normalized'] = (df_aligned['open_interest'] - df_aligned['open_interest'].mean()) / df_aligned['open_interest'].std()
    
    # ========================================
    # Step 3: Compute DFA Scaling Exponent (Alpha)
    # NOTE: Computed BEFORE labeling to enable structural persistence
    # ========================================
    if verbose:
        print("\n" + "-" * 40)
        print("STEP 3: Regime Detection (DFA Alpha)")
        print("-" * 40)
    
    # Using Rolling DFA on Prices
    alpha_series = rolling_dfa(df_aligned[price_col], window=config.HURST_WINDOW)
    df_aligned['hurst'] = alpha_series  # Storing in 'hurst' col for compatibility
    
    if verbose:
        valid_alpha = alpha_series.dropna()
        trending_pct = (valid_alpha > 1.0).mean() * 100
        print(f"DFA Alpha stats:")
        print(f"  Mean: {valid_alpha.mean():.4f}")
        print(f"  {trending_pct:.1f}% periods are trending (Alpha > 1.0)")
    
    # ========================================
    # Step 4: Triple Barrier Labeling (with Structural Persistence)
    # ========================================
    if verbose:
        print("\n" + "-" * 40)
        print("STEP 4: Triple Barrier Labeling (Structural Persistence)")
        print("-" * 40)
    
    # Compute volatility for dynamic barriers
    volatility = compute_daily_volatility(
        df_aligned[price_col],
        window=config.VOLATILITY_WINDOW
    )
    df_aligned['volatility'] = volatility
    
    # Generate labels with structural persistence
    # If Alpha > PERSISTENCE_ALPHA_THRESHOLD at time timeout, extend holding period
    labels, holding_periods = triple_barrier_labels(
        df_aligned[price_col],
        volatility,
        upper_mult=config.BARRIER_UPPER_MULT,
        lower_mult=config.BARRIER_LOWER_MULT,
        max_holding_period=config.BARRIER_TIME_DAYS,
        alpha_series=alpha_series,  # Enable structural persistence
        persistence_threshold=config.PERSISTENCE_ALPHA_THRESHOLD,
        max_extension=config.PERSISTENCE_EXTENSION_CANDLES
    )
    df_aligned['label'] = labels
    df_aligned['holding_period'] = holding_periods
    
    if verbose:
        valid_labels = labels.dropna()
        print(f"Label distribution (with persistence extension):")
        print(f"  Long (1):  {(valid_labels == 1).sum()} ({(valid_labels == 1).mean()*100:.1f}%)")
        print(f"  Short (-1): {(valid_labels == -1).sum()} ({(valid_labels == -1).mean()*100:.1f}%)")
        print(f"  Flat (0):  {(valid_labels == 0).sum()} ({(valid_labels == 0).mean()*100:.1f}%)")
        avg_hold = holding_periods.dropna().mean()
        print(f"  Avg holding period: {avg_hold:.1f} candles")
    
    # ========================================
    # Step 5: Generate Multi-Scale Signature Features
    # ========================================
    if verbose:
        print("\n" + "-" * 40)
        print("STEP 5: Generating Multi-Scale Signature Features")
        print("-" * 40)
    
    # Prepare dimensions - Handle Hybrid Mode holes
    dims_available = ['price_fracdiff']
    if 'cvd_normalized' in df_aligned.columns:
        dims_available.append('cvd_normalized')
    if 'oi_normalized' in df_aligned.columns:
        dims_available.append('oi_normalized')
    if 'funding_rate' in df_aligned.columns:
        dims_available.append('funding_rate')
    if 'liquidations_normalized' in df_aligned.columns:
        dims_available.append('liquidations_normalized')
    
    if verbose:
        print(f"Path dimensions: {len(dims_available)} ({', '.join(dims_available)})")
        print(f"Multi-scale windows: {config.MULTI_SCALE_WINDOWS}")
    
    # Use max window for alignment
    max_window = max(config.MULTI_SCALE_WINDOWS)
    
    # Generate multi-scale 'Fractal Signature' features
    features = generate_multiscale_signature_features(
        price_fracdiff=df_aligned['price_fracdiff'].values,
        cvd=df_aligned.get('cvd_normalized', pd.Series([None])).values if 'cvd_normalized' in df_aligned.columns else None,
        oi=df_aligned.get('oi_normalized', pd.Series([None])).values if 'oi_normalized' in df_aligned.columns else None,
        funding_rate=df_aligned.get('funding_rate', pd.Series([None])).values if 'funding_rate' in df_aligned.columns else None,
        liquidations=df_aligned.get('liquidations_normalized', pd.Series([None])).values if 'liquidations_normalized' in df_aligned.columns else None,
        window_sizes=config.MULTI_SCALE_WINDOWS,
        degree=config.SIGNATURE_DEGREE
    )
    
    if verbose:
        print(f"Generated {len(features)} samples")
        print(f"Multi-scale signature dimension: {features.shape[1]} (before PCA)")
    
    # ========================================
    # Step 5b: Append Regime Features (Alpha/Volatility)
    # ========================================
    # Align targets, alpha, volatility with features (using max_window)
    target_start = max_window
    target_end = target_start + len(features)
    
    targets = df_aligned['label'].values[target_start:target_end]
    alpha_aligned = df_aligned['hurst'].values[target_start:target_end]
    vol_aligned = df_aligned['volatility'].values[target_start:target_end]
    timestamps = df_aligned.index[target_start:target_end]
    
    # Compute actual returns for backtesting
    if 'log_return' in df_aligned.columns:
        actual_returns = df_aligned['log_return'].values[target_start:target_end]
    else:
        actual_returns = np.log(df_aligned[price_col] / df_aligned[price_col].shift(1)).values[target_start:target_end]
    
    # Handle NaN in targets/SideInfo
    valid_mask = ~np.isnan(targets) & ~np.isnan(alpha_aligned) & ~np.isnan(vol_aligned)
    features = features[valid_mask]
    targets = targets[valid_mask].astype(int)
    alpha_aligned = alpha_aligned[valid_mask]
    vol_aligned = vol_aligned[valid_mask]
    actual_returns = actual_returns[valid_mask]
    timestamps = timestamps[valid_mask]
    
    # Append regime features to signature features
    # This gives the ensemble context about current market regime
    regime_features = np.column_stack([alpha_aligned, vol_aligned])
    features_with_regime = np.hstack([features, regime_features])
    
    if verbose:
        print(f"After removing NaN: {len(features)} samples")
        print(f"Appended regime features (alpha, volatility): {features_with_regime.shape[1]} total features")
    
    # ========================================
    # Step 6: Nested Walk-Forward Validation
    # ========================================
    if verbose:
        print("\n" + "-" * 40)
        print("STEP 6: Nested Purged Walk-Forward Validation")
        print("Using regime-augmented features (signatures + alpha + volatility)")
        print("-" * 40)
    
    # Initialize classifier ensemble instance factory
    model_factory = lambda: BaseEnsemble()
    
    # Side info for Meta-Labeler (used for meta-features, not ensemble input)
    side_info = {
        'hurst': alpha_aligned,
        'volatility': vol_aligned
    }
    
    # Cross-validator
    cv = PurgedWalkForwardCV(
        n_splits=config.N_SPLITS,
        purge_window=config.PURGE_WINDOW,
        expanding=True
    )
    
    # Run nested validation with regime-augmented features
    val_results = nested_walk_forward_validation(
        X=features_with_regime,  # Features now include regime context
        y=targets,
        side_info=side_info,
        model_factory=model_factory,
        cv=cv,
        meta_labeler_params=config.META_LABELER_PARAMS,
        verbose=verbose
    )
    
    # ========================================
    # Step 7: Backtesting
    # ========================================
    if verbose:
        print("\n" + "-" * 40)
        print("STEP 7: Backtesting with Regime Filter")
        print("-" * 40)
    
    backtester = Backtester(
        hurst_threshold=config.DFA_ALPHA_THRESHOLD, # DFA Alpha trending threshold
        meta_threshold=0.5
    )
    
    # Get aligned data for backtest
    test_indices = val_results['test_indices']
    
    backtest_results, metrics = backtester.run_backtest(
        predictions=val_results['predictions'],
        actual_returns=actual_returns[test_indices],
        hurst=alpha_aligned[test_indices],
        meta_proba=val_results['meta_probabilities'],
        timestamps=timestamps[test_indices]
    )
    
    backtester.print_metrics(metrics)
    
    # ========================================
    # Step 8: Save Results
    # ========================================
    if save_results:
        results_path = 'backtest_results.csv'
        backtest_results.to_csv(results_path)
        if verbose:
            print(f"\nResults saved to: {results_path}")
    
    if verbose:
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'data': df_aligned,
        'features': features,
        'targets': targets,
        'validation': val_results,
        'backtest': backtest_results,
        'metrics': metrics
    }


def run_quick_test():
    """
    Run a quick test with synthetic data to verify pipeline logic.
    """
    print("=" * 60)
    print("GME QUICK TEST (Synthetic Data - Multi-Scale)")
    print("=" * 60)
    
    np.random.seed(config.RANDOM_SEED)
    
    # Generate synthetic 5D data
    n_samples = 1000
    
    # Fractionally differentiated price (stationary)
    price_fracdiff = np.random.randn(n_samples) * 0.01
    
    # CVD
    cvd = np.cumsum(np.random.randn(n_samples) * 100)
    cvd_norm = (cvd - cvd.mean()) / cvd.std()
    
    # Synthetic OI
    oi = np.cumsum(np.random.randn(n_samples) * 50) + 1000
    oi_norm = (oi - oi.mean()) / oi.std()
    
    # Synthetic Funding Rate
    funding = np.random.randn(n_samples) * 0.0001
    
    # Synthetic Liquidations (5th dimension)
    liqs = np.random.randn(n_samples) * 1000
    liqs_norm = (liqs - liqs.mean()) / liqs.std()
    
    print(f"\nGenerated {n_samples} synthetic samples (5D)")
    print(f"Multi-scale windows: {config.MULTI_SCALE_WINDOWS}")
    
    # Generate multi-scale signature features
    print("\nGenerating multi-scale signature features...")
    features = generate_multiscale_signature_features(
        price_fracdiff=price_fracdiff,
        cvd=cvd_norm,
        oi=oi_norm,
        funding_rate=funding,
        liquidations=liqs_norm,
        window_sizes=config.MULTI_SCALE_WINDOWS,
        degree=config.SIGNATURE_DEGREE
    )
    print(f"Multi-scale signature shape: {features.shape}")
    
    # Alignment offset
    max_window = max(config.MULTI_SCALE_WINDOWS)
    
    # Generate synthetic labels
    n_feat_samples = len(features)
    labels = np.random.choice([-1, 0, 1], size=n_feat_samples, p=[0.3, 0.2, 0.5])
    
    # Synthetic DFA values and volatility
    hurst = np.random.uniform(0.8, 1.2, n_feat_samples)  # Around 1.0
    vol = np.random.uniform(0.01, 0.05, n_feat_samples)
    
    # Append regime features (as in main pipeline)
    regime_features = np.column_stack([hurst, vol])
    features_with_regime = np.hstack([features, regime_features])
    print(f"Features with regime: {features_with_regime.shape}")
    
    # Run validation
    print("\nRunning Nested Walk-Forward Validation...")
    cv = PurgedWalkForwardCV(n_splits=3, purge_window=max_window, min_train_size=100)
    
    side_info = {'hurst': hurst, 'volatility': vol}
    
    # Use factory
    model_factory = lambda: BaseEnsemble()
    
    val_results = nested_walk_forward_validation(
        features_with_regime, labels, side_info,
        model_factory, cv=cv, meta_labeler_params={'n_estimators': 10, 'max_depth': 2},
        verbose=True
    )
    
    print("\n[OK] Quick test completed successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geometric Manifold Ensemble BTC/USD Classification Framework"
    )
    parser.add_argument(
        '--test', 
        action='store_true',
        help="Run quick test with synthetic data"
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default=None,
        help="Path to CSV file with data"
    )
    parser.add_argument(
        '--no-binance',
        action='store_true',
        help="Skip Binance/CCXT and use yfinance only"
    )
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help="Suppress verbose output"
    )
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help="Don't save results to CSV"
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_quick_test()
    else:
        run_pipeline(
            data_path=args.data,
            use_binance=not args.no_binance,
            verbose=not args.quiet,
            save_results=not args.no_save
        )
