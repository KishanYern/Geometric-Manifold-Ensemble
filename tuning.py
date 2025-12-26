"""
Hyperparameter tuning for GME Strategy.
Focuses on optimizing:
1. Regime Filter Threshold (DFA Alpha)
2. Triple Barrier Parameters
3. Feature Window Size
4. Model Parameters (LightGBM)

Uses RandomizedSearchCV-style approach with PurgedWalkForwardCV.
"""

import numpy as np
import pandas as pd
import itertools
from typing import Dict, List, Any
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import config
from data_loader import load_btc_data, prepare_dataset
from features import generate_signature_features
from models import BaseEnsemble
from validation import PurgedWalkForwardCV, nested_walk_forward_validation
from backtest import Backtester
from fractional_diff import find_optimal_d
from labeling import compute_daily_volatility, triple_barrier_labels
from regime import rolling_dfa

# Define Search Space
PARAM_GRID = {
    'dfa_threshold': [1.3, 1.45, 1.6],
    'barrier_upper': [1.5, 2.0, 3.0],
    'barrier_time': [3, 5, 10],
    'window_size': [14, 20, 30],
    'lgbm_lr': [0.01, 0.05, 0.1]
}

def load_and_prep_data(data_path: str = None):
    """Load data once to reuse across trials."""
    print("Loading data...")
    df_raw = load_btc_data(filepath=data_path)
    print(f"DEBUG: df_raw len: {len(df_raw)}")
    df = prepare_dataset(df_raw)
    
    # Restore volume for CVD
    if 'Volume' in df_raw.columns:
        df['volume'] = df_raw['Volume']
    elif 'volume' in df_raw.columns:
        df['volume'] = df_raw['volume']
    
    # Fracdiff
    price_col = 'close' if 'close' in df.columns else 'price'
    optimal_d, price_fracdiff = find_optimal_d(
        df[price_col],
        d_range=config.FRACDIFF_D_RANGE,
        target_pvalue=config.FRACDIFF_ADF_PVALUE,
        threshold=config.FRACDIFF_THRESHOLD
    )
    
    print(f"DEBUG: Optimal d: {optimal_d:.4f}")
    
    # Align other columns with fracdiff output
    align_start = len(df) - len(price_fracdiff)
    if align_start > 0:
         print(f"DEBUG: Dropped {align_start} rows due to FFD (d={optimal_d:.4f})")
    
    df_aligned = df.iloc[align_start:].copy()
    df_aligned['price_fracdiff'] = price_fracdiff.values
    
    # Normalize CVD/OI if present
    if 'cvd' in df_aligned.columns: 
        df_aligned['cvd_normalized'] = (df_aligned['cvd'] - df_aligned['cvd'].mean()) / df_aligned['cvd'].std()
    elif 'volume' in df_aligned.columns:
         cvd = df_aligned['volume'].cumsum()
         df_aligned['cvd_normalized'] = (cvd - cvd.mean()) / cvd.std()
    
    # Compute Volatility (base)
    volatility = compute_daily_volatility(
        df_aligned[price_col],
        window=config.VOLATILITY_WINDOW
    )
    df_aligned['volatility'] = volatility
    
    # Compute DFA (base)
    # Note: Window size for DFA might also be a param, but keeping fixed for now
    alpha_series = rolling_dfa(df_aligned[price_col], window=config.HURST_WINDOW)
    df_aligned['hurst'] = alpha_series
    
    return df_aligned

def run_trial(
    df: pd.DataFrame, 
    params: Dict[str, Any],
    cv: PurgedWalkForwardCV
) -> Dict[str, float]:
    """Execute one trial of the pipeline with given params."""
    
    price_col = 'close' if 'close' in df.columns else 'price'
    
    # 1. Generate Labels based on params
    labels, _ = triple_barrier_labels(
        df[price_col],
        df['volatility'],
        upper_mult=params['barrier_upper'],
        lower_mult=config.BARRIER_LOWER_MULT, # Fixed stop loss
        max_holding_period=params['barrier_time']
    )
    
    # 2. Generate Features based on params
    # Note: Window size changes feature length, so we must be careful with alignment
    features = generate_signature_features(
        price_fracdiff=df['price_fracdiff'].values,
        cvd=df['cvd_normalized'].values if 'cvd_normalized' in df.columns else None,
        window_size=params['window_size'],
        degree=config.SIGNATURE_DEGREE
    )
    
    # Align
    target_start = params['window_size']
    target_end = target_start + len(features)
    
    if target_end > len(df):
        # Truncate if mismatch
        limit = len(df) - target_start
        features = features[:limit]
        target_end = target_start + len(features)
        
    y = labels.values[target_start:target_end]
    alpha = df['hurst'].values[target_start:target_end]
    vol = df['volatility'].values[target_start:target_end]
    returns = np.log(df[price_col] / df[price_col].shift(1)).values[target_start:target_end]
    
    # Drop NaNs
    valid_mask = ~np.isnan(y) & ~np.isnan(alpha) & ~np.isnan(vol)
    X_trial = features[valid_mask]
    y_trial = y[valid_mask].astype(int)
    alpha_trial = alpha[valid_mask]
    vol_trial = vol[valid_mask]
    rets_trial = returns[valid_mask]
    
    if len(X_trial) < 200:
        return {'sharpe_ratio': -999, 'trades': 0}
        
    # 3. Model Factory with updated params
    def model_factory():
        ens = BaseEnsemble()
        # Update LGBM params mostly
        ens.lgb.set_params(learning_rate=params['lgbm_lr'])
        return ens
        
    # 4. Validation
    # Skip Nested CV for speed in tuning, typically PurgedCV is enough
    # Or use a simplified Fold loop here manually
    
    all_preds = []
    all_rets = []
    
    for train_idx, test_idx in cv.split(X_trial):
        X_tr, X_te = X_trial[train_idx], X_trial[test_idx]
        y_tr = y_trial[train_idx]
        
        # Fit
        model = model_factory()
        model.fit(X_tr, y_tr)
        
        # Predict
        preds, _ = model.predict_ensemble(X_te)
        all_preds.extend(preds)
        all_rets.extend(rets_trial[test_idx])
        
    # 5. Backtest with Regime Filter
    backtester = Backtester(
        hurst_threshold=params['dfa_threshold'],
        transaction_cost=0.0005
    )
    
    # Since we simplified the loop above and just collected preds, 
    # we need aligned hurst values for the test sets.
    # Re-loop to get indices or just note that strict CV order is maintained
    # Ideally we'd store indices. Let's assume order is preserved for 'all_rets'.
    # But for 'hurst', we need the matching values.
    
    # Re-collect test indices for alignment
    test_indices_all = []
    for _, test_idx in cv.split(X_trial):
        test_indices_all.extend(test_idx)
        
    test_hurst = alpha_trial[test_indices_all]
    
    _, metrics = backtester.run_backtest(
        predictions=np.array(all_preds),
        actual_returns=np.array(all_rets),
        hurst=test_hurst,
        meta_proba=None # Skip meta for speed
    )
    
    return metrics

def run_tuning(n_trials: int = 10):
    """Run random search."""
    print("Starting Hyperparameter Tuning...")
    df = load_and_prep_data()
    print(f"Data loaded. Rows: {len(df)}")
    
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    
    # Generate all combinations first
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Select random sample
    if n_trials < len(all_combos):
        import random
        random.shuffle(all_combos)
        trials = all_combos[:n_trials]
    else:
        trials = all_combos
        
    print(f"Running {len(trials)} trials...")
    
    results = []
    cv = PurgedWalkForwardCV(n_splits=3, purge_window=20)
    
    for i, params in enumerate(trials):
        print(f"\nTrial {i+1}/{len(trials)}: {params}")
        try:
            metrics = run_trial(df, params, cv)
            print(f"  -> Sharpe: {metrics.get('sharpe_ratio', -99):.2f} | Trades: {metrics.get('trades_taken', 0)}")
            
            res = params.copy()
            res.update(metrics)
            results.append(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  -> Failed: {e}")
            
    # Process results
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('sharpe_ratio', ascending=False)
    
    print("\n" + "="*50)
    print("TOP 5 CONFIGURATIONS")
    print("="*50)
    print(res_df.head(5).to_string())
    
    # Save
    res_df.to_csv('tuning_results.csv', index=False)
    print("\nSaved all results to tuning_results.csv")

if __name__ == "__main__":
    run_tuning(n_trials=20)
