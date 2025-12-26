"""
Stress testing and sensitivity analysis for GME strategy.
Includes synthetic data generation (GBM/OU) and slippage sensitivity.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import config
from backtest import Backtester
from models import BaseEnsemble
from meta_labeler import MetaLabeler
# Assuming we can mock or use lightweight versions for stress testing
# For full integration, we would need to generate features on fly

class SyntheticDataGenerator:
    """Generate synthetic price paths using GBM and OU processes."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        
    def generate_gbm(
        self, 
        n_steps: int, 
        start_price: float = 100.0, 
        mu: float = 0.0, 
        sigma: float = 0.02
    ) -> np.ndarray:
        """
        Generate Geometric Brownian Motion path.
        
        S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) * dt + sigma * dW_t)
        """
        dt = 1.0
        dw = self.rng.normal(0, np.sqrt(dt), n_steps)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * dw
        
        log_returns = drift + diffusion
        price = start_price * np.exp(np.cumsum(log_returns))
        price = np.insert(price, 0, start_price)[:-1] # shift to start at start_price
        
        return price
    
    def generate_ou(
        self, 
        n_steps: int, 
        start_price: float = 100.0, 
        theta: float = 0.1, # Mean reversion speed
        mu: float = 100.0,  # Long term mean
        sigma: float = 2.0  # Volatility
    ) -> np.ndarray:
        """
        Generate Ornstein-Uhlenbeck Process path.
        
        dX_t = theta * (mu - X_t) * dt + sigma * dW_t
        """
        dt = 1.0
        prices = np.zeros(n_steps)
        prices[0] = start_price
        
        for t in range(1, n_steps):
            # dX_t = theta * (mu - X_{t-1}) * dt + sigma * dW_t
            dw = self.rng.normal(0, np.sqrt(dt))
            dx = theta * (mu - prices[t-1]) * dt + sigma * dw
            prices[t] = prices[t-1] + dx
            
        return prices


class StressTester:
    """Run stress tests on strategy."""
    
    def __init__(self, strategy_components: Dict):
        """
        Args:
            strategy_components: Dict containing trained models and configs
        """
        self.base_ensemble = strategy_components.get('base_ensemble')
        self.meta_labeler = strategy_components.get('meta_labeler')
        
    def run_synthetic_test(
        self, 
        n_paths: int = 100, 
        path_type: str = 'gbm',
        path_params: Dict = {}
    ) -> pd.DataFrame:
        """
        Run strategy on N synthetic paths.
        
        Returns:
            DataFrame of performance metrics for each path
        """
        generator = SyntheticDataGenerator()
        results = []
        
        for i in tqdm(range(n_paths), desc=f"Running {path_type} stress test"):
            # 1. Generate path
            if path_type == 'gbm':
                prices = generator.generate_gbm(n_steps=config.WINDOW_SIZE * 5, **path_params)
            else:
                prices = generator.generate_ou(n_steps=config.WINDOW_SIZE * 5, **path_params)
            
            # 2. Extract features (Mocking features for speed/viability in this context)
            # In a real scenario, we'd need to compute signatures on these prices
            # For this test, we assume the model logic filters out random noise
            
            # Since generating features is heavy, we simulate the "filtering" check
            # We assume for random data:
            # - Hurst should be around 0.5 (random) or < 0.5 (mean revert)
            # - Meta-labeler should have low confidence
            
            # TODO: Full feature generation integration
            # For now, we simulate the logic:
            
            # Verify Hurst Filter
            # Calculate Hurst on this path
            # If Hurst is not trending, Strategy Check Pass
            
            pass 
            
        return pd.DataFrame(results)


class SensitivityAnalyzer:
    """Analyze sensitivity to slippage and costs."""
    
    def __init__(self, backtester: Backtester):
        self.backtester = backtester
        
    def run_slippage_sweep(
        self, 
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        hurst: np.ndarray,
        meta_proba: np.ndarray,
        costs_bps: List[int] = [2, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        Test strategy at different transaction cost levels.
        """
        sweep_results = []
        
        for bps in costs_bps:
            cost = bps / 10000.0
            self.backtester.transaction_cost = cost
            
            _, metrics = self.backtester.run_backtest(
                predictions=predictions,
                actual_returns=actual_returns,
                hurst=hurst,
                meta_proba=meta_proba
            )
            
            metrics['cost_bps'] = bps
            sweep_results.append(metrics)
            
        return pd.DataFrame(sweep_results)
    
    def calculate_capacity_threshold(
        self, 
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        hurst: np.ndarray,
        meta_proba: np.ndarray,
        min_sharpe: float = 1.0
    ) -> float:
        """
        Find maximum slippage (in bps) before Sharpe Ratio drops below threshold.
        """
        low = 0
        high = 100 # 100 bps max search
        
        for _ in range(10): # Binary search
            mid = (low + high) / 2
            self.backtester.transaction_cost = mid / 10000.0
            
            _, metrics = self.backtester.run_backtest(
                predictions=predictions,
                actual_returns=actual_returns,
                hurst=hurst,
                meta_proba=meta_proba
            )
            
            if metrics['sharpe_ratio'] < min_sharpe:
                high = mid
            else:
                low = mid
                
        return low

if __name__ == "__main__":
    # Test generation
    gen = SyntheticDataGenerator()
    gbm = gen.generate_gbm(100)
    ou = gen.generate_ou(100)
    
    print(f"GBM path len: {len(gbm)}")
    print(f"OU path len: {len(ou)}")
    
    # Test sensitivity
    analyzer = SensitivityAnalyzer(Backtester())
    # Mock data
    n = 1000
    preds = np.random.randint(-1, 2, n)
    rets = np.random.randn(n) * 0.01
    hurst = np.random.rand(n)
    meta = np.random.rand(n)
    
    print("\nRunning Slippage Sweep...")
    df = analyzer.run_slippage_sweep(preds, rets, hurst, meta)
    print(df[['cost_bps', 'sharpe_ratio', 'total_return']])
    
    print("\nCapacity Threshold:")
    cap = analyzer.calculate_capacity_threshold(preds, rets, hurst, meta)
    print(f"Max slippage for SR > 1.0: {cap:.2f} bps")
