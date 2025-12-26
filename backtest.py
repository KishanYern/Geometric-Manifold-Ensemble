"""
Backtesting engine for GME trading strategy.
Updated for classification-based trading with Hurst regime filtering.

Trading Logic:
1. Base ensemble predicts: Long (1), Short (-1), Flat (0)
2. Meta-labeler vetos if confidence is low
3. Hurst filter only allows trades in trending regimes (H > 0.55)
4. Final signal is executed only if all filters pass
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import config
from regime import rolling_hurst, regime_filter


class Backtester:
    """
    Backtesting engine for classification-based trading strategy.
    
    Filters:
    1. Hurst Regime Filter: Only trade when H > trending_threshold
    2. Meta-Labeler Filter: Only trade when meta-confidence > threshold
    3. Direction: Follow base ensemble's Long/Short/Flat signal
    """
    
    def __init__(
        self,
        hurst_threshold: float = config.HURST_TRENDING_THRESHOLD,
        meta_threshold: float = 0.5,
        transaction_cost: float = 0.0005  # 5 bps per trade
    ):
        """
        Initialize backtester.
        
        Args:
            hurst_threshold: Minimum H to allow trading
            meta_threshold: Minimum meta-labeler probability to act
            transaction_cost: Cost per trade as fraction of position
        """
        self.hurst_threshold = hurst_threshold
        self.meta_threshold = meta_threshold
        self.transaction_cost = transaction_cost
    
    def generate_signals(
        self,
        predictions: np.ndarray,
        hurst: np.ndarray,
        meta_proba: Optional[np.ndarray] = None,
        apply_hurst_filter: bool = True,
        apply_meta_filter: bool = True
    ) -> np.ndarray:
        """
        Generate trading signals with regime and meta filtering.
        
        Signal values:
        - 1: Long position
        - -1: Short position
        - 0: Flat (no position)
        
        Args:
            predictions: Base ensemble predictions (1, -1, 0)
            hurst: Hurst exponent values for each time point
            meta_proba: Meta-labeler probabilities (optional)
            apply_hurst_filter: Whether to apply Hurst regime filter
            apply_meta_filter: Whether to apply meta-labeler filter
            
        Returns:
            Array of trading signals
        """
        signals = predictions.copy().astype(float)
        
        # Apply Hurst regime filter
        if apply_hurst_filter:
            # Only trade in trending regimes
            non_trending = hurst < self.hurst_threshold
            signals[non_trending] = 0
        
        # Apply meta-labeler filter
        if apply_meta_filter and meta_proba is not None:
            low_confidence = meta_proba < self.meta_threshold
            signals[low_confidence] = 0
        
        return signals.astype(int)
    
    def compute_returns(
        self,
        signals: np.ndarray,
        actual_returns: np.ndarray,
        include_costs: bool = True
    ) -> pd.DataFrame:
        """
        Compute strategy returns based on signals.
        
        Args:
            signals: Trading signals (1 for long, -1 for short, 0 for flat)
            actual_returns: Actual log returns that were realized
            include_costs: Whether to include transaction costs
            
        Returns:
            DataFrame with strategy returns and metrics
        """
        # Strategy returns = signal * actual return
        strategy_returns = signals * actual_returns
        
        # Transaction costs on position changes
        if include_costs:
            position_changes = np.abs(np.diff(signals, prepend=0))
            costs = position_changes * self.transaction_cost
            strategy_returns -= costs
        
        # Buy and hold returns (long only)
        buy_hold_returns = actual_returns.copy()
        
        return pd.DataFrame({
            'signal': signals,
            'actual_return': actual_returns,
            'strategy_return': strategy_returns,
            'buy_hold_return': buy_hold_returns
        })
    
    def compute_metrics(
        self,
        returns: pd.Series,
        periods_per_year: int = 365
    ) -> Dict[str, float]:
        """
        Compute comprehensive performance metrics.
        
        Args:
            returns: Series of strategy returns
            periods_per_year: Number of periods per year (365 for daily)
            
        Returns:
            Dictionary of performance metrics
        """
        if len(returns) == 0:
            return {'error': 'No returns to analyze'}
        
        # Basic statistics
        total_return = np.expm1(returns.sum())  # Convert log to simple
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Annualized metrics
        ann_return = mean_return * periods_per_year
        ann_volatility = std_return * np.sqrt(periods_per_year)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if ann_volatility > 0:
            sharpe_ratio = ann_return / ann_volatility
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative = returns.cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (returns > 0).sum()
        losing_trades = (returns < 0).sum()
        total_trades = winning_trades + losing_trades
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0.0
        
        # Profit factor
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        
        if gross_losses > 0:
            profit_factor = gross_profits / gross_losses
        else:
            profit_factor = float('inf') if gross_profits > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.sqrt((negative_returns ** 2).mean())
            ann_downside = downside_std * np.sqrt(periods_per_year)
            sortino_ratio = ann_return / ann_downside if ann_downside > 0 else 0.0
        else:
            sortino_ratio = float('inf') if ann_return > 0 else 0.0
        
        return {
            'total_return': total_return,
            'annual_return': ann_return,
            'annual_volatility': ann_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }
    
    def run_backtest(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        hurst: np.ndarray,
        meta_proba: Optional[np.ndarray] = None,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Run complete backtest.
        
        Args:
            predictions: Base ensemble predictions (1, -1, 0)
            actual_returns: Actual returns that occurred
            hurst: Hurst exponent values
            meta_proba: Optional meta-labeler probabilities
            timestamps: Optional datetime index for results
            
        Returns:
            Tuple of (results_df, metrics_dict)
        """
        # Generate filtered signals
        signals = self.generate_signals(
            predictions=predictions,
            hurst=hurst,
            meta_proba=meta_proba,
            apply_hurst_filter=True,
            apply_meta_filter=(meta_proba is not None)
        )
        
        # Compute returns
        results = self.compute_returns(signals, actual_returns)
        
        # Add Hurst and meta info
        results['hurst'] = hurst
        if meta_proba is not None:
            results['meta_proba'] = meta_proba
        
        # Add regime classification
        results['regime'] = np.where(
            hurst > config.HURST_TRENDING_THRESHOLD, 'trending',
            np.where(hurst < config.HURST_MEAN_REVERT_THRESHOLD, 'mean_revert', 'random')
        )
        
        # Set index if timestamps provided
        if timestamps is not None:
            results.index = timestamps
        
        # Compute metrics
        strategy_metrics = self.compute_metrics(results['strategy_return'])
        buy_hold_metrics = self.compute_metrics(results['buy_hold_return'])
        
        # Add comparison metrics
        strategy_metrics['buy_hold_return'] = buy_hold_metrics['total_return']
        strategy_metrics['alpha'] = strategy_metrics['total_return'] - buy_hold_metrics['total_return']
        
        # Calculate regime statistics
        regime_counts = pd.Series(results['regime']).value_counts()
        strategy_metrics['pct_trending'] = regime_counts.get('trending', 0) / len(results)
        strategy_metrics['pct_mean_revert'] = regime_counts.get('mean_revert', 0) / len(results)
        strategy_metrics['pct_random'] = regime_counts.get('random', 0) / len(results)
        
        # Trade filter statistics
        strategy_metrics['trades_taken'] = (signals != 0).sum()
        strategy_metrics['trades_filtered'] = ((predictions != 0) & (signals == 0)).sum()
        
        return results, strategy_metrics
    
    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print formatted performance metrics."""
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        
        print("\n📈 Returns:")
        print(f"  Total Return:     {metrics['total_return']:>10.2%}")
        print(f"  Annual Return:    {metrics['annual_return']:>10.2%}")
        print(f"  Buy & Hold:       {metrics.get('buy_hold_return', 0):>10.2%}")
        print(f"  Alpha:            {metrics.get('alpha', 0):>10.2%}")
        
        print("\n📊 Risk Metrics:")
        print(f"  Annual Volatility:{metrics['annual_volatility']:>10.2%}")
        print(f"  Max Drawdown:     {metrics['max_drawdown']:>10.2%}")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:    {metrics['sortino_ratio']:>10.2f}")
        
        print("\n🎯 Trade Statistics:")
        print(f"  Total Trades:     {metrics['total_trades']:>10.0f}")
        print(f"  Win Rate:         {metrics['win_rate']:>10.2%}")
        print(f"  Profit Factor:    {metrics['profit_factor']:>10.2f}")
        print(f"  Trades Taken:     {metrics.get('trades_taken', 0):>10.0f}")
        print(f"  Trades Filtered:  {metrics.get('trades_filtered', 0):>10.0f}")
        
        print("\n🔄 Regime Statistics:")
        print(f"  Trending:         {metrics.get('pct_trending', 0):>10.1%}")
        print(f"  Mean-Reverting:   {metrics.get('pct_mean_revert', 0):>10.1%}")
        print(f"  Random Walk:      {metrics.get('pct_random', 0):>10.1%}")


if __name__ == "__main__":
    # Test backtester with classification signals
    np.random.seed(42)
    
    n_samples = 500
    
    # Simulate predictions and actual returns
    # Create some correlation between predictions and returns for realistic test
    true_trend = np.sin(np.linspace(0, 8 * np.pi, n_samples)) * 0.02
    actual_returns = true_trend + np.random.randn(n_samples) * 0.01
    
    # Predictions with some accuracy
    predictions = np.sign(true_trend + np.random.randn(n_samples) * 0.01)
    predictions = predictions.astype(int)
    
    # Simulate Hurst exponent (varying between 0.3 and 0.7)
    hurst = 0.5 + 0.15 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    hurst += np.random.randn(n_samples) * 0.05
    
    # Simulate meta-labeler probabilities
    meta_proba = 0.5 + 0.3 * np.random.rand(n_samples)
    
    print("Testing Classification Backtester")
    print("=" * 50)
    
    backtester = Backtester(hurst_threshold=0.55, meta_threshold=0.5)
    
    results, metrics = backtester.run_backtest(
        predictions=predictions,
        actual_returns=actual_returns,
        hurst=hurst,
        meta_proba=meta_proba
    )
    
    print(f"\nResults shape: {results.shape}")
    print(f"Columns: {list(results.columns)}")
    
    backtester.print_metrics(metrics)
