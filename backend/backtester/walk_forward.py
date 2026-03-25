"""
Walk-Forward Validation Module for QuantSignal Arena.

Implements walk-forward analysis for out-of-sample testing of trading signals.
"""

import logging
import pandas as pd
from typing import List, Tuple, Optional
from .signal_base import SignalBase
from .engine import BacktestEngine
from .metrics import MetricsCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation splitter for time series backtesting.
    
    Supports both rolling and anchored window strategies with
    configurable gap periods to prevent look-ahead bias.
    """
    
    def __init__(self) -> None:
        """Initialize walk-forward validator."""
        logger.info("WalkForwardValidator initialized")
    
    def split(
        self,
        data: pd.DataFrame,
        train_size: int,
        test_size: int,
        gap: int = 0,
        anchored: bool = False
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into train/test folds for walk-forward validation.
        
        Args:
            data: Time series DataFrame with DatetimeIndex
            train_size: Training window size in days
            test_size: Test window size in days
            gap: Buffer days between train end and test start (default 0)
            anchored: If True, train window starts from day 0 and grows.
                     If False, rolling window slides forward.
        
        Returns:
            List of (train_df, test_df) tuples with non-overlapping test windows
            
        Raises:
            ValueError: If parameters are invalid or insufficient data
        """
        # Validate inputs
        self._validate_split_parameters(data, train_size, test_size, gap)
        
        logger.info(f"Splitting data: train_size={train_size}, test_size={test_size}, "
                   f"gap={gap}, anchored={anchored}, total_length={len(data)}")
        
        folds = []
        n = len(data)
        
        if anchored:
            # Anchored: train window starts at 0 and grows
            test_start = train_size + gap
            
            while test_start + test_size <= n:
                train_end = test_start - gap
                test_end = test_start + test_size
                
                train_df = data.iloc[0:train_end]
                test_df = data.iloc[test_start:test_end]
                
                folds.append((train_df, test_df))
                
                # Move to next fold
                test_start += test_size
        else:
            # Rolling: train window slides forward
            train_start = 0
            
            while train_start + train_size + gap + test_size <= n:
                train_end = train_start + train_size
                test_start = train_end + gap
                test_end = test_start + test_size
                
                train_df = data.iloc[train_start:train_end]
                test_df = data.iloc[test_start:test_end]
                
                folds.append((train_df, test_df))
                
                # Move to next fold
                train_start += test_size
        
        logger.info(f"Created {len(folds)} walk-forward folds")
        
        return folds
    
    def run_walk_forward(
        self,
        signal: SignalBase,
        data: pd.DataFrame,
        train_size: int,
        test_size: int,
        gap: int = 0,
        anchored: bool = False,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.02
    ) -> List[dict]:
        """
        Run walk-forward validation with backtesting on each fold.
        
        Args:
            signal: Trading signal instance
            data: Historical OHLCV DataFrame
            train_size: Training window size in days
            test_size: Test window size in days
            gap: Buffer days between train and test (default 0)
            anchored: Use anchored (True) or rolling (False) windows
            initial_capital: Starting capital for each fold
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            risk_free_rate: Annual risk-free rate for metrics
            
        Returns:
            List of dictionaries, one per fold, containing:
            - fold_number: Fold index (0-based)
            - train_start: Training period start date
            - train_end: Training period end date
            - test_start: Test period start date
            - test_end: Test period end date
            - All metrics from MetricsCalculator
        """
        logger.info(f"Running walk-forward validation for signal '{signal.name}'")
        
        # Get folds
        folds = self.split(data, train_size, test_size, gap, anchored)
        
        # Initialize engine and metrics calculator
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )
        metrics_calc = MetricsCalculator(risk_free_rate=risk_free_rate)
        
        results = []
        
        for fold_idx, (train_df, test_df) in enumerate(folds):
            logger.info(f"Processing fold {fold_idx + 1}/{len(folds)}")
            
            # Run backtest on test window only
            backtest_results = engine.run_backtest(
                signal=signal,
                ohlcv_data=test_df,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage
            )
            
            # Calculate metrics
            metrics = metrics_calc.calculate_metrics(
                portfolio_returns=backtest_results['metrics_input'],
                risk_free_rate=risk_free_rate
            )
            
            # Combine fold info with metrics
            fold_result = {
                'fold_number': fold_idx,
                'train_start': train_df.index[0],
                'train_end': train_df.index[-1],
                'test_start': test_df.index[0],
                'test_end': test_df.index[-1],
                **metrics  # Unpack all metrics
            }
            
            results.append(fold_result)
        
        logger.info(f"Walk-forward validation completed: {len(results)} folds processed")
        
        return results
    
    def _validate_split_parameters(
        self,
        data: pd.DataFrame,
        train_size: int,
        test_size: int,
        gap: int
    ) -> None:
        """Validate split parameters."""
        # Check train_size is positive integer
        if not isinstance(train_size, int) or train_size <= 0:
            raise ValueError(f"train_size must be a positive integer, got {train_size}")
        
        # Check test_size is positive integer
        if not isinstance(test_size, int) or test_size <= 0:
            raise ValueError(f"test_size must be a positive integer, got {test_size}")
        
        # Check gap is non-negative
        if not isinstance(gap, int) or gap < 0:
            raise ValueError(f"gap must be a non-negative integer, got {gap}")
        
        # Check data is not empty
        if len(data) == 0:
            raise ValueError("data cannot be empty")
        
        # Check sufficient data
        min_required = train_size + gap + test_size
        if len(data) < min_required:
            raise ValueError(
                f"Insufficient data: need at least {min_required} days "
                f"(train_size={train_size} + gap={gap} + test_size={test_size}), "
                f"but got {len(data)} days"
            )
