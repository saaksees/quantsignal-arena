"""
Backtesting Engine Module for QuantSignal Arena.

Executes vectorized backtests using vectorbt for portfolio simulation.
"""

from typing import Optional, Callable, Any
import logging
import pandas as pd
import numpy as np
import vectorbt as vbt
from .signal_base import SignalBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Vectorized backtesting engine using vectorbt.
    
    Simulates portfolio performance based on trading signals
    with configurable costs and constraints.
    """
    
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        freq: str = "D"
    ) -> None:
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting portfolio value in USD
            commission: Commission rate per trade (0.001 = 0.1%)
            slippage: Slippage rate per trade (0.0005 = 0.05%)
            freq: Rebalancing frequency ("D" for daily)
            
        Raises:
            ValueError: If parameters are invalid
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.freq = freq
        
        self._validate_parameters()
        logger.info(f"BacktestEngine initialized with capital={initial_capital}, commission={commission}, slippage={slippage}")
    
    def run_backtest(
        self,
        signal: SignalBase,
        ohlcv_data: pd.DataFrame,
        initial_capital: Optional[float] = None,
        commission: Optional[float] = None,
        slippage: Optional[float] = None,
        freq: Optional[str] = None,
        fee_model: Optional[Callable] = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Execute backtest for a signal on OHLCV data.
        
        Args:
            signal: Trading signal instance
            ohlcv_data: Historical OHLCV DataFrame
            initial_capital: Override initial capital (optional)
            commission: Override commission rate (optional)
            slippage: Override slippage rate (optional)
            freq: Override frequency (optional)
            fee_model: Optional custom fee calculation function
            **kwargs: Additional vectorbt portfolio parameters
            
        Returns:
            Dictionary containing:
            - portfolio_value: Series of portfolio values over time
            - returns: Series of daily returns
            - trades: DataFrame of executed trades
            - positions: DataFrame of positions over time
            - signal_series: Series of signal values used
            - metrics_input: Daily returns for metrics calculation
            
        Raises:
            ValueError: If signal or data is invalid
        """
        # Use instance defaults or override
        capital = initial_capital if initial_capital is not None else self.initial_capital
        comm = commission if commission is not None else self.commission
        slip = slippage if slippage is not None else self.slippage
        frequency = freq if freq is not None else self.freq
        
        # Validate overridden parameters
        if capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {capital}")
        if comm < 0:
            raise ValueError(f"commission must be non-negative, got {comm}")
        if slip < 0:
            raise ValueError(f"slippage must be non-negative, got {slip}")
        
        logger.info(f"Starting backtest for signal '{signal.name}' with {len(ohlcv_data)} data points")
        
        # Validate inputs
        self._validate_inputs(signal, ohlcv_data)
        
        # Generate signal series
        signals = self._generate_signal_series(signal, ohlcv_data)
        
        # Create portfolio
        portfolio = self._create_portfolio(
            signals=signals,
            prices=ohlcv_data['Close'],
            capital=capital,
            commission=comm,
            slippage=slip,
            fee_model=fee_model,
            **kwargs
        )
        
        # Extract results
        results = self._extract_results(portfolio, signals)
        
        logger.info(f"Backtest completed for signal '{signal.name}'")
        
        return results
    
    def _validate_inputs(
        self,
        signal: SignalBase,
        ohlcv_data: pd.DataFrame
    ) -> None:
        """Validate signal and OHLCV data before backtesting."""
        # Check signal is SignalBase instance using MRO name check
        base_class_names = [c.__name__ for c in type(signal).__mro__]
        if "SignalBase" not in base_class_names:
            raise ValueError(f"signal must be a SignalBase instance, got {type(signal)}")
        
        # Check OHLCV data is DataFrame
        if not isinstance(ohlcv_data, pd.DataFrame):
            raise ValueError(f"ohlcv_data must be a pandas DataFrame, got {type(ohlcv_data)}")
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in ohlcv_data.columns]
        if missing_columns:
            raise ValueError(f"ohlcv_data missing required columns: {missing_columns}")
        
        # Check data is not empty
        if len(ohlcv_data) == 0:
            raise ValueError("ohlcv_data cannot be empty")
        
        # Check index is DatetimeIndex
        if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
            raise ValueError(f"ohlcv_data index must be DatetimeIndex, got {type(ohlcv_data.index)}")
    
    def _generate_signal_series(
        self,
        signal: SignalBase,
        ohlcv_data: pd.DataFrame
    ) -> pd.Series:
        """Generate and validate signal series."""
        try:
            # Use the __call__ method which includes validation
            signals = signal(ohlcv_data)
            
            # Additional validation for NaN
            if signals.isna().any():
                raise ValueError("Signal contains NaN values")
            
            # Check values are in valid range
            valid_values = {-1, 0, 1}
            unique_values = set(signals.unique())
            invalid_values = unique_values - valid_values
            if invalid_values:
                raise ValueError(f"Signal contains invalid values: {invalid_values}. Expected {valid_values}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            raise
    
    def _create_portfolio(
        self,
        signals: pd.Series,
        prices: pd.Series,
        capital: float,
        commission: float,
        slippage: float,
        fee_model: Optional[Callable],
        **kwargs: Any
    ) -> vbt.Portfolio:
        """Create vectorbt portfolio from signals and prices."""
        # Convert signals to entries and exits
        # Signal: 1 = long, -1 = short, 0 = neutral
        
        # For long positions: enter when signal becomes 1, exit when it becomes 0 or -1
        # For short positions: enter when signal becomes -1, exit when it becomes 0 or 1
        
        # Calculate position changes
        signal_diff = signals.diff()
        
        # Long entries: signal changes to 1
        long_entries = (signals == 1) & (signal_diff != 0)
        long_entries.iloc[0] = (signals.iloc[0] == 1)  # Handle first position
        
        # Long exits: signal changes from 1 to something else
        long_exits = (signals.shift(1) == 1) & (signals != 1)
        
        # Short entries: signal changes to -1
        short_entries = (signals == -1) & (signal_diff != 0)
        short_entries.iloc[0] = (signals.iloc[0] == -1)  # Handle first position
        
        # Short exits: signal changes from -1 to something else
        short_exits = (signals.shift(1) == -1) & (signals != -1)
        
        # Calculate total fees (commission + slippage)
        total_fees = commission + slippage
        
        # Create portfolio using vectorbt
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=capital,
            fees=total_fees,
            freq=self.freq,
            **kwargs
        )
        
        return portfolio
    
    def _extract_results(
        self,
        portfolio: vbt.Portfolio,
        signals: pd.Series
    ) -> dict[str, Any]:
        """Extract results from vectorbt portfolio object."""
        # Get portfolio value over time
        portfolio_value = portfolio.value()
        
        # Get returns
        returns = portfolio.returns()
        
        # Get trades
        trades = portfolio.trades.records_readable
        
        # Get positions (asset value over time)
        positions = portfolio.asset_value()
        
        # Create results dictionary
        results = {
            'portfolio_value': portfolio_value,
            'returns': returns,
            'trades': trades,
            'positions': positions,
            'signal_series': signals,
            'metrics_input': returns  # Daily returns for metrics calculation
        }
        
        return results
    
    def _validate_parameters(self) -> None:
        """Validate engine parameters."""
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        
        if self.commission < 0:
            raise ValueError(f"commission must be non-negative, got {self.commission}")
        
        if self.slippage < 0:
            raise ValueError(f"slippage must be non-negative, got {self.slippage}")
