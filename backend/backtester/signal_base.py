"""
Signal Base Module for QuantSignal Arena Backtesting Engine.

Defines the abstract base class for all trading signals.
"""

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalBase(ABC):
    """
    Abstract base class for trading signals.
    
    All custom signals must inherit from this class and implement
    the generate_signals method.
    """
    
    def __init__(self, **parameters: Any) -> None:
        """
        Initialize signal with parameters.
        
        Args:
            **parameters: Signal-specific configuration parameters
        """
        self._parameters = parameters
        self.validate_parameters()
    
    @abstractmethod
    def generate_signals(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with OHLCV columns and DatetimeIndex
            
        Returns:
            Series with DatetimeIndex and signal values:
            - 1: Long position
            - 0: Neutral (no position)
            - -1: Short position
            
        Raises:
            ValueError: If signal generation fails
        """
        pass
    
    @property
    def name(self) -> str:
        """Return signal class name."""
        return self.__class__.__name__
    
    @property
    def parameters(self) -> dict[str, Any]:
        """Return signal parameters as dictionary."""
        return self._parameters.copy()
    
    def validate_parameters(self) -> None:
        """
        Validate signal parameters.
        
        Override this method to add custom parameter validation.
        
        Raises:
            ValueError: If parameters are invalid
        """
        pass
    
    def _validate_signal_output(
        self,
        signals: pd.Series,
        ohlcv_data: pd.DataFrame
    ) -> None:
        """
        Validate signal output meets requirements.
        
        Args:
            signals: Generated signal series
            ohlcv_data: Input OHLCV data
            
        Raises:
            ValueError: If signals are invalid
        """
        # Check that signals is a Series
        if not isinstance(signals, pd.Series):
            raise ValueError(f"Signal output must be a pandas Series, got {type(signals)}")
        
        # Check length matches input
        if len(signals) != len(ohlcv_data):
            raise ValueError(
                f"Signal length ({len(signals)}) must match input data length ({len(ohlcv_data)})"
            )
        
        # Check for NaN values
        if signals.isna().any():
            raise ValueError("Signal contains NaN values")
        
        # Check that all values are in {-1, 0, 1}
        valid_values = {-1, 0, 1}
        unique_values = set(signals.unique())
        invalid_values = unique_values - valid_values
        if invalid_values:
            raise ValueError(
                f"Signal contains invalid values. Expected {valid_values}, found: {invalid_values}"
            )
        
        # Check that index is DatetimeIndex
        if not isinstance(signals.index, pd.DatetimeIndex):
            raise ValueError(f"Signal index must be DatetimeIndex, got {type(signals.index)}")
        
        # Check that index matches input
        if not signals.index.equals(ohlcv_data.index):
            raise ValueError("Signal index must match input data index")
    
    def __call__(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """
        Generate and validate signals with error wrapping.
        
        Args:
            ohlcv_data: DataFrame with OHLCV columns and DatetimeIndex
            
        Returns:
            Series with validated signal values
            
        Raises:
            ValueError: If signal generation or validation fails
        """
        try:
            signals = self.generate_signals(ohlcv_data)
            self._validate_signal_output(signals, ohlcv_data)
            return signals
        except Exception as e:
            raise ValueError(f"Signal '{self.name}' failed: {e}") from e


class MomentumSignal(SignalBase):
    """
    Simple momentum signal based on N-day returns.
    
    Goes long (1) when N-day return is positive, short (-1) when negative.
    """
    
    def __init__(self, lookback_period: int = 20, **kwargs: Any) -> None:
        """
        Initialize momentum signal.
        
        Args:
            lookback_period: Number of days to calculate momentum (default 20)
            **kwargs: Additional parameters
        """
        self.lookback_period = lookback_period
        super().__init__(lookback_period=lookback_period, **kwargs)
    
    def validate_parameters(self) -> None:
        """Validate momentum signal parameters."""
        if not isinstance(self.lookback_period, int):
            raise ValueError(f"lookback_period must be an integer, got {type(self.lookback_period)}")
        
        if self.lookback_period <= 0:
            raise ValueError(f"lookback_period must be positive, got {self.lookback_period}")
        
        if self.lookback_period > 252:
            raise ValueError(f"lookback_period must be <= 252 days, got {self.lookback_period}")
    
    def generate_signals(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals based on N-day returns.
        
        Args:
            ohlcv_data: DataFrame with OHLCV columns and DatetimeIndex
            
        Returns:
            Series with signal values: 1 (long), -1 (short), 0 (neutral)
        """
        # Calculate N-day returns
        returns = ohlcv_data['Close'].pct_change(periods=self.lookback_period)
        
        # Generate signals
        signals = pd.Series(0, index=ohlcv_data.index, dtype=np.int8)
        signals[returns > 0] = 1
        signals[returns < 0] = -1
        
        # First N days will have NaN returns, set to neutral
        signals.iloc[:self.lookback_period] = 0
        
        return signals
