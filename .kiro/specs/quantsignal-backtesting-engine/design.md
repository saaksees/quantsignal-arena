# Design Document: QuantSignal Arena Backtesting Engine (Month 1)

## Overview

The QuantSignal Arena Backtesting Engine is a high-performance, vectorized backtesting system built on vectorbt and pandas. This Month 1 implementation establishes the core infrastructure for validating trading signals against historical market data with accurate performance metrics and walk-forward validation.

### Key Design Principles

1. **Vectorization First**: All computations use NumPy/pandas vectorized operations to process entire time series in batch
2. **Separation of Concerns**: Clear boundaries between data acquisition, signal generation, execution simulation, and metrics calculation
3. **Extensibility**: Abstract base classes and dependency injection enable custom signals and fee models
4. **Performance**: Target <5 seconds for 10-year daily backtests through efficient data structures and caching
5. **Type Safety**: Full type hints throughout for IDE support and runtime validation

### Technology Stack

- **Python 3.10+**: Core language with modern type hints
- **vectorbt 0.26.0**: High-performance vectorized backtesting framework
- **pandas 2.0.0**: Time series data manipulation
- **numpy 1.24.0**: Numerical computations
- **yfinance 0.2.28**: Market data acquisition
- **pyarrow 12.0.0**: Parquet file I/O for caching
- **pytest 7.4.0**: Testing framework

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User / Client Code                       │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──────────────────────────────────────────────────┐
             │                                                  │
             ▼                                                  ▼
┌────────────────────────┐                    ┌─────────────────────────┐
│   Data Loader Module   │                    │   Signal Base Module    │
│   (data_loader.py)     │                    │   (signal_base.py)      │
├────────────────────────┤                    ├─────────────────────────┤
│ - DataLoader           │                    │ - SignalBase (ABC)      │
│ - Cache management     │                    │ - Signal validation     │
│ - yfinance integration │                    │ - Parameter management  │
└────────┬───────────────┘                    └──────────┬──────────────┘
         │                                               │
         │  OHLCV DataFrame                             │  Signal Series
         │  (DatetimeIndex)                             │  (DatetimeIndex)
         │                                               │
         └───────────────────┬───────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   Backtesting Engine       │
                │   (engine.py)              │
                ├────────────────────────────┤
                │ - BacktestEngine           │
                │ - vectorbt integration     │
                │ - Portfolio simulation     │
                │ - Trade execution logic    │
                └────────┬───────────────────┘
                         │
                         │  Portfolio Returns
                         │  Trades DataFrame
                         │  Positions DataFrame
                         │
                         ▼
                ┌────────────────────────────┐
                │   Metrics Calculator       │
                │   (metrics.py)             │
                ├────────────────────────────┤
                │ - MetricsCalculator        │
                │ - Sharpe, Sortino, etc.    │
                │ - Drawdown analysis        │
                └────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│   Walk-Forward Validator (walk_forward.py)                      │
├─────────────────────────────────────────────────────────────────┤
│ - WalkForwardValidator                                          │
│ - Time series splitting                                         │
│ - Orchestrates multiple backtests across train/test windows     │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. User Request
   ↓
2. DataLoader.load_data(ticker, start_date, end_date)
   ↓
3. Check local cache (parquet files)
   ├─ Cache Hit → Return cached DataFrame
   └─ Cache Miss → yfinance.download()
   ↓
4. OHLCV DataFrame (columns: Open, High, Low, Close, Volume, Adj Close)
   ↓
5. Signal.generate_signals(ohlcv_data)
   ↓
6. Signal Series (values: -1, 0, 1)
   ↓
7. BacktestEngine.run_backtest(signal, ohlcv_data, **config)
   ↓
8. vectorbt Portfolio Simulation
   ↓
9. Results Dictionary:
   - portfolio_value: Series
   - returns: Series
   - trades: DataFrame
   - positions: DataFrame
   ↓
10. MetricsCalculator.calculate_metrics(returns)
    ↓
11. Metrics Dictionary:
    - sharpe_ratio: float
    - sortino_ratio: float
    - max_drawdown: float
    - cagr: float
    - total_return: float
    - volatility: float
    - win_rate: float
```

## Components and Interfaces

### 1. Data Loader Module (data_loader.py)

#### Purpose
Fetches historical OHLCV data from yfinance with local caching for performance.

#### Class: DataLoader

```python
from typing import Optional, List
from datetime import datetime
import pandas as pd
from pathlib import Path

class DataLoader:
    """
    Handles market data acquisition with local caching.
    
    Attributes:
        cache_dir: Directory path for storing cached parquet files
        cache_enabled: Whether to use local caching
    """
    
    def __init__(
        self,
        cache_dir: str = "./data/cache",
        cache_enabled: bool = True
    ) -> None:
        """
        Initialize DataLoader with cache configuration.
        
        Args:
            cache_dir: Path to cache directory
            cache_enabled: Enable/disable caching
        """
        pass
    
    def load_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a single ticker.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            
        Returns:
            DataFrame with DatetimeIndex and columns:
            [Open, High, Low, Close, Volume, Adj Close]
            
        Raises:
            ValueError: If ticker is invalid or dates are malformed
            ConnectionError: If yfinance API is unreachable after retries
        """
        pass
    
    def load_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> dict[str, pd.DataFrame]:
        """
        Load OHLCV data for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            
        Returns:
            Dictionary mapping ticker to OHLCV DataFrame
        """
        pass
    
    def _get_cache_path(self, ticker: str) -> Path:
        """Get cache file path for a ticker."""
        pass
    
    def _load_from_cache(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available and complete."""
        pass
    
    def _save_to_cache(
        self,
        ticker: str,
        data: pd.DataFrame
    ) -> None:
        """Save data to cache in parquet format."""
        pass
    
    def _fetch_from_yfinance(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        retries: int = 3
    ) -> pd.DataFrame:
        """Fetch data from yfinance with retry logic."""
        pass
    
    def _validate_dates(
        self,
        start_date: str,
        end_date: str
    ) -> tuple[datetime, datetime]:
        """Validate and parse date strings."""
        pass
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize OHLCV data:
        - Convert timezone to UTC
        - Forward-fill missing values (max 5 days)
        - Validate no negative prices
        """
        pass
```

### 2. Signal Base Module (signal_base.py)

#### Purpose
Abstract base class defining the interface for all trading signals.

#### Class: SignalBase

```python
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

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
        pass
```

### 3. Backtesting Engine Module (engine.py)

#### Purpose
Executes vectorized backtests using vectorbt for portfolio simulation.

#### Class: BacktestEngine

```python
from typing import Optional, Callable, Any
import pandas as pd
import numpy as np
import vectorbt as vbt
from signal_base import SignalBase

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
        pass
    
    def run_backtest(
        self,
        signal: SignalBase,
        ohlcv_data: pd.DataFrame,
        fee_model: Optional[Callable] = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Execute backtest for a signal on OHLCV data.
        
        Args:
            signal: Trading signal instance
            ohlcv_data: Historical OHLCV DataFrame
            fee_model: Optional custom fee calculation function
            **kwargs: Additional vectorbt portfolio parameters
            
        Returns:
            Dictionary containing:
            - portfolio_value: Series of portfolio values over time
            - returns: Series of daily returns
            - trades: DataFrame of executed trades
            - positions: DataFrame of positions over time
            - signal_series: Series of signal values used
            
        Raises:
            ValueError: If signal or data is invalid
        """
        pass
    
    def _validate_inputs(
        self,
        signal: SignalBase,
        ohlcv_data: pd.DataFrame
    ) -> None:
        """Validate signal and OHLCV data before backtesting."""
        pass
    
    def _generate_signal_series(
        self,
        signal: SignalBase,
        ohlcv_data: pd.DataFrame
    ) -> pd.Series:
        """Generate and validate signal series."""
        pass
    
    def _create_portfolio(
        self,
        signals: pd.Series,
        prices: pd.Series,
        fee_model: Optional[Callable],
        **kwargs: Any
    ) -> vbt.Portfolio:
        """Create vectorbt portfolio from signals and prices."""
        pass
    
    def _extract_results(
        self,
        portfolio: vbt.Portfolio,
        signals: pd.Series
    ) -> dict[str, Any]:
        """Extract results from vectorbt portfolio object."""
        pass
    
    def _validate_parameters(self) -> None:
        """Validate engine parameters."""
        pass
```

### 4. Metrics Calculator Module (metrics.py)

#### Purpose
Computes risk-adjusted performance metrics from backtest results.

#### Class: MetricsCalculator

```python
from typing import Optional
import pandas as pd
import numpy as np

class MetricsCalculator:
    """
    Calculates performance metrics from portfolio returns.
    
    Computes standard risk-adjusted metrics used in quantitative finance.
    """
    
    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """
        Initialize metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252
    
    def calculate_metrics(
        self,
        returns: pd.Series,
        portfolio_value: Optional[pd.Series] = None
    ) -> dict[str, Optional[float]]:
        """
        Calculate all performance metrics.
        
        Args:
            returns: Series of daily portfolio returns
            portfolio_value: Optional series of portfolio values over time
            
        Returns:
            Dictionary with keys:
            - sharpe_ratio: Annualized Sharpe ratio
            - sortino_ratio: Annualized Sortino ratio
            - max_drawdown: Maximum drawdown as decimal
            - cagr: Compound annual growth rate
            - total_return: Total return as decimal
            - volatility: Annualized volatility
            - win_rate: Percentage of profitable days
            
            Returns None for metrics that cannot be computed.
        """
        pass
    
    def sharpe_ratio(self, returns: pd.Series) -> Optional[float]:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Daily returns series
            
        Returns:
            Sharpe ratio or None if volatility is zero
        """
        pass
    
    def sortino_ratio(self, returns: pd.Series) -> Optional[float]:
        """
        Calculate annualized Sortino ratio.
        
        Uses only downside deviation in denominator.
        
        Args:
            returns: Daily returns series
            
        Returns:
            Sortino ratio or None if downside deviation is zero
        """
        pass
    
    def max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Daily returns series
            
        Returns:
            Maximum drawdown as positive decimal (e.g., 0.25 for 25% drawdown)
        """
        pass
    
    def cagr(self, returns: pd.Series) -> float:
        """
        Calculate compound annual growth rate.
        
        Args:
            returns: Daily returns series
            
        Returns:
            CAGR as decimal (e.g., 0.15 for 15% annual growth)
        """
        pass
    
    def total_return(self, returns: pd.Series) -> float:
        """
        Calculate total return.
        
        Args:
            returns: Daily returns series
            
        Returns:
            Total return as decimal (e.g., 0.50 for 50% gain)
        """
        pass
    
    def volatility(self, returns: pd.Series) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Daily returns series
            
        Returns:
            Annualized standard deviation of returns
        """
        pass
    
    def win_rate(self, returns: pd.Series) -> float:
        """
        Calculate win rate (percentage of profitable days).
        
        Args:
            returns: Daily returns series
            
        Returns:
            Win rate as decimal (e.g., 0.55 for 55% win rate)
        """
        pass
    
    def _validate_returns(self, returns: pd.Series) -> bool:
        """Check if returns series has sufficient data for metrics."""
        pass
```

### 5. Walk-Forward Validator Module (walk_forward.py)

#### Purpose
Implements walk-forward validation for time series cross-validation.

#### Class: WalkForwardValidator

```python
from typing import List, Tuple
import pandas as pd

class WalkForwardValidator:
    """
    Implements walk-forward validation for time series data.
    
    Splits data into sequential train/test windows to prevent
    look-ahead bias and assess out-of-sample performance.
    """
    
    def __init__(
        self,
        train_size: int,
        test_size: int,
        gap_period: int = 0,
        anchored: bool = False
    ) -> None:
        """
        Initialize walk-forward validator.
        
        Args:
            train_size: Number of days in training window
            test_size: Number of days in testing window
            gap_period: Number of days between train and test (default 0)
            anchored: If True, training window grows; if False, it rolls
            
        Raises:
            ValueError: If window sizes are invalid
        """
        pass
    
    def split(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into train/test windows.
        
        Args:
            data: Time series DataFrame with DatetimeIndex
            
        Returns:
            List of (train_data, test_data) tuples
            
        Raises:
            ValueError: If data is too short for specified windows
        """
        pass
    
    def get_n_splits(self, data: pd.DataFrame) -> int:
        """
        Calculate number of train/test splits.
        
        Args:
            data: Time series DataFrame
            
        Returns:
            Number of complete windows that can be generated
        """
        pass
    
    def _validate_parameters(self) -> None:
        """Validate window size parameters."""
        pass
    
    def _validate_data_length(self, data: pd.DataFrame) -> None:
        """Validate data is long enough for at least one split."""
        pass
    
    def _generate_windows(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate window indices.
        
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        pass
```

## Data Models

### OHLCV DataFrame Schema

The standard OHLCV DataFrame returned by DataLoader and consumed by all components:

```python
# DataFrame Structure
Index: DatetimeIndex (UTC timezone)
Columns:
  - Open: float64       # Opening price
  - High: float64       # Highest price during period
  - Low: float64        # Lowest price during period
  - Close: float64      # Closing price
  - Volume: int64       # Trading volume
  - Adj Close: float64  # Adjusted closing price (for splits/dividends)

# Example
"""
                         Open    High     Low   Close    Volume  Adj Close
Date                                                                       
2020-01-02 00:00:00+00:00  74.06  75.15  73.80  75.09  135480400     73.91
2020-01-03 00:00:00+00:00  74.29  75.14  74.13  74.36  146322800     73.19
2020-01-06 00:00:00+00:00  73.45  74.99  73.19  74.95  118387200     73.77
"""
```

### Signal Series Schema

The signal series returned by SignalBase.generate_signals():

```python
# Series Structure
Index: DatetimeIndex (matching OHLCV data)
Values: int8 (constrained to -1, 0, 1)
Name: "signal"

# Example
"""
Date
2020-01-02 00:00:00+00:00    1
2020-01-03 00:00:00+00:00    1
2020-01-06 00:00:00+00:00    0
2020-01-07 00:00:00+00:00   -1
Name: signal, dtype: int8
"""
```

### Backtest Results Dictionary Schema

```python
{
    "portfolio_value": pd.Series,  # Portfolio value over time
    "returns": pd.Series,          # Daily returns
    "trades": pd.DataFrame,        # Trade log with columns:
                                   # [Date, Side, Size, Price, Fees]
    "positions": pd.DataFrame,     # Position sizes over time
    "signal_series": pd.Series     # Signal values used
}
```

### Metrics Dictionary Schema

```python
{
    "sharpe_ratio": float | None,
    "sortino_ratio": float | None,
    "max_drawdown": float,
    "cagr": float,
    "total_return": float,
    "volatility": float,
    "win_rate": float
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property Reflection

After analyzing all acceptance criteria, I've identified the following consolidations to eliminate redundancy:

**Consolidations:**
- Properties 3.3, 3.4, 3.5 (configurable capital, commission, slippage) can be combined into one property about configuration parameters affecting backtest results
- Properties 5.2, 5.3, 5.4 (configurable train/test/gap sizes) can be combined into one property about window configuration
- Properties 2.3, 2.4, 2.5 (signal output validation) can be combined into one comprehensive signal validation property
- Properties 7.2, 7.3, 7.4, 7.5, 7.6, 7.7 (various input validations) can be grouped as they all test input validation with different invalid inputs

**Properties to Keep:**
- Data loading and caching properties (1.1, 1.5, 1.6, 1.8, 1.9, 6.1, 6.2, 6.3, 6.6, 6.8)
- Signal generation and validation (consolidated 2.3-2.5)
- Backtest execution (3.2, 3.6, 3.7, 3.9, 9.7)
- Metrics calculation (4.3, 4.4, 4.6, 4.8)
- Walk-forward validation (5.1, 5.5, 5.7, 5.11)
- Error handling (consolidated 7.1-7.7)

### Property 1: Valid Data Loading Returns OHLCV Schema

*For any* valid ticker and date range, loading data should return a DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume, Adj Close] with appropriate dtypes.

**Validates: Requirements 1.1**

### Property 2: Multiple Ticker Loading Completeness

*For any* list of valid tickers and date range, loading multiple tickers should return a dictionary containing an entry for each requested ticker.

**Validates: Requirements 1.5**

### Property 3: Date Range Validation

*For any* date pair where end_date is before or equal to start_date, the DataLoader should raise a ValueError.

**Validates: Requirements 1.6**

### Property 4: Timestamp Normalization to UTC

*For any* loaded OHLCV data, all timestamps in the DatetimeIndex should be in UTC timezone.

**Validates: Requirements 1.8**

### Property 5: Forward-Fill Missing Data

*For any* OHLCV data with gaps of 5 or fewer consecutive missing days, the DataLoader should forward-fill those gaps.

**Validates: Requirements 1.9**

### Property 6: Cache File Creation

*For any* ticker that is successfully loaded, a corresponding cache file should exist in the cache directory in parquet format.

**Validates: Requirements 6.1, 6.2, 6.3**

### Property 7: Cache Append Behavior

*For any* ticker with existing cached data, loading an extended date range should append new data to the existing cache file without duplicating overlapping dates.

**Validates: Requirements 6.6**

### Property 8: Cache Metadata Presence

*For any* cached parquet file, it should contain metadata including a last_updated timestamp.

**Validates: Requirements 6.8**

### Property 9: Signal Output Validation

*For any* signal implementation and valid OHLCV input, the generated signal series should: (1) have the same length as input data, (2) contain only values in {-1, 0, 1}, (3) have a DatetimeIndex matching the input, and (4) contain no NaN values.

**Validates: Requirements 2.3, 2.4, 2.5**

### Property 10: Backtest Results Completeness

*For any* valid signal and OHLCV data, running a backtest should return a dictionary containing all required keys: portfolio_value, returns, trades, positions, and signal_series.

**Validates: Requirements 3.2, 3.7, 9.7**

### Property 11: Position Rebalancing on Signal Change

*For any* backtest where the signal value changes from one day to the next, the positions DataFrame should reflect a corresponding change in position size.

**Validates: Requirements 3.6**

### Property 12: Configuration Parameters Affect Results

*For any* two backtests with identical signals and data but different initial_capital, commission, or slippage values, the resulting portfolio values should differ.

**Validates: Requirements 3.3, 3.4, 3.5**

### Property 13: Invalid Signal Rejection

*For any* signal series containing NaN values or values outside {-1, 0, 1}, the BacktestEngine should raise a descriptive error before executing the backtest.

**Validates: Requirements 3.9, 7.3, 7.4**

### Property 14: Maximum Drawdown Calculation

*For any* returns series, the maximum drawdown should be non-negative and represent the largest peak-to-trough decline in cumulative returns.

**Validates: Requirements 4.3**

### Property 15: Win Rate Calculation

*For any* returns series, the win rate should equal the count of positive returns divided by the total count of non-zero returns, expressed as a decimal between 0 and 1.

**Validates: Requirements 4.4**

### Property 16: Total Return from Cumulative Product

*For any* returns series, the total return should equal (1 + returns).prod() - 1.

**Validates: Requirements 4.6**

### Property 17: Metrics Dictionary Structure

*For any* returns series with sufficient data (≥30 days), calculate_metrics should return a dictionary containing all required keys: sharpe_ratio, sortino_ratio, max_drawdown, cagr, total_return, volatility, and win_rate.

**Validates: Requirements 4.8**

### Property 18: Sequential Train-Test Windows

*For any* time series data split by WalkForwardValidator, the test windows should be sequential and non-overlapping, with each test window starting after the previous test window ends.

**Validates: Requirements 5.1, 5.5**

### Property 19: Window Configuration Respected

*For any* WalkForwardValidator with specified train_size, test_size, and gap_period, each generated (train, test) tuple should have train data of length train_size and test data of length test_size, with gap_period days between them.

**Validates: Requirements 5.2, 5.3, 5.4**

### Property 20: Split Output Format

*For any* data split by WalkForwardValidator, the output should be a list of tuples where each tuple contains exactly two DataFrames (train_data, test_data).

**Validates: Requirements 5.7**

### Property 21: Insufficient Data Error

*For any* data where total length is less than (train_size + test_size + gap_period), WalkForwardValidator.split() should raise a descriptive ValueError.

**Validates: Requirements 5.11**

### Property 22: Input Validation Errors

*For any* invalid inputs to system components (negative prices in OHLCV, negative commission/slippage, non-positive initial capital, invalid date formats, missing required parameters), the system should raise a ValueError or appropriate exception with a descriptive message.

**Validates: Requirements 7.1, 7.2, 7.5, 7.6, 7.7**

## Error Handling

### Error Categories

1. **Input Validation Errors**
   - Invalid ticker symbols → ValueError with message "Invalid ticker: {ticker}"
   - Malformed date strings → ValueError with message "Invalid date format. Expected YYYY-MM-DD, got: {date}"
   - End date before start date → ValueError with message "end_date must be after start_date"
   - Negative prices in OHLCV → ValueError with message "OHLCV data contains negative prices"
   - Invalid signal values → ValueError with message "Signal contains invalid values. Expected {-1, 0, 1}, found: {invalid_values}"
   - NaN in signals → ValueError with message "Signal contains NaN values"

2. **Configuration Errors**
   - Non-positive initial capital → ValueError with message "initial_capital must be positive, got: {value}"
   - Negative commission/slippage → ValueError with message "{param} must be non-negative, got: {value}"
   - Invalid window sizes → ValueError with message "train_size and test_size must be positive integers"
   - Insufficient data for windows → ValueError with message "Data length {length} is insufficient for train_size={train} + test_size={test} + gap={gap}"

3. **Network Errors**
   - yfinance API failures → Retry up to 3 times with exponential backoff (1s, 2s, 4s)
   - After 3 retries → ConnectionError with message "Failed to fetch data for {ticker} after 3 retries"

4. **Data Integrity Errors**
   - Missing data beyond 5 days → Log warning and return available data
   - Empty data returned → ValueError with message "No data available for {ticker} in range {start} to {end}"

### Error Handling Strategy

- All errors include contextual information (ticker, dates, parameter values)
- Validation happens early (fail-fast principle)
- Network errors use retry with exponential backoff
- Data quality issues log warnings but don't fail unless critical
- All exceptions inherit from appropriate base classes (ValueError, ConnectionError, etc.)

### Logging

- Error logs include: timestamp, module name, function name, error message, stack trace
- Warning logs for: missing data, date range adjustments, cache misses
- Info logs for: successful data loads, cache hits, backtest completion
- Debug logs for: detailed execution flow, intermediate calculations

## Testing Strategy

### Dual Testing Approach

This project uses both unit tests and property-based tests for comprehensive coverage:

**Unit Tests** focus on:
- Specific examples with known expected outputs
- Edge cases (empty data, single-day backtests, zero volatility)
- Integration points between modules
- Error conditions and exception handling
- Mock-based testing for external dependencies (yfinance)

**Property-Based Tests** focus on:
- Universal properties that hold for all valid inputs
- Comprehensive input coverage through randomization
- Invariants that should never be violated
- Round-trip properties (cache save/load, signal generation)

### Property-Based Testing Configuration

**Framework**: Hypothesis (Python property-based testing library)

**Configuration**:
- Minimum 100 iterations per property test
- Each test tagged with comment: `# Feature: quantsignal-backtesting-engine, Property {N}: {property_text}`
- Custom strategies for generating valid OHLCV data, signals, date ranges
- Shrinking enabled to find minimal failing examples

**Example Property Test Structure**:
```python
from hypothesis import given, strategies as st
import hypothesis.extra.pandas as pdst

@given(
    ticker=st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=('Lu',))),
    start_date=st.dates(min_value=date(2010, 1, 1), max_value=date(2020, 1, 1)),
    end_date=st.dates(min_value=date(2020, 1, 2), max_value=date(2023, 1, 1))
)
@settings(max_examples=100)
def test_property_1_valid_data_loading_returns_ohlcv_schema(ticker, start_date, end_date):
    """
    Feature: quantsignal-backtesting-engine, Property 1: Valid Data Loading Returns OHLCV Schema
    
    For any valid ticker and date range, loading data should return a DataFrame 
    with DatetimeIndex and columns [Open, High, Low, Close, Volume, Adj Close].
    """
    # Test implementation
    pass
```

### Unit Test Coverage Targets

- data_loader.py: 95% coverage
- signal_base.py: 90% coverage
- engine.py: 95% coverage
- metrics.py: 95% coverage
- walk_forward.py: 90% coverage

### Test Organization

```
backend/tests/
├── unit/
│   ├── test_data_loader.py
│   ├── test_signal_base.py
│   ├── test_engine.py
│   ├── test_metrics.py
│   └── test_walk_forward.py
├── property/
│   ├── test_properties_data.py
│   ├── test_properties_signal.py
│   ├── test_properties_engine.py
│   ├── test_properties_metrics.py
│   └── test_properties_walk_forward.py
├── integration/
│   └── test_end_to_end.py
├── performance/
│   └── test_performance.py
└── conftest.py  # Shared fixtures
```

### Key Test Fixtures

```python
@pytest.fixture
def sample_ohlcv_data():
    """Generate synthetic OHLCV data for testing."""
    pass

@pytest.fixture
def simple_signal():
    """Create a simple test signal (e.g., always long)."""
    pass

@pytest.fixture
def mock_yfinance(monkeypatch):
    """Mock yfinance.download() for isolated testing."""
    pass

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide temporary cache directory for testing."""
    pass
```

### Performance Tests

- 10-year daily backtest completes in <5 seconds
- Cache load completes in <50ms
- 1000 walk-forward splits complete in <10 seconds

## Dependencies

### Core Dependencies

```
# Data manipulation and numerical computing
pandas==2.0.3
numpy==1.24.3
pyarrow==12.0.1  # For parquet I/O

# Backtesting framework
vectorbt==0.26.0

# Market data
yfinance==0.2.28

# Type checking
typing-extensions==4.7.1  # For Python 3.10 compatibility
```

### Development Dependencies

```
# Testing
pytest==7.4.0
pytest-cov==4.1.0
pytest-mock==3.11.1
hypothesis==6.82.0

# Code quality
black==23.7.0
flake8==6.0.0
mypy==1.4.1
isort==5.12.0

# Documentation
sphinx==7.1.0
sphinx-rtd-theme==1.3.0
```

### Optional Dependencies

```
# Performance profiling
line-profiler==4.0.3
memory-profiler==0.61.0

# Jupyter notebooks for research
jupyter==1.0.0
matplotlib==3.7.2
seaborn==0.12.2
```

## Requirements File

### backend/requirements.txt

```txt
# Core dependencies for QuantSignal Arena Backtesting Engine (Month 1)

# Data manipulation
pandas==2.0.3
numpy==1.24.3
pyarrow==12.0.1

# Backtesting
vectorbt==0.26.0

# Market data
yfinance==0.2.28
requests==2.31.0  # Required by yfinance

# Type hints
typing-extensions==4.7.1

# Testing
pytest==7.4.0
pytest-cov==4.1.0
pytest-mock==3.11.1
hypothesis==6.82.0

# Code quality
black==23.7.0
flake8==6.0.0
mypy==1.4.1
isort==5.12.0
```

### backend/requirements-dev.txt

```txt
# Development dependencies (includes all from requirements.txt)
-r requirements.txt

# Documentation
sphinx==7.1.0
sphinx-rtd-theme==1.3.0

# Performance profiling
line-profiler==4.0.3
memory-profiler==0.61.0

# Jupyter for research
jupyter==1.0.0
matplotlib==3.7.2
seaborn==0.12.2
ipython==8.14.0
```

## Implementation Notes

### Performance Considerations

1. **Vectorization**: All operations use pandas/numpy vectorized methods. No Python loops over time series data.

2. **Caching Strategy**: 
   - Cache files stored as parquet (10x faster than CSV, 50% smaller)
   - Cache key: ticker symbol
   - Cache invalidation: manual (no TTL for historical data)

3. **Memory Management**:
   - Use pandas categorical dtype for repeated string values
   - Use int8 for signal values instead of int64
   - Stream large datasets in chunks if needed (future enhancement)

4. **vectorbt Optimization**:
   - Pre-compile portfolio simulation with numba
   - Reuse portfolio objects when possible
   - Use vectorbt's built-in caching for repeated calculations

### Extensibility Points

1. **Custom Signals**: Inherit from SignalBase and implement generate_signals()
2. **Custom Fee Models**: Pass callable to BacktestEngine.run_backtest(fee_model=...)
3. **Custom Metrics**: Extend MetricsCalculator with additional methods
4. **Data Sources**: Extend DataLoader to support additional providers beyond yfinance

### Future Enhancements (Post-Month 1)

- Multi-asset portfolio backtesting
- Intraday data support (minute/hourly bars)
- Transaction cost analysis
- Slippage models based on volume
- Portfolio optimization integration
- Real-time data streaming
- Database backend for large-scale caching
- Distributed backtesting across multiple machines

## Appendix: Example Usage

### Basic Backtest Example

```python
from backend.backtester.data_loader import DataLoader
from backend.backtester.signal_base import SignalBase
from backend.backtester.engine import BacktestEngine
from backend.backtester.metrics import MetricsCalculator

# 1. Load data
loader = DataLoader(cache_dir="./data/cache")
ohlcv = loader.load_data("AAPL", "2020-01-01", "2023-12-31")

# 2. Define signal (example: simple moving average crossover)
class SMASignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        short_ma = ohlcv_data['Close'].rolling(20).mean()
        long_ma = ohlcv_data['Close'].rolling(50).mean()
        signals = pd.Series(0, index=ohlcv_data.index)
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1
        return signals

signal = SMASignal()

# 3. Run backtest
engine = BacktestEngine(initial_capital=100_000, commission=0.001)
results = engine.run_backtest(signal, ohlcv)

# 4. Calculate metrics
calc = MetricsCalculator(risk_free_rate=0.02)
metrics = calc.calculate_metrics(results['returns'])

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"CAGR: {metrics['cagr']:.2%}")
```

### Walk-Forward Validation Example

```python
from backend.backtester.walk_forward import WalkForwardValidator

# Load data
ohlcv = loader.load_data("AAPL", "2015-01-01", "2023-12-31")

# Configure walk-forward
validator = WalkForwardValidator(
    train_size=252,  # 1 year training
    test_size=63,    # 1 quarter testing
    gap_period=5,    # 1 week gap
    anchored=False   # Rolling window
)

# Split data
splits = validator.split(ohlcv)
print(f"Generated {len(splits)} train/test splits")

# Run backtest on each split
all_metrics = []
for train_data, test_data in splits:
    # Train signal on train_data (if needed)
    # Run backtest on test_data
    results = engine.run_backtest(signal, test_data)
    metrics = calc.calculate_metrics(results['returns'])
    all_metrics.append(metrics)

# Aggregate results
avg_sharpe = np.mean([m['sharpe_ratio'] for m in all_metrics if m['sharpe_ratio']])
print(f"Average Out-of-Sample Sharpe: {avg_sharpe:.2f}")
```

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Status**: Ready for Implementation
