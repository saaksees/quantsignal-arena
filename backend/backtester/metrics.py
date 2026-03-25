"""
Metrics Calculator Module for QuantSignal Arena.

Computes performance metrics for backtested trading strategies.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate performance metrics for trading strategies.
    
    Computes risk-adjusted returns, drawdowns, and other
    key performance indicators.
    """
    
    TRADING_DAYS_PER_YEAR = 252
    MIN_DAYS_FOR_RATIOS = 30
    
    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """
        Initialize metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 0.02 = 2%)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"MetricsCalculator initialized with risk_free_rate={risk_free_rate}")
    
    def calculate_metrics(
        self,
        portfolio_returns: pd.Series,
        risk_free_rate: Optional[float] = None
    ) -> dict:
        """
        Calculate all performance metrics for a return series.
        
        Args:
            portfolio_returns: Series of daily portfolio returns
            risk_free_rate: Override risk-free rate (optional)
            
        Returns:
            Dictionary containing:
            - sharpe_ratio: Annualized Sharpe ratio (None if insufficient data or zero volatility)
            - sortino_ratio: Annualized Sortino ratio (None if insufficient data or zero downside)
            - max_drawdown: Maximum drawdown as negative float (e.g., -0.23)
            - win_rate: Percentage of positive return days (0.0 to 1.0)
            - cagr: Compound annual growth rate
            - total_return: Total return as float (e.g., 0.45 = 45%)
            - volatility: Annualized standard deviation
            - calmar_ratio: CAGR / abs(max_drawdown) (None if drawdown is zero)
        """
        rfr = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        logger.info(f"Calculating metrics for {len(portfolio_returns)} return observations")
        
        # Handle edge cases
        if len(portfolio_returns) == 0:
            return self._empty_metrics()
        
        # Calculate each metric
        metrics = {
            'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_returns, rfr),
            'sortino_ratio': self._calculate_sortino_ratio(portfolio_returns, rfr),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'win_rate': self._calculate_win_rate(portfolio_returns),
            'cagr': self._calculate_cagr(portfolio_returns),
            'total_return': self._calculate_total_return(portfolio_returns),
            'volatility': self._calculate_volatility(portfolio_returns),
            'calmar_ratio': None  # Will be calculated after we have CAGR and max_drawdown
        }
        
        # Calculate Calmar ratio using CAGR and max_drawdown
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(
            metrics['cagr'],
            metrics['max_drawdown']
        )
        
        logger.info(f"Metrics calculated: Sharpe={metrics['sharpe_ratio']}, "
                   f"Sortino={metrics['sortino_ratio']}, MaxDD={metrics['max_drawdown']}")
        
        return metrics
    
    def _calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float
    ) -> Optional[float]:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < self.MIN_DAYS_FOR_RATIOS:
            return None
        
        # Calculate excess returns
        daily_rf_rate = risk_free_rate / self.TRADING_DAYS_PER_YEAR
        excess_returns = returns - daily_rf_rate
        
        # Calculate volatility
        volatility = returns.std()
        
        if volatility == 0 or np.isnan(volatility):
            return None
        
        # Annualize
        sharpe = (excess_returns.mean() / volatility) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        return float(sharpe)
    
    def _calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float
    ) -> Optional[float]:
        """Calculate annualized Sortino ratio using downside deviation."""
        if len(returns) < self.MIN_DAYS_FOR_RATIOS:
            return None
        
        # Calculate excess returns
        daily_rf_rate = risk_free_rate / self.TRADING_DAYS_PER_YEAR
        excess_returns = returns - daily_rf_rate
        
        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return None
        
        downside_std = downside_returns.std()
        
        if downside_std == 0 or np.isnan(downside_std):
            return None
        
        # Annualize
        sortino = (excess_returns.mean() / downside_std) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        return float(sortino)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown as negative float."""
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Get maximum drawdown (most negative value)
        max_dd = drawdown.min()
        
        return float(max_dd) if not np.isnan(max_dd) else 0.0
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate percentage of days with positive returns."""
        if len(returns) == 0:
            return 0.0
        
        positive_days = (returns > 0).sum()
        total_days = len(returns)
        
        return float(positive_days / total_days)
    
    def _calculate_cagr(self, returns: pd.Series) -> Optional[float]:
        """Calculate compound annual growth rate."""
        if len(returns) == 0:
            return None
        
        # Calculate total return
        total_return = (1 + returns).prod() - 1
        
        # Calculate number of years
        n_years = len(returns) / self.TRADING_DAYS_PER_YEAR
        
        if n_years == 0:
            return None
        
        # Calculate CAGR
        cagr = (1 + total_return) ** (1 / n_years) - 1
        
        return float(cagr) if not np.isnan(cagr) else None
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate total return."""
        if len(returns) == 0:
            return 0.0
        
        total_return = (1 + returns).prod() - 1
        
        return float(total_return) if not np.isnan(total_return) else 0.0
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0
        
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        return float(annual_vol) if not np.isnan(annual_vol) else 0.0
    
    def _calculate_calmar_ratio(
        self,
        cagr: Optional[float],
        max_drawdown: float
    ) -> Optional[float]:
        """Calculate Calmar ratio (CAGR / abs(max_drawdown))."""
        if cagr is None or max_drawdown == 0:
            return None
        
        calmar = cagr / abs(max_drawdown)
        
        return float(calmar)
    
    def _empty_metrics(self) -> dict:
        """Return metrics dict with None/zero values for empty input."""
        return {
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'cagr': None,
            'total_return': 0.0,
            'volatility': 0.0,
            'calmar_ratio': None
        }
