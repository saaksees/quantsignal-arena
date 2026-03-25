"""
QuantSignal Arena Backtesting Engine

A high-performance, vectorized backtesting system for quantitative trading signals.
"""

from .data_loader import DataLoader
from .signal_base import SignalBase, MomentumSignal
from .engine import BacktestEngine
from .metrics import MetricsCalculator
from .walk_forward import WalkForwardValidator

__all__ = ['DataLoader', 'SignalBase', 'MomentumSignal', 'BacktestEngine', 'MetricsCalculator', 'WalkForwardValidator']
__version__ = '0.1.0'
