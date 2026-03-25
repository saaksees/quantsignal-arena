"""
QuantSignal Arena Backtesting Engine

A high-performance, vectorized backtesting system for quantitative trading signals.
"""

from .data_loader import DataLoader
from .signal_base import SignalBase, MomentumSignal

__all__ = ['DataLoader', 'SignalBase', 'MomentumSignal']
__version__ = '0.1.0'
