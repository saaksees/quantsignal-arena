"""
Drift Detector Module for QuantSignal Arena SHAP Layer.

Detects distribution drift in signals and returns using PSI.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

from backtester.signal_base import SignalBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects distribution drift in signals and returns using PSI.
    
    Monitors signal stability by comparing reference and recent
    distributions using Population Stability Index.
    """
    
    def __init__(
        self,
        reference_window: int = 126,
        detection_window: int = 21
    ) -> None:
        """
        Initialize DriftDetector.
        
        Args:
            reference_window: Days for reference period (default 126 = 6 months)
            detection_window: Days for recent period (default 21 = 1 month)
        """
        self.reference_window = reference_window
        self.detection_window = detection_window
        logger.info(f"DriftDetector initialized with reference_window={reference_window}, detection_window={detection_window}")
    
    def compute_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Compute Population Stability Index between two distributions.
        
        Formula: PSI = Σ (current_pct - reference_pct) * ln(current_pct / reference_pct)
        
        Args:
            reference: Reference distribution
            current: Current distribution to compare
            bins: Number of bins (default 10)
            
        Returns:
            PSI value as float (0 = identical, >0.2 = significant drift)
        """
        # Create bin edges from reference distribution
        breakpoints = np.linspace(0, 100, bins + 1)
        ref_percentiles = np.percentile(reference, breakpoints)
        ref_percentiles = np.unique(ref_percentiles)  # remove duplicates
        
        # Count observations in each bin
        ref_counts = np.histogram(reference, bins=ref_percentiles)[0]
        cur_counts = np.histogram(current, bins=ref_percentiles)[0]
        
        # Convert to proportions
        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)
        
        # Add epsilon to avoid log(0) or division by zero
        epsilon = 1e-6
        ref_props = np.where(ref_props == 0, epsilon, ref_props)
        cur_props = np.where(cur_props == 0, epsilon, cur_props)
        
        # PSI = sum((current - reference) * ln(current/reference))
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        return float(psi)
    
    def detect(
        self,
        signal: SignalBase,
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect drift in signal and return distributions.
        
        Args:
            signal: Trading signal instance to monitor
            ohlcv_data: Historical OHLCV data
            
        Returns:
            Dictionary containing:
            - signal_psi: PSI for signal distribution
            - return_psi: PSI for return distribution
            - drift_detected: True if either PSI > 0.2
            - drift_level: "none", "moderate", or "significant"
            - recommendation: action recommendation string
            - max_psi: maximum of signal_psi and return_psi
            - reference_period: tuple of (start_date, end_date)
            - detection_period: tuple of (start_date, end_date)
            
        Raises:
            ValueError: If insufficient data
        """
        # Validate sufficient data
        min_required = self.reference_window + self.detection_window
        if len(ohlcv_data) < min_required:
            raise ValueError(
                f"Insufficient data: need at least {min_required} rows "
                f"(reference_window={self.reference_window} + detection_window={self.detection_window}), "
                f"got {len(ohlcv_data)}"
            )
        
        logger.info(f"Detecting drift for signal '{signal.name}' with {len(ohlcv_data)} data points")
        
        # Split data into reference and detection periods using iloc (positional)
        reference_data = ohlcv_data.iloc[:self.reference_window]
        detection_data = ohlcv_data.iloc[-self.detection_window:]
        
        # Generate signals for full dataset
        all_signals = signal.generate_signals(ohlcv_data)
        
        # Slice signals using iloc (positional) to match data splits
        reference_signals = all_signals.iloc[:self.reference_window]
        detection_signals = all_signals.iloc[-self.detection_window:]
        
        # Compute signal PSI
        signal_psi = self.compute_psi(reference_signals, detection_signals)
        
        # Compute returns FIRST, then slice — never index returns by ohlcv index
        close = ohlcv_data['Close']
        returns = close.pct_change().dropna()  # this drops first row
        
        # Slice reference and detection periods from RETURNS index, not ohlcv index
        ref_return_data = returns.iloc[:self.reference_window]
        det_return_data = returns.iloc[-self.detection_window:]
        
        # Compute return PSI
        return_psi = self.compute_psi(ref_return_data, det_return_data)
        
        # Determine max PSI and drift level
        max_psi = max(signal_psi, return_psi)
        
        if max_psi < 0.1:
            drift_level = "none"
            recommendation = "signal stable"
        elif max_psi <= 0.2:
            drift_level = "moderate"
            recommendation = "monitor closely"
        else:
            drift_level = "significant"
            recommendation = "consider retraining"
        
        drift_detected = max_psi > 0.2
        
        result = {
            'signal_psi': float(signal_psi),
            'return_psi': float(return_psi),
            'drift_detected': drift_detected,
            'drift_level': drift_level,
            'recommendation': recommendation,
            'max_psi': float(max_psi),
            'reference_period': (reference_data.index[0], reference_data.index[-1]),
            'detection_period': (detection_data.index[0], detection_data.index[-1])
        }
        
        logger.info(f"Drift detection complete: drift_level={drift_level}, max_psi={max_psi:.4f}")
        
        return result
    
    def rolling_psi(
        self,
        signal: SignalBase,
        ohlcv_data: pd.DataFrame
    ) -> pd.Series:
        """
        Compute rolling PSI over time.
        
        Args:
            signal: Trading signal instance to monitor
            ohlcv_data: Historical OHLCV data
            
        Returns:
            Series with DatetimeIndex (window end dates) and PSI values
        """
        # Use first reference_window rows as fixed reference
        reference_data = ohlcv_data.iloc[:self.reference_window]
        
        # Generate signals for full dataset
        all_signals = signal.generate_signals(ohlcv_data)
        reference_signals = all_signals.loc[reference_data.index]
        
        # Slide detection_window across remaining data
        psi_values = []
        window_end_dates = []
        
        start_idx = self.reference_window
        end_idx = len(ohlcv_data)
        
        for i in range(start_idx, end_idx - self.detection_window + 1):
            # Get detection window
            detection_data = ohlcv_data.iloc[i:i + self.detection_window]
            detection_signals = all_signals.loc[detection_data.index]
            
            # Compute PSI
            psi = self.compute_psi(reference_signals, detection_signals)
            
            psi_values.append(psi)
            window_end_dates.append(detection_data.index[-1])
        
        result = pd.Series(psi_values, index=pd.DatetimeIndex(window_end_dates))
        
        logger.info(f"Rolling PSI computed: {len(result)} windows")
        
        return result
