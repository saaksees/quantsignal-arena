"""
Signal Explainer Module for QuantSignal Arena SHAP Layer.

Explains trading signal predictions using SHAP values.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
import logging

from backtester.signal_base import SignalBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalExplainer:
    """
    Explains trading signal predictions using SHAP values.
    
    Engineers technical features from OHLCV data, trains a surrogate
    classifier to mimic signal behavior, and computes SHAP values to
    identify key feature drivers.
    """
    
    def __init__(
        self,
        backtest_engine: Optional[Any] = None
    ) -> None:
        """
        Initialize SignalExplainer.
        
        Args:
            backtest_engine: Optional BacktestEngine instance (not used currently)
        """
        self.backtest_engine = backtest_engine
        self._last_explain_result = None
        logger.info("SignalExplainer initialized")
    
    def _build_features(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical features from OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with OHLCV columns and DatetimeIndex
            
        Returns:
            DataFrame with 8 feature columns:
            - returns_1d: 1-day percentage returns
            - returns_5d: 5-day percentage returns
            - returns_20d: 20-day percentage returns
            - volatility_20d: 20-day rolling std of returns (annualized)
            - volume_ratio: volume / 20-day avg volume
            - price_momentum: close / 20-day avg close - 1
            - rsi_14: 14-period RSI
            - bb_position: position within Bollinger Bands
        """
        close = ohlcv_data['Close']
        volume = ohlcv_data['Volume']
        
        # Returns features
        returns_1d = close.pct_change(1)
        returns_5d = close.pct_change(5)
        returns_20d = close.pct_change(20)
        
        # Volatility (annualized)
        volatility_20d = returns_1d.rolling(20).std() * np.sqrt(252)
        
        # Volume ratio
        volume_ratio = volume / volume.rolling(20).mean()
        
        # Price momentum
        price_momentum = close / close.rolling(20).mean() - 1
        
        # RSI 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi_14 = 100 - (100 / (1 + rs))
        
        # Bollinger Bands position
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)
        
        # Avoid division by zero
        band_width = upper_band - lower_band
        bb_position = np.where(
            band_width > 0,
            (close - lower_band) / band_width,
            0.5  # Default to middle if bands collapse
        )
        
        # Create features DataFrame
        features = pd.DataFrame({
            'returns_1d': returns_1d,
            'returns_5d': returns_5d,
            'returns_20d': returns_20d,
            'volatility_20d': volatility_20d,
            'volume_ratio': volume_ratio,
            'price_momentum': price_momentum,
            'rsi_14': rsi_14,
            'bb_position': bb_position
        }, index=ohlcv_data.index)
        
        # Drop NaN rows
        features = features.dropna()
        
        logger.info(f"Engineered {len(features.columns)} features from {len(ohlcv_data)} rows, {len(features)} rows after dropping NaN")
        
        return features
    
    def explain(
        self,
        signal: SignalBase,
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for signal predictions.
        
        Args:
            signal: Trading signal instance to explain
            ohlcv_data: Historical OHLCV data
            
        Returns:
            Dictionary containing:
            - shap_values: numpy array of shape (n_samples, n_features)
            - feature_names: list of feature names
            - feature_importance: dict mapping feature -> mean |SHAP|
            - base_value: float baseline prediction
            - summary: string describing top 3 features
            - model: trained classifier instance
        """
        logger.info(f"Explaining signal '{signal.name}' with {len(ohlcv_data)} data points")
        
        # Build features
        X = self._build_features(ohlcv_data)
        
        # Generate signals
        y = signal.generate_signals(ohlcv_data)
        
        # Align X and y by index
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # Filter to only long (1) and short (-1) positions, exclude neutral (0)
        mask = (y == 1) | (y == -1)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        if len(X_filtered) == 0:
            raise ValueError("No non-neutral signals found. Cannot train classifier.")
        
        logger.info(f"Training on {len(X_filtered)} samples ({len(X_filtered[y_filtered == 1])} long, {len(X_filtered[y_filtered == -1])} short)")
        
        # Train RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_filtered, y_filtered)
        
        # Get SHAP values
        tree_explainer = shap.TreeExplainer(model)
        shap_values_raw = tree_explainer.shap_values(X_filtered)
        
        # RandomForest returns either list [shap_class_0, shap_class_1] or 3D array (n_samples, n_features, n_classes)
        # Take class 1 (positive signal) values only
        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[1]
        elif len(shap_values_raw.shape) == 3:
            # 3D array: take class 1 (index 1)
            shap_values = shap_values_raw[:, :, 1]
        else:
            shap_values = shap_values_raw
        
        # Mean absolute SHAP per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # base_value — also a list for multiclass, take class 1
        base_val = tree_explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_value = float(base_val[1])
        else:
            base_value = float(base_val)
        
        # Feature importance dict sorted descending
        feature_names = X.columns.tolist()
        feature_importance = dict(
            sorted(
                zip(feature_names, mean_abs_shap.tolist()),
                key=lambda x: x[1],
                reverse=True
            )
        )
        
        # Mean shap per feature (signed, for direction)
        mean_shap_signed = np.array(shap_values).mean(axis=0).flatten()
        
        # Build summary string with top 3 features
        top_3_features = list(feature_importance.keys())[:3]
        summary_parts = []
        for feat in top_3_features:
            feat_idx = feature_names.index(feat)
            direction = "positive" if float(mean_shap_signed[feat_idx]) > 0 else "negative"
            summary_parts.append(f"{feat} ({direction})")
        summary = "Top drivers: " + ", ".join(summary_parts)
        
        result = {
            'shap_values': shap_values,  # Already extracted class 1, should be 2D
            'feature_names': feature_names,
            'feature_importance': feature_importance,
            'base_value': base_value,
            'summary': summary,
            'model': model,
            'mean_shap_signed': mean_shap_signed
        }
        
        # Store for get_top_features
        self._last_explain_result = result
        
        logger.info(f"SHAP explanation complete. {summary}")
        
        return result
    
    def get_top_features(
        self,
        explain_result: Dict[str, Any],
        n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get top N most important features.
        
        Args:
            explain_result: Result dictionary from explain() method
            n: Number of top features to return
            
        Returns:
            List of dicts with keys:
            - name: feature name
            - mean_shap: mean absolute SHAP value
            - direction: "positive" or "negative"
        """
        importance = explain_result["feature_importance"]
        shap_vals = explain_result["shap_values"]
        feature_names = explain_result["feature_names"]
        mean_signed = np.array(shap_vals).mean(axis=0).flatten()
        signed_map = dict(zip(feature_names, mean_signed.tolist()))
        
        top = list(importance.items())[:n]
        return [
            {
                "name": name,
                "mean_shap": val,
                "direction": "positive" if float(signed_map[name]) > 0 else "negative"
            }
            for name, val in top
        ]
