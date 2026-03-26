# Requirements Document

## Introduction

The SHAP Explainability Layer provides interpretability and monitoring capabilities for trading signals in QuantSignal Arena. It explains why signals generate predictions, detects when signal behavior changes over time (drift), and produces professional quantitative tearsheet reports. This layer enables users to understand signal drivers, monitor signal stability, and generate publication-ready performance documentation.

## Glossary

- **SHAP_Explainer**: Component that uses SHAP (SHapley Additive exPlanations) to attribute signal predictions to input features
- **Drift_Detector**: Component that monitors signal and return distributions over time to detect regime changes
- **Report_Builder**: Component that generates PDF tearsheet reports with performance metrics, charts, and explanations
- **Signal**: Trading signal instance inheriting from SignalBase that generates position recommendations
- **OHLCV_Data**: DataFrame containing Open, High, Low, Close, Volume market data with DatetimeIndex
- **Feature**: Derived technical indicator computed from OHLCV data (e.g., returns, volatility, RSI)
- **PSI**: Population Stability Index, a metric measuring distribution drift between two datasets
- **Backtest_Engine**: Existing vectorized backtesting engine that simulates portfolio performance
- **TreeExplainer**: SHAP explainer optimized for tree-based models like LightGBM and RandomForest
- **Tearsheet**: Professional quantitative report documenting strategy hypothesis, performance, and analysis

## Requirements

### Requirement 1: Signal Feature Engineering

**User Story:** As a quant researcher, I want to automatically generate technical features from OHLCV data, so that I can explain signal behavior using interpretable indicators.

#### Acceptance Criteria

1. WHEN OHLCV_Data is provided, THE SHAP_Explainer SHALL compute returns_1d as 1-day percentage returns
2. WHEN OHLCV_Data is provided, THE SHAP_Explainer SHALL compute returns_5d as 5-day percentage returns
3. WHEN OHLCV_Data is provided, THE SHAP_Explainer SHALL compute returns_20d as 20-day percentage returns
4. WHEN OHLCV_Data is provided, THE SHAP_Explainer SHALL compute volatility_20d as 20-day rolling standard deviation of returns
5. WHEN OHLCV_Data is provided, THE SHAP_Explainer SHALL compute volume_ratio as current volume divided by 20-day average volume
6. WHEN OHLCV_Data is provided, THE SHAP_Explainer SHALL compute price_momentum as 10-day rate of change
7. WHEN OHLCV_Data is provided, THE SHAP_Explainer SHALL compute rsi_14 as 14-period Relative Strength Index
8. WHEN OHLCV_Data is provided, THE SHAP_Explainer SHALL compute bb_position as normalized position within Bollinger Bands (20-period, 2 standard deviations)
9. THE SHAP_Explainer SHALL return all features as a DataFrame with the same DatetimeIndex as OHLCV_Data
10. WHEN any feature computation results in NaN values, THE SHAP_Explainer SHALL forward-fill or drop those rows

### Requirement 2: Signal Explanation via SHAP

**User Story:** As a quant researcher, I want to understand which features drive my signal's predictions, so that I can validate the signal's logic and identify key market drivers.

#### Acceptance Criteria

1. WHEN a Signal and OHLCV_Data are provided, THE SHAP_Explainer SHALL train a classifier to predict signal direction from features
2. WHERE LightGBM is available, THE SHAP_Explainer SHALL use LightGBM classifier
3. WHERE LightGBM is not available, THE SHAP_Explainer SHALL use RandomForest classifier as fallback
4. THE SHAP_Explainer SHALL train the classifier to predict signal values (1 vs -1), excluding neutral positions (0)
5. WHEN the trained model is available, THE SHAP_Explainer SHALL compute SHAP values using TreeExplainer
6. THE SHAP_Explainer SHALL return a dict containing shap_values as numpy array
7. THE SHAP_Explainer SHALL return a dict containing feature_names as list of strings
8. THE SHAP_Explainer SHALL return a dict containing feature_importance as dict sorted by absolute mean SHAP value descending
9. THE SHAP_Explainer SHALL return a dict containing base_value as float representing the model's baseline prediction
10. THE SHAP_Explainer SHALL return a dict containing summary as string describing the top 3 features driving the signal
11. THE SHAP_Explainer SHALL ensure shap_values array shape matches (n_samples, n_features)

### Requirement 3: Top Feature Extraction

**User Story:** As a quant researcher, I want to quickly identify the most important features, so that I can focus on the key drivers of signal performance.

#### Acceptance Criteria

1. WHEN get_top_features is called with parameter n, THE SHAP_Explainer SHALL return exactly n features
2. THE SHAP_Explainer SHALL return features sorted by absolute mean SHAP value in descending order
3. THE SHAP_Explainer SHALL return each feature as a dict containing name as string
4. THE SHAP_Explainer SHALL return each feature as a dict containing mean_shap as float
5. THE SHAP_Explainer SHALL return each feature as a dict containing direction as string ("positive" or "negative")
6. WHEN mean_shap is positive, THE SHAP_Explainer SHALL set direction to "positive"
7. WHEN mean_shap is negative, THE SHAP_Explainer SHALL set direction to "negative"

### Requirement 4: Distribution Drift Detection via PSI

**User Story:** As a quant researcher, I want to detect when signal or return distributions change significantly, so that I can identify when market regimes shift and signals may need retraining.

#### Acceptance Criteria

1. WHEN two distributions are provided, THE Drift_Detector SHALL compute Population Stability Index (PSI)
2. THE Drift_Detector SHALL discretize distributions into configurable number of bins (default 10)
3. THE Drift_Detector SHALL compute PSI as sum of (reference_pct - current_pct) * ln(reference_pct / current_pct) across bins
4. WHEN reference and current distributions are identical, THE Drift_Detector SHALL return PSI approximately equal to 0.0
5. WHEN PSI is less than 0.1, THE Drift_Detector SHALL classify drift_level as "none"
6. WHEN PSI is between 0.1 and 0.2 inclusive, THE Drift_Detector SHALL classify drift_level as "moderate"
7. WHEN PSI is greater than 0.2, THE Drift_Detector SHALL classify drift_level as "significant"
8. THE Drift_Detector SHALL return PSI as float

### Requirement 5: Signal Stability Monitoring

**User Story:** As a quant researcher, I want to monitor signal stability over time, so that I can detect when signals stop working and take corrective action.

#### Acceptance Criteria

1. WHEN detect method is called, THE Drift_Detector SHALL split OHLCV_Data into reference period using first reference_window days
2. WHEN detect method is called, THE Drift_Detector SHALL split OHLCV_Data into recent period using last detection_window days
3. THE Drift_Detector SHALL generate signal values for both reference and recent periods
4. THE Drift_Detector SHALL compute PSI on signal distributions between reference and recent periods
5. THE Drift_Detector SHALL compute PSI on return distributions between reference and recent periods
6. THE Drift_Detector SHALL return dict containing signal_psi as float
7. THE Drift_Detector SHALL return dict containing return_psi as float
8. THE Drift_Detector SHALL return dict containing drift_detected as boolean
9. WHEN either signal_psi or return_psi exceeds 0.2, THE Drift_Detector SHALL set drift_detected to True
10. WHEN both signal_psi and return_psi are 0.2 or less, THE Drift_Detector SHALL set drift_detected to False
11. THE Drift_Detector SHALL return dict containing drift_level as string
12. THE Drift_Detector SHALL return dict containing recommendation as string
13. WHEN drift_level is "none", THE Drift_Detector SHALL set recommendation to "signal stable"
14. WHEN drift_level is "moderate", THE Drift_Detector SHALL set recommendation to "monitor closely"
15. WHEN drift_level is "significant", THE Drift_Detector SHALL set recommendation to "consider retraining"

### Requirement 6: Rolling PSI Analysis

**User Story:** As a quant researcher, I want to track PSI over rolling time windows, so that I can visualize drift trends and identify when instability began.

#### Acceptance Criteria

1. WHEN rolling_psi is called, THE Drift_Detector SHALL compute PSI on rolling windows across OHLCV_Data
2. THE Drift_Detector SHALL use reference_window as the reference period size
3. THE Drift_Detector SHALL use detection_window as the comparison period size
4. THE Drift_Detector SHALL return a Series with DatetimeIndex representing window end dates
5. THE Drift_Detector SHALL return PSI values as floats in the Series
6. THE Drift_Detector SHALL compute PSI for each valid rolling window pair

### Requirement 7: Professional Tearsheet Generation

**User Story:** As a quant researcher, I want to generate publication-quality PDF reports, so that I can document and share strategy performance with stakeholders.

#### Acceptance Criteria

1. WHEN build method is called, THE Report_Builder SHALL create a PDF file in the output directory
2. THE Report_Builder SHALL include a header section containing strategy name, hypothesis text, and generation date
3. THE Report_Builder SHALL include a performance summary table with Sharpe ratio, Sortino ratio, max drawdown, CAGR, and win rate
4. THE Report_Builder SHALL include an equity curve chart showing portfolio value over time using matplotlib
5. THE Report_Builder SHALL include a SHAP feature importance chart as horizontal bar chart showing top features
6. THE Report_Builder SHALL include a drift analysis section with PSI values color-coded by status
7. WHEN drift_level is "none", THE Report_Builder SHALL display PSI status in green
8. WHEN drift_level is "moderate", THE Report_Builder SHALL display PSI status in yellow
9. WHEN drift_level is "significant", THE Report_Builder SHALL display PSI status in red
10. THE Report_Builder SHALL include generated signal code as syntax-highlighted code block
11. THE Report_Builder SHALL save PDF to output_dir with filename format {strategy_name}_{date}.pdf
12. THE Report_Builder SHALL return the full file path as string
13. WHEN output directory does not exist, THE Report_Builder SHALL create it

### Requirement 8: Chart Generation for Reports

**User Story:** As a quant researcher, I want high-quality charts in my tearsheets, so that I can visually communicate strategy performance and feature importance.

#### Acceptance Criteria

1. WHEN equity curve is generated, THE Report_Builder SHALL use matplotlib to plot portfolio_value over time
2. THE Report_Builder SHALL label equity curve x-axis as "Date" and y-axis as "Portfolio Value ($)"
3. WHEN SHAP feature importance chart is generated, THE Report_Builder SHALL create horizontal bar chart
4. THE Report_Builder SHALL sort features by absolute importance in descending order
5. THE Report_Builder SHALL display top 10 features maximum in feature importance chart
6. THE Report_Builder SHALL label feature importance x-axis as "Mean |SHAP Value|" and y-axis as "Feature"
7. THE Report_Builder SHALL embed charts as images in the PDF document
8. THE Report_Builder SHALL use appropriate figure sizes for PDF layout

### Requirement 9: Integration with Existing Backtesting Engine

**User Story:** As a quant researcher, I want the SHAP layer to work seamlessly with the existing backtesting engine, so that I can explain any backtested signal without code changes.

#### Acceptance Criteria

1. THE SHAP_Explainer SHALL accept Backtest_Engine instance in constructor
2. THE SHAP_Explainer SHALL accept any Signal instance inheriting from SignalBase
3. THE SHAP_Explainer SHALL work with OHLCV_Data in the same format as Backtest_Engine
4. THE Report_Builder SHALL accept backtest_results dict from Backtest_Engine.run_backtest
5. THE Report_Builder SHALL accept metrics dict from MetricsCalculator.calculate_metrics
6. THE Report_Builder SHALL integrate SHAP results and drift results into unified report

### Requirement 10: Dependency Management

**User Story:** As a developer, I want all required dependencies documented and installed, so that the SHAP layer works without manual dependency resolution.

#### Acceptance Criteria

1. THE requirements.txt SHALL include shap version 0.44.0 or higher
2. THE requirements.txt SHALL include lightgbm version 4.0.0 or higher
3. THE requirements.txt SHALL include fpdf2 version 2.7.0 or higher
4. THE requirements.txt SHALL include scikit-learn version 1.3.0 or higher
5. THE requirements.txt SHALL include matplotlib for chart generation
