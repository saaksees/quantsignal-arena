# Implementation Plan: SHAP Explainability Layer

## Overview

This plan implements the SHAP Explainability Layer for QuantSignal Arena, adding interpretability and monitoring capabilities to trading signals. The implementation follows a component-by-component approach: SignalExplainer for SHAP-based feature importance analysis, DriftDetector for distribution stability monitoring using PSI, and ReportBuilder for professional PDF tearsheet generation. Each component is implemented with comprehensive testing including property-based tests to validate universal correctness properties.

## Tasks

- [x] 1. Implement SignalExplainer core functionality
  - [x] 1.1 Create SignalExplainer class with initialization and feature engineering
    - Create `backend/shap_layer/explainer.py` with SignalExplainer class
    - Implement `__init__` method accepting signal, use_lightgbm, n_estimators, random_state parameters
    - Implement `engineer_features` method computing all 8 technical features from OHLCV data
    - Feature formulas: returns_1d (1-day pct_change), returns_5d (5-day pct_change), returns_20d (20-day pct_change), volatility_20d (20-day rolling std), volume_ratio (volume / 20-day avg), price_momentum (10-day rate of change), rsi_14 (14-period RSI), bb_position (Bollinger Bands position)
    - Return DataFrame with same DatetimeIndex as input, drop NaN rows
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10_

  - [ ]* 1.2 Write property test for feature engineering correctness
    - **Property 1: Feature Engineering Correctness**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10**
    - Generate random OHLCV data (≥20 days) using Hypothesis
    - Verify engineer_features computes all 8 features with correct formulas
    - Verify output DataFrame has matching DatetimeIndex with no NaN values
    - Tag: `# Feature: shap-explainability-layer, Property 1: Feature Engineering Correctness`

  - [x] 1.3 Implement SHAP explanation generation
    - Implement `explain` method in SignalExplainer class
    - Train LightGBM classifier (or RandomForest fallback) to predict signal direction (1 vs -1)
    - Filter training data to exclude neutral positions (signal == 0)
    - Compute SHAP values using TreeExplainer
    - Return dict with keys: shap_values, feature_names, feature_importance, base_value, summary, model, features_used
    - Ensure shap_values shape matches (n_samples, n_features)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11_

  - [ ]* 1.4 Write property test for SHAP output schema completeness
    - **Property 2: SHAP Output Schema Completeness**
    - **Validates: Requirements 2.6, 2.7, 2.8, 2.9, 2.10, 2.11**
    - Generate random Signal and OHLCV data using Hypothesis
    - Verify explain() returns dict with all required keys and correct types
    - Verify shap_values shape matches (n_samples, n_features)
    - Tag: `# Feature: shap-explainability-layer, Property 2: SHAP Output Schema Completeness`

  - [ ]* 1.5 Write property test for signal filtering during training
    - **Property 3: Signal Filtering for Training**
    - **Validates: Requirements 2.4**
    - Generate random Signal with mixed positions (1, 0, -1) using Hypothesis
    - Verify training data excludes all neutral positions (0)
    - Verify only samples with signal values 1 or -1 are included
    - Tag: `# Feature: shap-explainability-layer, Property 3: Signal Filtering for Training`

  - [x] 1.6 Implement top features extraction
    - Implement `get_top_features` method in SignalExplainer class
    - Accept parameter n for number of top features to return
    - Return list of dicts with keys: name, mean_shap, direction
    - Sort by absolute mean_shap in descending order
    - Set direction to "positive" if mean_shap > 0, else "negative"
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [ ]* 1.7 Write property test for top features structure and sorting
    - **Property 4: Top Features Structure and Sorting**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**
    - Generate random n and SHAP results using Hypothesis
    - Verify get_top_features(n) returns exactly n features (or fewer if fewer exist)
    - Verify each feature has keys {name, mean_shap, direction}
    - Verify features sorted by absolute mean_shap descending
    - Verify direction is "positive" if mean_shap > 0 else "negative"
    - Tag: `# Feature: shap-explainability-layer, Property 4: Top Features Structure and Sorting`

  - [ ]* 1.8 Write unit tests for SignalExplainer
    - Create `backend/tests/test_explainer.py`
    - Test feature engineering with known OHLCV data and expected feature values
    - Test explain() with MomentumSignal and verify output structure
    - Test get_top_features with n=3 and verify correct features returned
    - Test error handling for insufficient data (<20 rows)
    - Test error handling for signal with all neutral positions
    - Test LightGBM vs RandomForest fallback behavior

- [ ] 2. Checkpoint - Verify SignalExplainer implementation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Implement DriftDetector core functionality
  - [x] 3.1 Create DriftDetector class with PSI computation
    - Create `backend/shap_layer/drift_detector.py` with DriftDetector class
    - Implement `__init__` method accepting signal, reference_window, detection_window, n_bins parameters
    - Implement `compute_psi` method calculating Population Stability Index
    - PSI formula: Σ (current_pct - reference_pct) * ln(current_pct / reference_pct)
    - Discretize distributions into n_bins using quantiles from reference
    - Add epsilon (1e-10) to avoid division by zero
    - Return PSI as float
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.8_

  - [ ]* 3.2 Write property test for PSI symmetry and identity
    - **Property 5: PSI Symmetry and Identity**
    - **Validates: Requirements 4.1, 4.4, 4.8**
    - Generate random distribution using Hypothesis
    - Verify compute_psi(dist, dist) returns approximately 0.0 (within tolerance)
    - Verify PSI is symmetric: compute_psi(ref, cur) ≈ compute_psi(cur, ref)
    - Tag: `# Feature: shap-explainability-layer, Property 5: PSI Symmetry and Identity`

  - [ ]* 3.3 Write property test for PSI classification thresholds
    - **Property 6: PSI Classification Thresholds**
    - **Validates: Requirements 4.5, 4.6, 4.7**
    - Generate random PSI values using Hypothesis
    - Verify drift_level is "none" if PSI < 0.1
    - Verify drift_level is "moderate" if 0.1 ≤ PSI ≤ 0.2
    - Verify drift_level is "significant" if PSI > 0.2
    - Tag: `# Feature: shap-explainability-layer, Property 6: PSI Classification Thresholds`

  - [x] 3.4 Implement drift detection with window splitting
    - Implement `detect` method in DriftDetector class
    - Split OHLCV data into reference period (first reference_window days) and recent period (last detection_window days)
    - Generate signal values for both periods using signal instance
    - Compute signal_psi on signal distributions between periods
    - Compute return_psi on return distributions between periods
    - Classify drift_level based on PSI thresholds
    - Set drift_detected to True if either PSI > 0.2
    - Map drift_level to recommendation: "none" → "signal stable", "moderate" → "monitor closely", "significant" → "consider retraining"
    - Return dict with keys: signal_psi, return_psi, drift_detected, drift_level, recommendation, reference_period, recent_period
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 5.12, 5.13, 5.14, 5.15_

  - [ ]* 3.5 Write property test for drift detection window splitting
    - **Property 7: Drift Detection Window Splitting**
    - **Validates: Requirements 5.1, 5.2**
    - Generate random OHLCV data with sufficient length using Hypothesis
    - Verify detect() splits data into reference period (first reference_window days) and recent period (last detection_window days)
    - Verify both periods have expected number of samples
    - Tag: `# Feature: shap-explainability-layer, Property 7: Drift Detection Window Splitting`

  - [ ]* 3.6 Write property test for drift detection output schema
    - **Property 8: Drift Detection Output Schema**
    - **Validates: Requirements 5.6, 5.7, 5.8, 5.11, 5.12**
    - Generate random Signal and OHLCV data using Hypothesis
    - Verify detect() returns dict with all required keys and correct types
    - Keys: signal_psi, return_psi, drift_detected, drift_level, recommendation, reference_period, recent_period
    - Tag: `# Feature: shap-explainability-layer, Property 8: Drift Detection Output Schema`

  - [ ]* 3.7 Write property test for drift detection logic
    - **Property 9: Drift Detection Logic**
    - **Validates: Requirements 5.9, 5.10**
    - Generate random PSI values for signal_psi and return_psi using Hypothesis
    - Verify drift_detected is True if and only if either signal_psi > 0.2 or return_psi > 0.2
    - Tag: `# Feature: shap-explainability-layer, Property 9: Drift Detection Logic`

  - [ ]* 3.8 Write property test for recommendation mapping
    - **Property 10: Recommendation Mapping**
    - **Validates: Requirements 5.13, 5.14, 5.15**
    - Generate random drift_level values using Hypothesis
    - Verify recommendation is "signal stable" if drift_level is "none"
    - Verify recommendation is "monitor closely" if drift_level is "moderate"
    - Verify recommendation is "consider retraining" if drift_level is "significant"
    - Tag: `# Feature: shap-explainability-layer, Property 10: Recommendation Mapping`

  - [x] 3.9 Implement rolling PSI analysis
    - Implement `rolling_psi` method in DriftDetector class
    - Accept ohlcv_data and metric parameter ("signal" or "returns")
    - Compute PSI on rolling windows across OHLCV data
    - Use reference_window as reference period size and detection_window as comparison period size
    - Return Series with DatetimeIndex (window end dates) and PSI values as floats
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 3.10 Write property test for rolling PSI output structure
    - **Property 11: Rolling PSI Output Structure**
    - **Validates: Requirements 6.4, 6.5, 6.6**
    - Generate random OHLCV data with sufficient length using Hypothesis
    - Verify rolling_psi returns Series with DatetimeIndex and float values
    - Verify one PSI value for each valid rolling window pair
    - Tag: `# Feature: shap-explainability-layer, Property 11: Rolling PSI Output Structure`

  - [ ]* 3.11 Write unit tests for DriftDetector
    - Create `backend/tests/test_drift_detector.py`
    - Test compute_psi with hand-calculated examples (identical distributions, known drift)
    - Test drift classification at threshold boundaries (PSI = 0.09, 0.1, 0.2, 0.21)
    - Test detect() with MomentumSignal and verify output structure
    - Test rolling_psi with sufficient data and verify Series structure
    - Test error handling for insufficient data
    - Test error handling for invalid window parameters

- [ ] 4. Checkpoint - Verify DriftDetector implementation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement ReportBuilder core functionality
  - [ ] 5.1 Create ReportBuilder class with PDF generation
    - Create `backend/shap_layer/report_builder.py` with ReportBuilder class
    - Implement `__init__` method accepting output_dir, page_width, page_height parameters
    - Implement `build` method accepting strategy_name, hypothesis, backtest_results, metrics, shap_results, drift_results, generated_code, ohlcv_data
    - Create output directory if it doesn't exist
    - Generate PDF filename as {strategy_name}_{date}.pdf
    - Return full file path as string
    - _Requirements: 7.1, 7.11, 7.12, 7.13_

  - [ ]* 5.2 Write property test for PDF file creation and naming
    - **Property 12: PDF File Creation and Naming**
    - **Validates: Requirements 7.1, 7.11, 7.12, 7.13**
    - Generate random strategy names and dates using Hypothesis
    - Verify build() creates PDF file in output directory
    - Verify filename format matches {strategy_name}_{date}.pdf
    - Verify method returns full file path as string
    - Verify output directory is created if it doesn't exist
    - Tag: `# Feature: shap-explainability-layer, Property 12: PDF File Creation and Naming`

  - [ ] 5.3 Implement report header and performance summary
    - Implement `_add_header` method adding strategy name, hypothesis, date range, generation timestamp
    - Implement `_add_performance_summary` method creating table with all metrics
    - Display metrics: Sharpe ratio, Sortino ratio, max drawdown, CAGR, win rate, total return, volatility, Calmar ratio
    - Handle None values by displaying "N/A"
    - _Requirements: 7.2, 7.3_

  - [ ] 5.4 Implement equity curve chart generation
    - Implement `_add_equity_curve` method using matplotlib
    - Plot portfolio_value over time from backtest_results
    - Label x-axis as "Date" and y-axis as "Portfolio Value ($)"
    - Save chart as temporary image file
    - Embed image in PDF with dimensions 150mm × 80mm
    - _Requirements: 7.4, 8.1, 8.2, 8.7, 8.8_

  - [ ] 5.5 Implement SHAP feature importance chart
    - Implement `_add_shap_analysis` method creating horizontal bar chart
    - Sort features by absolute importance descending
    - Display top 10 features maximum
    - Label x-axis as "Mean |SHAP Value|" and y-axis as "Feature"
    - Save chart as temporary image file
    - Embed image in PDF with dimensions 150mm × 100mm
    - Add SHAP summary text describing top 3 features
    - _Requirements: 7.5, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

  - [ ]* 5.6 Write property test for feature importance chart limiting
    - **Property 14: Feature Importance Chart Limiting**
    - **Validates: Requirements 8.4, 8.5**
    - Generate random SHAP results with varying feature counts using Hypothesis
    - Verify feature importance chart displays at most 10 features
    - Verify features sorted by absolute importance descending
    - Tag: `# Feature: shap-explainability-layer, Property 14: Feature Importance Chart Limiting`

  - [ ] 5.7 Implement drift analysis section
    - Implement `_add_drift_analysis` method displaying PSI values with color-coded status
    - Color code: green (0, 128, 0) for "none", yellow (255, 200, 0) for "moderate", red (255, 0, 0) for "significant"
    - Display signal_psi, return_psi, drift_level, recommendation
    - Display reference and recent period dates
    - _Requirements: 7.6, 7.7, 7.8, 7.9_

  - [ ] 5.8 Implement code section with syntax highlighting
    - Implement `_add_code_section` method displaying generated signal code
    - Use monospace font (Courier, 8pt)
    - Add section title "Signal Implementation"
    - Format code block with proper indentation
    - _Requirements: 7.10_

  - [ ]* 5.9 Write property test for report content completeness
    - **Property 13: Report Content Completeness**
    - **Validates: Requirements 7.2, 7.3**
    - Generate random valid inputs using Hypothesis
    - Verify generated PDF contains all required sections
    - Sections: header, performance summary, equity curve, SHAP analysis, drift analysis, code
    - Tag: `# Feature: shap-explainability-layer, Property 13: Report Content Completeness`

  - [ ]* 5.10 Write unit tests for ReportBuilder
    - Create `backend/tests/test_report_builder.py`
    - Test build() creates PDF file with correct filename
    - Test _add_header with strategy name and hypothesis
    - Test _add_performance_summary with metrics dict
    - Test _add_equity_curve with portfolio_value Series
    - Test _add_shap_analysis with shap_results dict
    - Test _add_drift_analysis with drift_results dict
    - Test _add_code_section with Python code string
    - Test error handling for missing required data
    - Test error handling for chart generation failures

- [ ] 6. Checkpoint - Verify ReportBuilder implementation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Update dependencies and integration
  - [ ] 7.1 Update requirements.txt with SHAP layer dependencies
    - Add shap>=0.44.0 to backend/requirements.txt
    - Add lightgbm>=4.0.0 to backend/requirements.txt
    - Add fpdf2>=2.7.0 to backend/requirements.txt
    - Add scikit-learn>=1.3.0 to backend/requirements.txt
    - Add matplotlib>=3.7.0 to backend/requirements.txt
    - Add hypothesis>=6.82.0 for property-based testing
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 7.2 Write property test for SignalBase compatibility
    - **Property 15: SignalBase Compatibility**
    - **Validates: Requirements 9.2, 9.3**
    - Generate random SignalBase subclasses using Hypothesis
    - Verify SignalExplainer accepts any Signal instance inheriting from SignalBase
    - Verify SignalExplainer works with OHLCV data in same format as BacktestEngine
    - Tag: `# Feature: shap-explainability-layer, Property 15: SignalBase Compatibility`

  - [ ]* 7.3 Write end-to-end integration test
    - Create integration test in `backend/tests/test_explainer.py`
    - Test complete workflow: BacktestEngine → MetricsCalculator → SignalExplainer → DriftDetector → ReportBuilder
    - Use MomentumSignal with real OHLCV data
    - Verify PDF report is generated with all sections
    - Verify report file exists and has non-zero size
    - _Requirements: 9.1, 9.4, 9.5, 9.6_

- [ ] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using Hypothesis library (minimum 100 iterations)
- Unit tests validate specific examples and edge cases
- All property tests must include the tag format: `# Feature: shap-explainability-layer, Property {N}: {property_text}`
- Feature engineering uses exact formulas specified in design document
- PSI computation uses epsilon (1e-10) to avoid division by zero
- PDF layout follows A4 dimensions (210mm × 297mm) with 10mm margins
