# ADR-001: Backtesting Engine Design Decisions

## Status: Accepted

## Context
Month 1 scope — core backtesting engine for QuantSignal Arena.

## Decisions

### vectorbt over zipline or backtrader
Chose vectorbt for vectorised numpy operations. 10-year daily backtest completes in under 5 seconds. Zipline is unmaintained. Backtrader is loop-based and slow.

### parquet over CSV for caching
Parquet gives 10x faster reads and 5x smaller files than CSV for OHLCV data. Essential when loading 10 years of multi-ticker data repeatedly.

### walk-forward over simple train/test split
Simple splits leak future information in time series. Walk-forward with gap period prevents lookahead bias — critical for honest signal evaluation.

### abstract SignalBase over duck typing
Enforces interface contract at class definition time. Catches missing generate_signals() at import, not at 2am when a backtest crashes.

## Consequences
- All signals must inherit SignalBase — small overhead, large reliability gain
- vectorbt version pinned in requirements.txt — API changes between versions
