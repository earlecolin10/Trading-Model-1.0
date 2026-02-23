# Adaptive Opening Range Breakout (A-ORB) Intraday Trading Model

Production-ready, modular intraday system for US equities implementing a locked specification Adaptive Opening Range Breakout strategy.

## Strategy Overview

### Universe filter (daily)
- Price > $5
- 20-day average dollar volume > $25M
- ATR(14) > 1.5% of price

### Opening Range
- First 15 minutes (09:30-09:45 ET)
- Uses `OR_high` and `OR_low`

### Long entry trigger
All must hold:
1. Close > OR_high
2. Relative volume > 1.5
3. Price > intraday VWAP
4. SPY > intraday VWAP
5. Intraday range >= 0.8 x daily ATR(14)

### Trade management
- Enter at next bar open
- Stop: `min(OR_low, entry - 1.0 * intraday ATR proxy)`
- Initial target: 1R
- Post-1R adaptive logic:
  - If above VWAP + elevated volume + SPY trend: trail using 9 EMA
  - Else: exit at 1.5R
- Hard stop, adaptive exit, and force flat by market close

## Project structure

```text
intraday-trading-model/
├── src/
│   ├── config.py
│   ├── data_ingest.py
│   ├── features.py
│   ├── setup.py
│   ├── risk.py
│   ├── planner.py
│   ├── backtester.py
│   ├── reporting.py
│   └── main.py
├── data/
│   ├── minute/
│   └── daily/
├── backtests/
├── reports/
├── requirements.txt
└── README.md
```

## Data format

### Minute files: `data/minute/<SYMBOL>.csv`
Required columns:
- `datetime` (timestamp)
- `open, high, low, close, volume`

### Daily files: `data/daily/<SYMBOL>.csv`
Required columns:
- `date`
- `open, high, low, close, volume`

Rules:
- One file per symbol in each folder
- Include `SPY.csv` in both minute and daily
- Chronological order is strictly validated
- Duplicate timestamps/dates raise errors
- Leakage guard: minute dates cannot exceed last daily date

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

## Outputs

Written to `reports/`:
- `trade_plans.csv`
- `executed_trades.csv`
- `summary_metrics.csv`
- `equity_curve.png`

## Backtest metrics

The engine outputs:
- expectancy
- win rate
- profit factor
- max drawdown
- sharpe
- average R
- average trade duration
- total trades
- ending equity

## Notes

- No lookahead bias in signal confirmation and entries.
- Vectorized features where applicable.
- No overnight positions.
- Slippage defaults to 5 bps; commissions configurable in `src/config.py`.
