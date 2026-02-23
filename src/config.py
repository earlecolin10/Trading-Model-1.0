from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    data_dir: Path = Path("data")
    minute_dir_name: str = "minute"
    daily_dir_name: str = "daily"
    reports_dir: Path = Path("reports")
    backtests_dir: Path = Path("backtests")

    universe_min_price: float = 5.0
    universe_min_avg_dollar_vol: float = 25_000_000.0
    universe_min_atr_pct: float = 0.015

    opening_range_start: str = "09:30"
    opening_range_end: str = "09:45"
    relative_volume_lookback: int = 20

    risk_per_trade: float = 0.005
    max_daily_loss_pct: float = 0.02
    max_concurrent_trades: int = 3

    initial_capital: float = 100_000.0
    slippage_bps: float = 5.0
    commission_per_share: float = 0.005

    force_exit_time: str = "15:59"
    bar_interval_minutes: int = 1
    ema_trail_period: int = 9


DEFAULT_CONFIG = ModelConfig()
