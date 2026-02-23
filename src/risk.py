from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskManager:
    risk_per_trade: float = 0.005
    max_daily_loss_pct: float = 0.02
    max_concurrent_trades: int = 3

    def risk_amount(self, equity: float) -> float:
        return max(0.0, equity * self.risk_per_trade)

    def can_trade(self, day_pnl: float, equity_start_of_day: float, concurrent_trades: int) -> bool:
        daily_loss_limit = -equity_start_of_day * self.max_daily_loss_pct
        return (day_pnl > daily_loss_limit) and (concurrent_trades < self.max_concurrent_trades)


def calculate_position_size(equity: float, risk_fraction: float, entry_price: float, stop_price: float) -> int:
    stop_distance = max(entry_price - stop_price, 0.0)
    if stop_distance <= 0:
        return 0
    risk_amount = equity * risk_fraction
    return max(int(risk_amount // stop_distance), 0)
