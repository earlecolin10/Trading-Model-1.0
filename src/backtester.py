from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from risk import RiskManager


@dataclass
class ExecutedTrade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    r_multiple: float
    duration_bars: int
    exit_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _apply_slippage(price: float, bps: float, is_buy: bool) -> float:
    side = 1 if is_buy else -1
    return price * (1 + side * (bps / 10_000.0))


def run_backtest(
    feature_map: dict[str, pd.DataFrame],
    planned_trades: list,
    initial_capital: float,
    slippage_bps: float,
    commission_per_share: float,
    risk_manager: RiskManager,
) -> tuple[pd.DataFrame, pd.Series]:
    capital = initial_capital
    equity_curve = []
    executed: list[ExecutedTrade] = []

    plans_sorted = sorted(planned_trades, key=lambda p: p.entry_time)
    current_day = None
    day_start_equity = initial_capital
    day_pnl = 0.0

    for plan in plans_sorted:
        entry_day = plan.entry_time.date()
        if current_day != entry_day:
            current_day = entry_day
            day_start_equity = capital
            day_pnl = 0.0

        if not risk_manager.can_trade(day_pnl=day_pnl, equity_start_of_day=day_start_equity, concurrent_trades=0):
            continue

        df = feature_map[plan.symbol]
        post_entry = df[df["datetime"] >= plan.entry_time].copy()
        if post_entry.empty:
            continue

        entry_px = _apply_slippage(plan.entry_price, slippage_bps, is_buy=True)
        stop_px = plan.stop_loss
        one_r_target = plan.target_price
        extended_target = entry_px + (1.5 * (entry_px - stop_px))

        hit_1r = False
        exit_px = None
        exit_time = None
        exit_reason = ""
        duration = 0

        for i, row in post_entry.iterrows():
            duration += 1
            low = float(row["low"])
            high = float(row["high"])

            if low <= stop_px:
                exit_px = _apply_slippage(stop_px, slippage_bps, is_buy=False)
                exit_time = row["datetime"]
                exit_reason = "hard_stop"
                break

            if not hit_1r and high >= one_r_target:
                hit_1r = True

            if hit_1r:
                adaptive_conditions = (
                    (row["close"] > row["vwap"])
                    and (row["rel_volume"] > 1.5)
                    and (row.get("spy_close", row["close"]) > row.get("spy_ema9", row["ema9"]))
                )
                if adaptive_conditions:
                    stop_px = max(stop_px, float(row["ema9"]))
                elif high >= extended_target:
                    exit_px = _apply_slippage(extended_target, slippage_bps, is_buy=False)
                    exit_time = row["datetime"]
                    exit_reason = "adaptive_1_5r"
                    break

            if row["datetime"].strftime("%H:%M") >= "15:59":
                exit_px = _apply_slippage(float(row["close"]), slippage_bps, is_buy=False)
                exit_time = row["datetime"]
                exit_reason = "eod_flatten"
                break

        if exit_px is None:
            last_row = post_entry.iloc[-1]
            exit_px = _apply_slippage(float(last_row["close"]), slippage_bps, is_buy=False)
            exit_time = last_row["datetime"]
            exit_reason = "data_end"

        gross_pnl = (exit_px - entry_px) * plan.position_size
        fees = plan.position_size * commission_per_share * 2
        net_pnl = gross_pnl - fees

        risk_per_share = max(entry_px - plan.stop_loss, 1e-9)
        r_multiple = (exit_px - entry_px) / risk_per_share

        capital += net_pnl
        day_pnl += net_pnl
        equity_curve.append({"datetime": exit_time, "equity": capital})

        executed.append(
            ExecutedTrade(
                symbol=plan.symbol,
                entry_time=plan.entry_time,
                exit_time=exit_time,
                entry_price=entry_px,
                exit_price=exit_px,
                shares=plan.position_size,
                pnl=net_pnl,
                r_multiple=r_multiple,
                duration_bars=duration,
                exit_reason=exit_reason,
            )
        )

    trades_df = pd.DataFrame([t.to_dict() for t in executed])
    equity_df = pd.DataFrame(equity_curve)

    metrics = compute_backtest_metrics(trades_df, equity_df, initial_capital=initial_capital)
    return trades_df, metrics


def compute_backtest_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float) -> pd.Series:
    if trades_df.empty:
        return pd.Series(
            {
                "expectancy": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "average_r": 0.0,
                "trade_duration": 0.0,
                "total_trades": 0,
                "ending_equity": initial_capital,
            }
        )

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    expectancy = float(trades_df["pnl"].mean())
    win_rate = float((trades_df["pnl"] > 0).mean())
    profit_factor = float(wins["pnl"].sum() / abs(losses["pnl"].sum())) if not losses.empty else np.inf

    if equity_df.empty:
        max_dd = 0.0
        sharpe = 0.0
        ending_equity = initial_capital
    else:
        eq = equity_df["equity"].astype(float)
        running_max = eq.cummax()
        drawdown = (eq - running_max) / running_max
        max_dd = float(drawdown.min())
        returns = eq.pct_change().dropna()
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0
        ending_equity = float(eq.iloc[-1])

    return pd.Series(
        {
            "expectancy": expectancy,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "average_r": float(trades_df["r_multiple"].mean()),
            "trade_duration": float(trades_df["duration_bars"].mean()),
            "total_trades": int(len(trades_df)),
            "ending_equity": ending_equity,
        }
    )
