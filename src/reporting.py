from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_trade_outputs(
    trade_plans: list[dict],
    trades_df: pd.DataFrame,
    metrics: pd.Series,
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    plans_df = pd.DataFrame(trade_plans)
    plans_path = reports_dir / "trade_plans.csv"
    trades_path = reports_dir / "executed_trades.csv"
    metrics_path = reports_dir / "summary_metrics.csv"

    plans_df.to_csv(plans_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    metrics.to_frame(name="value").to_csv(metrics_path)


def plot_equity_curve(trades_df: pd.DataFrame, initial_capital: float, reports_dir: Path) -> Path | None:
    if trades_df.empty:
        return None

    reports_dir.mkdir(parents=True, exist_ok=True)
    curve = trades_df[["exit_time", "pnl"]].copy()
    curve["exit_time"] = pd.to_datetime(curve["exit_time"])
    curve = curve.sort_values("exit_time")
    curve["equity"] = initial_capital + curve["pnl"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curve["exit_time"], curve["equity"], linewidth=1.5)
    ax.set_title("A-ORB Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)

    out_path = reports_dir / "equity_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def print_summary(metrics: pd.Series) -> None:
    printable = metrics.copy()
    if "win_rate" in printable:
        printable["win_rate"] = f"{printable['win_rate']:.2%}"
    if "max_drawdown" in printable:
        printable["max_drawdown"] = f"{printable['max_drawdown']:.2%}"
    print("\n===== BACKTEST SUMMARY =====")
    print(printable.to_string())
