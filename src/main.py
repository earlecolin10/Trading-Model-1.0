from __future__ import annotations

from pathlib import Path

from backtester import run_backtest
from config import DEFAULT_CONFIG
from data_ingest import load_dataset
from features import build_feature_frame
from planner import generate_trade_plans
from reporting import plot_equity_curve, print_summary, save_trade_outputs
from risk import RiskManager
from setup import detect_orb_long


def run_pipeline(data_dir: Path | None = None) -> None:
    cfg = DEFAULT_CONFIG
    data_root = data_dir or cfg.data_dir

    dataset = load_dataset(data_root)

    feature_map = {}
    spy_features = build_feature_frame(
        minute_df=dataset["SPY"]["minute"],
        daily_df=dataset["SPY"]["daily"],
        opening_range_start=cfg.opening_range_start,
        opening_range_end=cfg.opening_range_end,
        relative_volume_lookback=cfg.relative_volume_lookback,
        ema_period=cfg.ema_trail_period,
    )
    feature_map["SPY"] = spy_features

    for symbol, payload in dataset.items():
        if symbol == "SPY":
            continue
        feature_map[symbol] = build_feature_frame(
            minute_df=payload["minute"],
            daily_df=payload["daily"],
            opening_range_start=cfg.opening_range_start,
            opening_range_end=cfg.opening_range_end,
            relative_volume_lookback=cfg.relative_volume_lookback,
            ema_period=cfg.ema_trail_period,
        )

    all_plans = []
    for symbol, df in feature_map.items():
        if symbol == "SPY":
            continue
        signal_df = detect_orb_long(df, spy_features)
        feature_map[symbol] = signal_df
        plans = generate_trade_plans(
            symbol=symbol,
            signal_df=signal_df,
            equity=cfg.initial_capital,
            risk_fraction=cfg.risk_per_trade,
        )
        all_plans.extend(plans)

    risk_manager = RiskManager(
        risk_per_trade=cfg.risk_per_trade,
        max_daily_loss_pct=cfg.max_daily_loss_pct,
        max_concurrent_trades=cfg.max_concurrent_trades,
    )

    trades_df, metrics = run_backtest(
        feature_map=feature_map,
        planned_trades=all_plans,
        initial_capital=cfg.initial_capital,
        slippage_bps=cfg.slippage_bps,
        commission_per_share=cfg.commission_per_share,
        risk_manager=risk_manager,
    )

    save_trade_outputs(
        trade_plans=[p.to_dict() for p in all_plans],
        trades_df=trades_df,
        metrics=metrics,
        reports_dir=cfg.reports_dir,
    )
    plot_equity_curve(trades_df=trades_df, initial_capital=cfg.initial_capital, reports_dir=cfg.reports_dir)
    print_summary(metrics)


if __name__ == "__main__":
    # Example execution:
    # 1) Place minute CSVs in data/minute and daily CSVs in data/daily (one file per symbol, include SPY).
    # 2) Run: python src/main.py
    run_pipeline()
