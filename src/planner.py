from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from risk import calculate_position_size


@dataclass
class TradePlan:
    symbol: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    target_price: float
    position_size: int
    confidence_score: float
    risk_per_share: float

    def to_dict(self) -> dict:
        return asdict(self)


def _confidence_score(row: pd.Series) -> float:
    breakout_strength = max((row["close"] - row["or_high"]) / max(row["or_high"], 1e-9), 0.0)
    relvol_strength = max((row["rel_volume"] - 1.5) / 2.0, 0.0)
    index_strength = max((row["spy_close"] - row["spy_vwap"]) / max(row["spy_vwap"], 1e-9), 0.0)
    score = 100.0 * (0.5 * breakout_strength + 0.35 * relvol_strength + 0.15 * index_strength)
    return float(np.clip(score, 0, 100))


def generate_trade_plans(symbol: str, signal_df: pd.DataFrame, equity: float, risk_fraction: float) -> list[TradePlan]:
    plans: list[TradePlan] = []
    signal_indices = signal_df.index[signal_df["entry_signal"]].tolist()

    for idx in signal_indices:
        if idx + 1 >= len(signal_df):
            continue
        entry_row = signal_df.iloc[idx + 1]
        signal_row = signal_df.iloc[idx]

        entry_price = float(entry_row["open"])
        stop_loss = float(min(signal_row["or_low"], entry_price - signal_row["intraday_atr_proxy"]))
        risk_per_share = max(entry_price - stop_loss, 0.0)
        if risk_per_share <= 0:
            continue

        position_size = calculate_position_size(
            equity=equity,
            risk_fraction=risk_fraction,
            entry_price=entry_price,
            stop_price=stop_loss,
        )
        if position_size <= 0:
            continue

        target_price = entry_price + risk_per_share
        confidence = _confidence_score(signal_row)

        plans.append(
            TradePlan(
                symbol=symbol,
                signal_time=signal_row["datetime"],
                entry_time=entry_row["datetime"],
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                position_size=position_size,
                confidence_score=confidence,
                risk_per_share=risk_per_share,
            )
        )

    return plans
