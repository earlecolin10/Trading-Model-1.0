from __future__ import annotations

import pandas as pd


def detect_orb_long(symbol_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    df = symbol_df.copy()
    spy_cols = spy_df[["datetime", "close", "vwap", "ema9"]].rename(
        columns={"close": "spy_close", "vwap": "spy_vwap", "ema9": "spy_ema9"}
    )
    df = df.merge(spy_cols, on="datetime", how="left")

    trade_time = df["datetime"].dt.strftime("%H:%M") >= "09:45"
    breakout = df["close"] > df["or_high"]
    relvol_ok = df["rel_volume"] > 1.5
    above_vwap = df["close"] > df["vwap"]
    spy_ok = df["spy_close"] > df["spy_vwap"]
    range_expansion = df["intraday_range"] >= (0.8 * df["atr14_prior"])

    df["entry_signal"] = trade_time & breakout & relvol_ok & above_vwap & spy_ok & range_expansion

    df["universe_ok"] = (
        (df["daily_close"] > 5.0)
        & (df["avg_dollar_vol20"] > 25_000_000)
        & (df["atr14_pct_prior"] > 0.015)
    )

    df["entry_signal"] = df["entry_signal"] & df["universe_ok"]
    return df
