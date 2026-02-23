from __future__ import annotations

import numpy as np
import pandas as pd


def compute_daily_atr(daily_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = daily_df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(period, min_periods=period).mean()
    df["atr14_pct"] = df["atr14"] / df["close"]
    return df


def compute_intraday_vwap(minute_df: pd.DataFrame) -> pd.Series:
    typical = (minute_df["high"] + minute_df["low"] + minute_df["close"]) / 3.0
    pv = typical * minute_df["volume"]
    grouped_pv = pv.groupby(minute_df["session_date"]).cumsum()
    grouped_vol = minute_df["volume"].groupby(minute_df["session_date"]).cumsum().replace(0, np.nan)
    return grouped_pv / grouped_vol


def compute_relative_volume(minute_df: pd.DataFrame, lookback_days: int = 20) -> pd.Series:
    cumulative_volume = minute_df["volume"].groupby(minute_df["session_date"]).cumsum()
    minute_slot = minute_df["datetime"].dt.strftime("%H:%M")
    baseline = cumulative_volume.groupby(minute_slot).transform(
        lambda s: s.shift(1).rolling(lookback_days, min_periods=5).mean()
    )
    return cumulative_volume / baseline.replace(0, np.nan)


def compute_opening_range(minute_df: pd.DataFrame, start: str = "09:30", end: str = "09:45") -> pd.DataFrame:
    df = minute_df.copy()
    intraday_time = df["datetime"].dt.strftime("%H:%M")
    in_or = (intraday_time >= start) & (intraday_time < end)

    or_levels = (
        df.loc[in_or, ["session_date", "high", "low"]]
        .groupby("session_date", as_index=False)
        .agg(or_high=("high", "max"), or_low=("low", "min"))
    )
    return df.merge(or_levels, on="session_date", how="left")


def compute_intraday_range(minute_df: pd.DataFrame) -> pd.Series:
    session_high = minute_df["high"].groupby(minute_df["session_date"]).cummax()
    session_low = minute_df["low"].groupby(minute_df["session_date"]).cummin()
    return session_high - session_low


def compute_ema(series: pd.Series, period: int = 9) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def build_feature_frame(
    minute_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    opening_range_start: str,
    opening_range_end: str,
    relative_volume_lookback: int,
    ema_period: int = 9,
) -> pd.DataFrame:
    daily_with_atr = compute_daily_atr(daily_df)
    daily_with_atr["atr14_prior"] = daily_with_atr["atr14"].shift(1)
    daily_with_atr["atr14_pct_prior"] = daily_with_atr["atr14_pct"].shift(1)
    daily_with_atr["avg_dollar_vol20"] = (
        (daily_with_atr["close"] * daily_with_atr["volume"]).rolling(20, min_periods=20).mean().shift(1)
    )

    df = minute_df.merge(
        daily_with_atr[["date", "atr14_prior", "atr14_pct_prior", "avg_dollar_vol20", "close"]].rename(
            columns={"close": "daily_close"}
        ),
        how="left",
        left_on="session_date",
        right_on="date",
    )

    df["vwap"] = compute_intraday_vwap(df)
    df["rel_volume"] = compute_relative_volume(df, lookback_days=relative_volume_lookback)
    df = compute_opening_range(df, start=opening_range_start, end=opening_range_end)
    df["intraday_range"] = compute_intraday_range(df)
    df["ema9"] = df.groupby("session_date")["close"].transform(lambda s: compute_ema(s, period=ema_period))
    df["intraday_atr_proxy"] = (
        (df["high"] - df["low"]).groupby(df["session_date"]).transform(lambda s: s.rolling(14, min_periods=3).mean())
    )

    return df
