from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


REQUIRED_MINUTE_COLS = {"datetime", "open", "high", "low", "close", "volume"}
REQUIRED_DAILY_COLS = {"date", "open", "high", "low", "close", "volume"}


def _validate_columns(df: pd.DataFrame, required: set[str], file_path: Path) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{file_path} is missing required columns: {sorted(missing)}")


def _validate_chronological(df: pd.DataFrame, time_col: str, file_path: Path) -> None:
    if not df[time_col].is_monotonic_increasing:
        raise ValueError(f"{file_path} is not in strict chronological order for {time_col}.")
    if df[time_col].duplicated().any():
        raise ValueError(f"{file_path} has duplicate timestamps in {time_col}.")


def _symbol_from_path(path: Path) -> str:
    return path.stem.upper()


def load_symbol_data(minute_file: Path, daily_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    minute_df = pd.read_csv(minute_file)
    daily_df = pd.read_csv(daily_file)

    _validate_columns(minute_df, REQUIRED_MINUTE_COLS, minute_file)
    _validate_columns(daily_df, REQUIRED_DAILY_COLS, daily_file)

    minute_df["datetime"] = pd.to_datetime(minute_df["datetime"], utc=False)
    minute_df["session_date"] = minute_df["datetime"].dt.date
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date

    minute_df = minute_df.sort_values("datetime").reset_index(drop=True)
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    _validate_chronological(minute_df, "datetime", minute_file)
    _validate_chronological(daily_df, "date", daily_file)

    if minute_df["session_date"].max() > daily_df["date"].max():
        raise ValueError(
            f"Potential leakage: minute data in {minute_file} extends beyond available daily data in {daily_file}."
        )

    return minute_df, daily_df


def load_dataset(data_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    minute_dir = data_dir / "minute"
    daily_dir = data_dir / "daily"

    if not minute_dir.exists() or not daily_dir.exists():
        raise FileNotFoundError("Data folders data/minute and data/daily must both exist.")

    minute_files = sorted(minute_dir.glob("*.csv"))
    if not minute_files:
        raise FileNotFoundError("No minute-level CSV files found in data/minute.")

    dataset: Dict[str, Dict[str, pd.DataFrame]] = {}
    for minute_file in minute_files:
        symbol = _symbol_from_path(minute_file)
        daily_file = daily_dir / f"{symbol}.csv"
        if not daily_file.exists():
            raise FileNotFoundError(f"Missing daily file for {symbol}: {daily_file}")

        minute_df, daily_df = load_symbol_data(minute_file, daily_file)
        dataset[symbol] = {"minute": minute_df, "daily": daily_df}

    if "SPY" not in dataset:
        raise ValueError("SPY data is required for index alignment filter (SPY > VWAP / trending).")

    return dataset
