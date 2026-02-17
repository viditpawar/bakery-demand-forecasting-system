from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import pandas as pd


SEASONS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "fall": (9, 10, 11),
}


@dataclass
class ForecastResult:
    item: str
    season: str
    year: int
    predicted_total: float


def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def forecast_season(
    model_path: Path,
    item: str,
    season: str,
    year: int,
) -> ForecastResult:
    season = season.lower().strip()
    if season not in SEASONS:
        raise ValueError(f"Invalid season '{season}'. Use one of: {list(SEASONS.keys())}")

    model = load_model(model_path)

    # Create daily dates for the target season in the requested year
    months = SEASONS[season]
    start = pd.Timestamp(year=year, month=months[0], day=1)

    # End date logic: pick the last day of the last month in the season
    end_month = months[-1]
    end = (pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0))

    # Handle winter spanning year boundary (Dec + Jan + Feb)
    if season == "winter":
        start = pd.Timestamp(year=year, month=12, day=1)
        end = (pd.Timestamp(year=year + 1, month=2, day=1) + pd.offsets.MonthEnd(0))

    dates = pd.date_range(start=start, end=end, freq="D")
    future = pd.DataFrame({"ds": dates})
    pred = model.predict(future)
    pred_total = float(pred["yhat"].clip(lower=0).sum())

    return ForecastResult(item=item, season=season, year=year, predicted_total=pred_total)
