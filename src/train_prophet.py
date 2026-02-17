from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class TrainResult:
    item: str
    mae: float
    rmse: float


def train_one_item(df_item: pd.DataFrame, holdout_days: int = 30) -> tuple[Prophet, TrainResult]:
    df_item = df_item.sort_values("ds")
    if df_item["ds"].nunique() < (holdout_days + 60):
        # Not enough data; still train but skip holdout metrics reliably
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df_item[["ds", "y"]])
        return m, TrainResult(item=str(df_item["item"].iloc[0]), mae=float("nan"), rmse=float("nan"))

    cutoff = df_item["ds"].max() - pd.Timedelta(days=holdout_days)
    train = df_item[df_item["ds"] <= cutoff]
    valid = df_item[df_item["ds"] > cutoff]

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(train[["ds", "y"]])

    future = valid[["ds"]].copy()
    pred = m.predict(future)

    y_true = valid["y"].values
    y_pred = pred["yhat"].clip(lower=0).values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return m, TrainResult(item=str(df_item["item"].iloc[0]), mae=mae, rmse=rmse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to preprocessed daily CSV (ds,item,y)")
    parser.add_argument("--out", required=True, help="Output directory for models + metrics")
    parser.add_argument("--top_n", type=int, default=10, help="Train models for top N items by total volume")
    parser.add_argument("--holdout_days", type=int, default=30, help="Holdout days for validation metrics")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df["ds"] = pd.to_datetime(df["ds"])

    # Pick top N items by total sales volume (keeps project manageable)
    totals = df.groupby("item")["y"].sum().sort_values(ascending=False)
    top_items = totals.head(args.top_n).index.tolist()

    metrics: list[dict] = []
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for item in top_items:
        df_item = df[df["item"] == item][["ds", "item", "y"]].copy()
        model, result = train_one_item(df_item, holdout_days=args.holdout_days)

        model_path = models_dir / f"prophet_{safe_name(item)}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        metrics.append({"item": item, "mae": result.mae, "rmse": result.rmse})
        print(f"✅ Trained: {item} -> {model_path.name} (MAE={result.mae:.2f} RMSE={result.rmse:.2f})")

    metrics_df = pd.DataFrame(metrics).sort_values("mae")
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✅ Wrote metrics: {metrics_path}")


def safe_name(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")


if __name__ == "__main__":
    main()
