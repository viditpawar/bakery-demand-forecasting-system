from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def preprocess_sales(
    input_csv: Path,
    output_csv: Path,
    date_col: str = "date",
    quantity_col: str = "Quantity",
    item_col: str = "article",
) -> None:
    """
    Preprocess raw bakery transactions into daily totals per item.

    Expected columns in input:
      - date (string or datetime)
      - article (item name)
      - Quantity (numeric)

    Output columns:
      - ds (date)
      - item (article)
      - y (daily total quantity)
    """
    df = pd.read_csv(input_csv)

    # Normalize column names (some Kaggle datasets vary)
    # Try common variants
    col_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc == "date":
            col_map[c] = "date"
        elif lc in ("article", "item", "product"):
            col_map[c] = "article"
        elif lc in ("quantity", "qty", "quantite"):
            col_map[c] = "Quantity"
    df = df.rename(columns=col_map)

    if "date" not in df.columns or "article" not in df.columns or "Quantity" not in df.columns:
        raise ValueError(
            f"Missing required columns. Found: {list(df.columns)}. "
            "Need date, article, Quantity (or common variants)."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date", "article", "Quantity"])

    # Clean Quantity
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["Quantity"])
    df = df[df["Quantity"] > 0]

    # Aggregate daily totals per item
    daily = (
        df.groupby(["date", "article"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"date": "ds", "article": "item", "Quantity": "y"})
    )

    daily["ds"] = pd.to_datetime(daily["ds"])  # Prophet expects datetime
    daily = daily.sort_values(["item", "ds"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_csv, index=False)
    print(f"✅ Wrote preprocessed data: {output_csv} (rows={len(daily)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw transactions CSV")
    parser.add_argument("--output", required=True, help="Path to output daily CSV")
    args = parser.parse_args()

    preprocess_sales(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
