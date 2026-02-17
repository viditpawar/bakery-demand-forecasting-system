#!/usr/bin/env bash
set -euo pipefail

# Example usage:
#   bash scripts/retrain.sh data/raw_transactions.csv

RAW="${1:-}"
if [[ -z "$RAW" ]]; then
  echo "Usage: bash scripts/retrain.sh <raw_transactions.csv>"
  exit 1
fi

mkdir -p outputs

echo "==> Preprocessing raw data..."
python src/preprocess.py --input "$RAW" --output outputs/daily_sales.csv

echo "==> Training Prophet models (top 10 items)..."
python src/train_prophet.py --data outputs/daily_sales.csv --out outputs --top_n 10 --holdout_days 30

echo "✅ Done. Models in outputs/models"
