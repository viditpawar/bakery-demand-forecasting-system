from __future__ import annotations

import os
from pathlib import Path
from flask import Flask, jsonify, request

from forecast_service import forecast_season


app = Flask(__name__)

# Where models live in the repo (or mounted volume in cloud)
MODELS_DIR = Path(os.getenv("MODELS_DIR", "outputs/models")).resolve()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/forecast")
def forecast():
    """
    Example:
      /forecast?item=baguette&season=summer&year=2026
    """
    item = request.args.get("item", "").strip()
    season = request.args.get("season", "").strip()
    year_str = request.args.get("year", "").strip()

    if not item or not season or not year_str:
        return jsonify({"error": "Missing required params: item, season, year"}), 400

    try:
        year = int(year_str)
    except ValueError:
        return jsonify({"error": "year must be an integer"}), 400

    model_path = MODELS_DIR / f"prophet_{safe_name(item)}.pkl"
    if not model_path.exists():
        return jsonify({
            "error": "Model not found for item",
            "item": item,
            "expected_model_path": str(model_path),
            "hint": "Train models first: python src/train_prophet.py --data <daily.csv> --out outputs"
        }), 404

    try:
        result = forecast_season(model_path=model_path, item=item, season=season, year=year)
        return jsonify({
            "item": result.item,
            "season": result.season,
            "year": result.year,
            "predicted_total": round(result.predicted_total, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def safe_name(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")


if __name__ == "__main__":
    # Local dev server
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
