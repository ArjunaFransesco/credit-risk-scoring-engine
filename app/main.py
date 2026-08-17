"""
Flask Web Application & REST API for Credit Risk Scoring Engine.
Serves interactive borrower risk assessments and real-time inference.
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, jsonify, render_template, request
from src.predict import CreditRiskPredictor

app = Flask(__name__)

# Initialize predictor
try:
    predictor = CreditRiskPredictor(
        model_path=os.path.join(os.path.dirname(__file__), "../models/best_credit_risk_model.joblib"),
        preprocessor_path=os.path.join(os.path.dirname(__file__), "../models/feature_engineer.joblib")
    )
except Exception as e:
    print(f"[!] Warning: Could not initialize model ({e}). Retrying from root paths.")
    predictor = CreditRiskPredictor()

# Load metrics report for stats endpoint
metrics_path = os.path.join(os.path.dirname(__file__), "../reports/model_evaluation_metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        MODEL_STATS = json.load(f)
else:
    MODEL_STATS = {}


@app.route("/")
def home():
    return render_template("index.html", stats=MODEL_STATS)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input payload provided"}), 400

        result = predictor.predict_single(data)
        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(MODEL_STATS)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "Credit Risk Scoring Engine v1.0"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Credit Risk Scoring Engine on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
