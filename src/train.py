"""
Training & Evaluation Pipeline for Credit Risk Scoring Engine.
Trains, benchmarks, tunes, and serializes machine learning models for Probability of Default (PD).
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from data_loader import load_data
from features import CreditFeatureEngineer


def calculate_ks_statistic(y_true, y_prob):
    """
    Computes Kolmogorov-Smirnov (KS) statistic, a gold-standard
    metric in credit scoring to measure separation between goods and bads.
    """
    df = pd.DataFrame({"target": y_true, "prob": y_prob})
    df = df.sort_values(by="prob", ascending=False).reset_index(drop=True)
    
    df["good"] = (df["target"] == 0).astype(int)
    df["bad"] = (df["target"] == 1).astype(int)
    
    df["cum_good"] = df["good"].cumsum() / df["good"].sum()
    df["cum_bad"] = df["bad"].cumsum() / df["bad"].sum()
    
    ks = np.max(np.abs(df["cum_bad"] - df["cum_good"]))
    return float(ks)


def train_and_benchmark(data_path="data/raw/credit_risk_dataset.csv", output_dir="models", reports_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=================================================================")
    print("      CREDIT RISK SCORING ENGINE - MODEL TRAINING PIPELINE       ")
    print("=================================================================")

    # 1. Load Data
    df = load_data(data_path)
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"].values

    # 2. Stratified Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[+] Dataset Split: Train = {len(X_train_raw)} samples, Test = {len(X_test_raw)} samples")
    print(f"[+] Train Default Rate: {np.mean(y_train):.2%}, Test Default Rate: {np.mean(y_test):.2%}")

    # 3. Feature Engineering
    fe = CreditFeatureEngineer()
    X_train = fe.fit_transform(X_train_raw)
    X_test = fe.transform(X_test_raw)

    # Class imbalance weight
    scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

    # 4. Multi-Model Candidates
    models = {
        "Logistic Regression (Scorecard)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.07,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42
        )
    }

    benchmark_results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\n[>] Training and evaluating: {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        roc_auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        ks = calculate_ks_statistic(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_curve, precision_curve)

        benchmark_results[name] = {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "ks_statistic": round(float(ks), 4),
            "f1_score": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "accuracy": round(float(acc), 4),
            "brier_score": round(float(brier), 4)
        }

        print(f"    ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | KS: {ks:.4f} | F1: {f1:.4f} | Recall: {rec:.4f}")

    # Select Best Model based on ROC-AUC + KS
    best_model_name = max(benchmark_results, key=lambda k: benchmark_results[k]["roc_auc"])
    best_model = fitted_models[best_model_name]
    print(f"\n[*] Best Candidate Model: {best_model_name} (ROC-AUC: {benchmark_results[best_model_name]['roc_auc']})")

    # 5. Optimal Threshold Analysis for Best Model
    y_test_probs = best_model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = [f1_score(y_test, (y_test_probs >= t).astype(int), zero_division=0) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[optimal_idx])
    best_f1 = float(f1_scores[optimal_idx])

    print(f"[+] Optimal Classification Threshold: {optimal_threshold:.2f} (Max F1: {best_f1:.4f})")

    # 6. Feature Importances
    feature_importance_dict = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importance_dict = {
            feat: round(float(imp), 4)
            for feat, imp in sorted(zip(fe.feature_names_, importances), key=lambda x: x[1], reverse=True)
        }

    # 7. Serialize Artifacts
    model_file = os.path.join(output_dir, "best_credit_risk_model.joblib")
    preprocessor_file = os.path.join(output_dir, "feature_engineer.joblib")
    metrics_file = os.path.join(reports_dir, "model_evaluation_metrics.json")

    joblib.dump(best_model, model_file)
    joblib.dump(fe, preprocessor_file)

    report_payload = {
        "best_model": best_model_name,
        "optimal_threshold": optimal_threshold,
        "best_f1_at_optimal_threshold": round(best_f1, 4),
        "benchmark_comparison": benchmark_results,
        "feature_importances": feature_importance_dict,
        "feature_names": fe.feature_names_,
        "dataset_metadata": {
            "total_records": len(df),
            "train_records": len(X_train_raw),
            "test_records": len(X_test_raw),
            "default_rate": round(float(np.mean(y)), 4)
        }
    }

    with open(metrics_file, "w") as f:
        json.dump(report_payload, f, indent=4)

    print(f"[+] Successfully serialized model to: {model_file}")
    print(f"[+] Preprocessor saved to: {preprocessor_file}")
    print(f"[+] Evaluation report saved to: {metrics_file}")
    print("=================================================================\n")

    return report_payload


if __name__ == "__main__":
    train_and_benchmark()
