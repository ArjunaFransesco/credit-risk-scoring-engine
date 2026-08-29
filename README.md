# 💳 Credit Risk Scoring Engine & Underwriting Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-111?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end Financial Machine Learning & Underwriting Intelligence engine designed to evaluate borrower creditworthiness, estimate the **Probability of Default (PD)**, calculate **FICO-equivalent Credit Scores (300–850)**, and deliver real-time risk explanations.

---

## 📌 Executive Summary & Financial Value

In banking and consumer credit, expected portfolio loss is governed by the Basel II formula:
$$\text{Expected Loss (EL)} = \text{PD} \times \text{LGD} \times \text{EAD}$$

- **PD (Probability of Default)**: Estimated via machine learning classifiers and scorecard scaling.
- **LGD (Loss Given Default)**: Unrecovered loan balance percentage.
- **EAD (Exposure at Default)**: Outstanding debt balance at default.

By optimizing the classification threshold and separating high-risk borrowers with high **Kolmogorov-Smirnov (KS > 0.52)** and **ROC-AUC (> 0.81)** separation, institutions can maximize interest margins while drastically curbing non-performing loan (NPL) rates.

---

## 🏗️ Architecture & Pipeline Flow

```
┌─────────────────────────┐
│ Raw Credit Applications │
└────────────┬────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ Feature Pipeline (WoE/IV, DTI, Tenures, One-Hot, Scale)│
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ Multi-Model Benchmarking (Scorecard, RF, GBDT, XGBoost)│
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ Calibration & FICO Scaling: Score = 550 - 75*ln(Odds)   │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ Production REST API & Interactive Fintech Dashboard     │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Model Benchmark & Performance Metrics

Evaluated on stratified out-of-time test partitions:

| Model Architecture | ROC-AUC | PR-AUC | KS-Statistic | F1-Score | Recall | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Scorecard)** 🌟 | **0.8105** | **0.4865** | **0.5222** | 0.4097 | **0.6782** | 0.1582 |
| **Random Forest Classifier** | 0.8098 | 0.4192 | 0.5500 | **0.4978** | 0.6552 | 0.0815 |
| **Gradient Boosting Classifier** | 0.7970 | 0.4390 | 0.5003 | 0.3448 | 0.2299 | 0.0768 |
| **XGBoost Classifier** | 0.7887 | 0.4217 | 0.5006 | 0.4344 | 0.5517 | 0.0912 |

> **Key Takeaway**: Logistic Regression Scorecard and Random Forest deliver superior discriminative power, achieving over **0.81 ROC-AUC** and capturing **> 67% of default events**.

---

## 📁 Repository Structure

```
credit-risk-scoring-engine/
├── app/
│   ├── static/
│   │   ├── css/style.css       # Fintech dark mode glassmorphic UI
│   │   └── js/app.js           # Live risk calculator & API client
│   ├── templates/
│   │   └── index.html          # Interactive underwriting dashboard
│   └── main.py                 # Flask server & REST endpoints
├── data/
│   └── raw/
│       └── credit_risk_dataset.csv  # Benchmark credit dataset
├── models/
│   ├── best_credit_risk_model.joblib # Serialized model
│   └── feature_engineer.joblib       # Preprocessing transformer
├── notebooks/
│   └── credit_risk_modeling_pipeline.ipynb # Full EDA & Model Exploration
├── reports/
│   └── model_evaluation_metrics.json # Automated evaluation report
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion & synthetic benchmark
│   ├── features.py             # Feature engineering & WoE/IV calculator
│   ├── train.py                # Multi-model training & benchmarking
│   └── predict.py              # Inference engine & score converter
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions
└── README.md                   # Documentation
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/ArjunaFransesco/credit-risk-scoring-engine.git
cd credit-risk-scoring-engine
pip install -r requirements.txt
```

### 2. Train and Benchmark Models
```bash
python src/train.py
```

### 3. Launch Interactive Underwriting Web Dashboard
```bash
python app/main.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🔌 REST API Specification

### Endpoint: `POST /api/predict`
Calculates real-time risk scores for an applicant payload:

```json
{
  "person_age": 34,
  "person_income": 95000,
  "loan_amnt": 12000,
  "loan_int_rate": 7.8,
  "person_emp_length": 7,
  "cb_person_cred_hist_length": 10,
  "person_home_ownership": "MORTGAGE",
  "loan_intent": "HOMEIMPROVEMENT",
  "loan_grade": "A",
  "cb_person_default_on_file": "N"
}
```

#### Response:
```json
{
  "status": "success",
  "data": {
    "probability_of_default": 0.1174,
    "credit_score": 701,
    "risk_category": "Low Risk",
    "risk_grade": "B",
    "decision": "APPROVE",
    "color_code": "#06b6d4",
    "risk_drivers": [
      "Healthy Debt-to-Income ratio (12.6%) within safe debt bounds",
      "Clean credit history: No prior historical default records on file",
      "Strong employment stability tenure (7 yrs)"
    ],
    "debt_to_income_ratio": 0.1263
  }
}
```

---

## 👤 Author & Portfolio
- **Author:** [Arjuna Fransesco](https://github.com/ArjunaFransesco)
- **GitHub Repositories:** [https://github.com/ArjunaFransesco?tab=repositories](https://github.com/ArjunaFransesco?tab=repositories)
- **Portfolio Website:** [https://github.com/ArjunaFransesco/arjuna-portfolio](https://github.com/ArjunaFransesco/arjuna-portfolio)


<!-- Last Maintenance Audit: 2026-08-29 -->
