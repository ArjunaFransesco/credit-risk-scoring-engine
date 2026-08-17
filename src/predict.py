"""
Inference Engine and Credit Scorecard Calculator.
Transforms input loan applications into Default Probabilities, FICO-like Credit Scores,
Risk Grades, and Explainable Risk Driver Breakdown.
"""

import os
import joblib
import numpy as np
import pandas as pd


class CreditRiskPredictor:
    def __init__(self, model_path="models/best_credit_risk_model.joblib", preprocessor_path="models/feature_engineer.joblib"):
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Model or preprocessor artifact not found. Please run train.py first.")
        
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def calculate_credit_score(self, default_prob: float) -> int:
        """
        Converts probability of default (PD) to FICO-like credit score (300 to 850).
        Formula: Score = Offset - Factor * ln(Odds)
        Odds = p / (1 - p)
        """
        p = np.clip(default_prob, 0.001, 0.999)
        odds = p / (1.0 - p)
        # Calibrated baseline: p=0.10 gives score ~710, p=0.50 gives score ~550, p=0.85 gives ~380
        score = 550 - 75 * np.log(odds)
        return int(np.clip(round(score), 300, 850))

    def evaluate_risk(self, score: int, pd_val: float):
        if score >= 720:
            category = "Minimal Risk"
            grade = "A"
            decision = "AUTO_APPROVE"
            color = "#10b981"  # Emerald green
        elif score >= 650:
            category = "Low Risk"
            grade = "B"
            decision = "APPROVE"
            color = "#06b6d4"  # Cyan
        elif score >= 580:
            category = "Moderate Risk"
            grade = "C"
            decision = "MANUAL_REVIEW"
            color = "#f59e0b"  # Amber
        elif score >= 500:
            category = "High Risk"
            grade = "D"
            decision = "MANUAL_REVIEW"
            color = "#f97316"  # Orange
        else:
            category = "Critical Risk"
            grade = "E"
            decision = "REJECT"
            color = "#ef4444"  # Red

        return category, grade, decision, color

    def identify_risk_drivers(self, applicant: dict) -> list:
        drivers = []
        income = float(applicant.get("person_income", 50000))
        loan_amnt = float(applicant.get("loan_amnt", 10000))
        dti = loan_amnt / max(income, 1000)
        
        if dti > 0.35:
            drivers.append(f"Elevated Debt-to-Income ratio ({dti:.1%}) exceeds recommended 35% ceiling")
        elif dti < 0.15:
            drivers.append(f"Healthy Debt-to-Income ratio ({dti:.1%}) within safe debt bounds")

        if applicant.get("cb_person_default_on_file", "N") == "Y":
            drivers.append("Adverse credit flag: Prior historical default record detected")
        else:
            drivers.append("Clean credit history: No prior historical default records on file")

        int_rate = float(applicant.get("loan_int_rate", 12.0))
        if int_rate >= 15.0:
            drivers.append(f"High risk premium: Interest rate ({int_rate:.1f}%) indicates subprime tier")

        emp_len = float(applicant.get("person_emp_length", 3.0))
        if emp_len < 2.0:
            drivers.append(f"Short employment stability tenure ({emp_len:.0f} yrs)")
        elif emp_len >= 5.0:
            drivers.append(f"Strong employment stability tenure ({emp_len:.0f} yrs)")

        return drivers

    def predict_single(self, applicant: dict) -> dict:
        """
        Runs full inference on a single applicant payload.
        """
        # Ensure default fields exist
        dti = float(applicant.get("loan_amnt", 10000)) / max(float(applicant.get("person_income", 50000)), 1000)
        applicant["loan_percent_income"] = round(dti, 2)
        
        df_input = pd.DataFrame([applicant])
        X_trans = self.preprocessor.transform(df_input)

        prob_default = float(self.model.predict_proba(X_trans)[0, 1])
        score = self.calculate_credit_score(prob_default)
        risk_category, grade, decision, color = self.evaluate_risk(score, prob_default)
        risk_drivers = self.identify_risk_drivers(applicant)

        return {
            "probability_of_default": round(prob_default, 4),
            "credit_score": score,
            "risk_category": risk_category,
            "risk_grade": grade,
            "decision": decision,
            "color_code": color,
            "risk_drivers": risk_drivers,
            "debt_to_income_ratio": round(dti, 4)
        }


if __name__ == "__main__":
    predictor = CreditRiskPredictor()
    
    # Test applicant 1 (Prime borrower)
    prime_borrower = {
        "person_age": 34,
        "person_income": 95000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 7,
        "loan_intent": "HOMEIMPROVEMENT",
        "loan_grade": "A",
        "loan_amnt": 12000,
        "loan_int_rate": 7.8,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 10
    }
    
    # Test applicant 2 (Subprime borrower)
    subprime_borrower = {
        "person_age": 23,
        "person_income": 22000,
        "person_home_ownership": "RENT",
        "person_emp_length": 1,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_grade": "E",
        "loan_amnt": 15000,
        "loan_int_rate": 18.5,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 2
    }

    print("\n--- Prime Borrower Prediction ---")
    print(predictor.predict_single(prime_borrower))

    print("\n--- Subprime Borrower Prediction ---")
    print(predictor.predict_single(subprime_borrower))
