"""
Data Loader & Dataset Generator for Credit Risk Assessment.
Handles loading real or statistically synthetic credit default records.
"""

import os
import numpy as np
import pandas as pd


def generate_credit_dataset(n_samples: int = 4000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic credit scoring benchmark dataset with realistic
    correlations, credit risk factors, and default probabilities.
    """
    np.random.seed(random_state)

    # 1. Demographics & Employment
    age = np.random.gamma(shape=9.0, scale=3.5, size=n_samples).astype(int)
    age = np.clip(age, 20, 72)

    # Employment length is positively correlated with age
    emp_length = np.clip(
        (age - 18) * np.random.uniform(0.1, 0.7, size=n_samples) + np.random.normal(0, 1.5, size=n_samples),
        0, 40
    ).astype(float)
    # Inject ~3% missing values in employment length
    emp_mask = np.random.rand(n_samples) < 0.03
    emp_length[emp_mask] = np.nan

    # Income log-normal distribution
    income = np.random.lognormal(mean=10.9, sigma=0.55, size=n_samples).astype(int)
    income = np.clip(income, 12000, 350000)

    # Home ownership
    home_choices = ["RENT", "MORTGAGE", "OWN", "OTHER"]
    home_probs = [0.50, 0.41, 0.08, 0.01]
    home_ownership = np.random.choice(home_choices, size=n_samples, p=home_probs)

    # Loan intent
    intent_choices = [
        "EDUCATION", "MEDICAL", "VENTURE",
        "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"
    ]
    intent_probs = [0.20, 0.18, 0.18, 0.17, 0.16, 0.11]
    loan_intent = np.random.choice(intent_choices, size=n_samples, p=intent_probs)

    # Loan amount
    loan_amnt = np.random.gamma(shape=3.5, scale=3000, size=n_samples).astype(int)
    loan_amnt = np.clip(loan_amnt, 1000, 35000)

    # Loan percent income
    loan_percent_income = np.round(loan_amnt / income, 2)

    # Loan grade (A to G) based on creditworthiness
    # Grade determines interest rate
    grade_probs = [0.33, 0.32, 0.20, 0.10, 0.03, 0.015, 0.005]
    grades = ["A", "B", "C", "D", "E", "F", "G"]
    loan_grade = np.random.choice(grades, size=n_samples, p=grade_probs)

    # Interest rate based on grade + noise
    base_rates = {"A": 7.5, "B": 10.5, "C": 13.5, "D": 16.0, "E": 18.5, "F": 20.5, "G": 22.5}
    loan_int_rate = np.array([
        base_rates[g] + np.random.normal(0, 0.8) for g in loan_grade
    ])
    loan_int_rate = np.round(np.clip(loan_int_rate, 5.0, 24.0), 2)
    # Inject ~5% missing values in interest rate
    int_mask = np.random.rand(n_samples) < 0.05
    loan_int_rate[int_mask] = np.nan

    # Historical Default on File
    # Higher for lower grades
    def_prob_by_grade = {"A": 0.03, "B": 0.08, "C": 0.18, "D": 0.30, "E": 0.45, "F": 0.60, "G": 0.70}
    default_on_file = np.array([
        "Y" if np.random.rand() < def_prob_by_grade[g] else "N"
        for g in loan_grade
    ])

    # Credit History Length
    cred_hist_length = np.clip(
        (age - 18) * np.random.uniform(0.3, 0.8, size=n_samples) + np.random.normal(0, 1.0, size=n_samples),
        2, 35
    ).astype(int)

    # Target variable: Probability of default (Log-odds logistic calculation)
    # Higher debt-to-income, lower income, higher interest rate, previous default -> higher risk
    int_rate_diff = loan_int_rate - 11.0
    int_rate_clean = np.nan_to_num(int_rate_diff, nan=0.0)

    z = (
        -3.8
        + 4.8 * loan_percent_income
        + 0.12 * int_rate_clean
        + 1.35 * (default_on_file == "Y").astype(int)
        + 0.45 * (home_ownership == "RENT").astype(int)
        - 0.35 * (home_ownership == "OWN").astype(int)
        - 0.03 * np.nan_to_num(emp_length, nan=5.0)
        - 0.000004 * income
        + 0.60 * (loan_intent == "DEBTCONSOLIDATION").astype(int)
        + 0.35 * (loan_intent == "MEDICAL").astype(int)
    )
    p_default = 1 / (1 + np.exp(-z))
    loan_status = (np.random.rand(n_samples) < p_default).astype(int)

    df = pd.DataFrame({
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home_ownership,
        "person_emp_length": emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": default_on_file,
        "cb_person_cred_hist_length": cred_hist_length,
        "loan_status": loan_status
    })

    return df


def load_data(data_path: str = "data/raw/credit_risk_dataset.csv", n_samples: int = 4000) -> pd.DataFrame:
    """
    Loads dataset from local CSV or generates fresh benchmark dataset if not found.
    """
    if os.path.exists(data_path):
        print(f"[+] Loading credit risk dataset from {data_path}")
        return pd.read_csv(data_path)
    
    print(f"[!] Dataset not found at {data_path}. Generating {n_samples} benchmark records...")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df = generate_credit_dataset(n_samples=n_samples)
    df.to_csv(data_path, index=False)
    print(f"[+] Benchmark dataset successfully generated and saved to {data_path}")
    return df


if __name__ == "__main__":
    df = load_data()
    print("Dataset Shape:", df.shape)
    print("Class Balance:\n", df["loan_status"].value_counts(normalize=True))
    print("\nSample Rows:")
    print(df.head())
