"""
Feature Engineering and Preprocessing Pipeline for Credit Risk Scoring Engine.
Includes WoE / IV analytics, missing value imputation, domain-specific ratios, and sklearn transformers.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer for credit scoring domain features.
    """
    def __init__(self):
        self.median_emp_length_ = None
        self.median_int_rate_ = None
        self.feature_names_ = []
        self.scaler_ = StandardScaler()
        self.grade_map_ = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

    def fit(self, X, y=None):
        df = X.copy()
        self.median_emp_length_ = df["person_emp_length"].median()
        self.median_int_rate_ = df["loan_int_rate"].median()
        
        # Transform to capture feature names and fit scaler
        transformed_df = self._transform_df(df)
        self.feature_names_ = list(transformed_df.columns)
        self.scaler_.fit(transformed_df)
        return self

    def _transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Impute missing values
        df["person_emp_length"] = df["person_emp_length"].fillna(self.median_emp_length_ if self.median_emp_length_ is not None else 5.0)
        df["loan_int_rate"] = df["loan_int_rate"].fillna(self.median_int_rate_ if self.median_int_rate_ is not None else 11.0)

        # 2. Domain-Specific Ratios
        df["debt_to_income"] = df["loan_amnt"] / np.maximum(df["person_income"], 1000)
        df["loan_to_age"] = df["loan_amnt"] / np.maximum(df["person_age"], 18)
        df["cred_hist_to_age"] = df["cb_person_cred_hist_length"] / np.maximum(df["person_age"], 18)
        df["emp_to_age"] = df["person_emp_length"] / np.maximum(df["person_age"], 18)

        # 3. Ordinal Encoding for Loan Grade
        df["loan_grade_num"] = df["loan_grade"].map(self.grade_map_).fillna(3)

        # 4. Binary Indicator
        df["cb_person_default_num"] = (df["cb_person_default_on_file"] == "Y").astype(int)

        # 5. One-Hot Encoding for categorical features
        # Fixed known categories to maintain consistency in production
        home_cols = ["RENT", "MORTGAGE", "OWN", "OTHER"]
        for cat in home_cols:
            df[f"home_{cat.lower()}"] = (df["person_home_ownership"] == cat).astype(int)

        intent_cols = ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
        for cat in intent_cols:
            df[f"intent_{cat.lower()}"] = (df["loan_intent"] == cat).astype(int)

        # Drop original raw categorical columns
        cols_to_drop = [
            "person_home_ownership", "loan_intent", "loan_grade",
            "cb_person_default_on_file"
        ]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        return df

    def transform(self, X):
        df = self._transform_df(X)
        # Ensure all columns match training order
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names_]
        return self.scaler_.transform(df)

    def transform_to_df(self, X):
        df = self._transform_df(X)
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_names_]


def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str = "loan_status", bins: int = 5):
    """
    Computes Weight of Evidence (WoE) and Information Value (IV) for a feature.
    Useful in credit risk modeling scorecard development.
    """
    data = df[[feature, target]].copy()
    if pd.api.types.is_numeric_dtype(data[feature]) and data[feature].nunique() > bins:
        data["bin"] = pd.qcut(data[feature], q=bins, duplicates="drop")
    else:
        data["bin"] = data[feature].astype(str)

    grouped = data.groupby("bin", observed=False).agg(
        total=(target, "count"),
        bad=(target, "sum")
    )
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    grouped["dist_good"] = grouped["good"] / (total_good if total_good > 0 else 1)
    grouped["dist_bad"] = grouped["bad"] / (total_bad if total_bad > 0 else 1)

    # Avoid div by zero
    grouped["dist_good"] = grouped["dist_good"].replace(0, 0.0001)
    grouped["dist_bad"] = grouped["dist_bad"].replace(0, 0.0001)

    grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
    grouped["iv_component"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
    total_iv = grouped["iv_component"].sum()

    return grouped, total_iv


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    
    # Test WoE / IV
    grp, iv = calculate_woe_iv(df, "loan_percent_income", "loan_status")
    print(f"[+] Information Value (IV) for loan_percent_income: {iv:.4f}")
    
    # Test Preprocessor
    X = df.drop(columns=["loan_status"])
    fe = CreditFeatureEngineer()
    X_trans = fe.fit_transform(X)
    print("[+] Transformed Feature Matrix Shape:", X_trans.shape)
    print("[+] Feature Names:", fe.feature_names_)
