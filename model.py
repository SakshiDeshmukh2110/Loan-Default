"""
model.py - Loan Default Prediction Model Training & Evaluation
Handles: preprocessing, feature engineering, imbalance, training, MLflow logging
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def load_data(filepath: str = "data/loan_data.csv") -> pd.DataFrame:
    """Load raw loan dataset."""
    df = pd.read_csv(filepath)
    print(f"[load_data] Shape: {df.shape}")
    print(f"[load_data] Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> tuple:
    """
    Full preprocessing pipeline:
    - Handle missing values
    - Outlier capping (IQR)
    - Feature engineering
    - Encoding
    - Train/test split
    """
    df = df.copy()

    # ── Target variable
    target_col = "Default"   # adjust if your CSV uses a different name
    if target_col not in df.columns:
        # try common alternatives
        for alt in ["loan_status", "Loan_Status", "default", "TARGET"]:
            if alt in df.columns:
                df.rename(columns={alt: target_col}, inplace=True)
                break

    # ── Drop ID-like columns
    id_cols = [c for c in df.columns if c.lower() in ("id", "loan_id", "customerid")]
    df.drop(columns=id_cols, errors="ignore", inplace=True)

    # ── Encode binary target
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"Y": 1, "N": 0, "Yes": 1, "No": 0,
                                              "1": 1, "0": 0}).fillna(df[target_col])
    df[target_col] = df[target_col].astype(int)

    # ── Handle missing values (numeric: median, categorical: mode)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    for c in num_cols:
        df[c].fillna(df[c].median(), inplace=True)
    for c in cat_cols:
        df[c].fillna(df[c].mode()[0], inplace=True)

    # ── Outlier capping (IQR × 1.5)
    for c in num_cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        df[c] = df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    # ── Feature engineering
    if "income_annum" in df.columns and "loan_amount" in df.columns:
        df["loan_to_income"] = df["loan_amount"] / (df["income_annum"] + 1)
    if "loan_amount" in df.columns and "loan_term" in df.columns:
        df["monthly_payment_est"] = df["loan_amount"] / (df["loan_term"] + 1)
    if "cibil_score" in df.columns:
        df["good_credit"] = (df["cibil_score"] >= 700).astype(int)
    if "no_of_dependents" in df.columns and "income_annum" in df.columns:
        df["income_per_dependent"] = df["income_annum"] / (df["no_of_dependents"] + 1)

    # ── Encode categorical columns
    le = LabelEncoder()
    for c in cat_cols:
        df[c] = le.fit_transform(df[c].astype(str))

    # ── Split
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[preprocess] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[preprocess] Class distribution (train): {dict(y_train.value_counts())}")
    return X_train, X_test, y_train, y_test, list(X.columns)


# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────
def get_model(model_name: str = "gradient_boosting", **hyperparams):
    """Return a sklearn estimator based on name."""
    models = {
        "gradient_boosting": GradientBoostingClassifier,
        "random_forest": RandomForestClassifier,
        "logistic_regression": LogisticRegression,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models)}")
    return models[model_name](**hyperparams)


def train(
    filepath: str = "data/loan_data.csv",
    model_name: str = "gradient_boosting",
    experiment_name: str = "loan_default_experiment",
    **hyperparams
):
    """
    Full training pipeline:
    - Load → preprocess → SMOTE → scale → train → evaluate → MLflow log → save
    """
    # ── MLflow setup
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_name}_run"):
        # ── Data
        df = load_data(filepath)
        X_train, X_test, y_train, y_test, feature_names = preprocess(df)

        # ── Log hyperparams
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("smote", True)
        for k, v in hyperparams.items():
            mlflow.log_param(k, v)

        # ── Pipeline: SMOTE → Scale → Model
        estimator = get_model(model_name, **hyperparams)
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)

        # ── Evaluation
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        print("\n[train] Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\n[train] Classification Report:")
        print(classification_report(y_test, y_pred))

        # ── Save model
        joblib.dump(pipeline, "model.pkl")
        mlflow.sklearn.log_model(pipeline, "model")
        print("[train] Model saved to model.pkl")

        # ── Cross-validation AUC
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="roc_auc"
        )
        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_auc_std", cv_scores.std())
        print(f"[train] CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return pipeline, metrics, feature_names


# ─────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Experiment 1: Gradient Boosting (default params)
    print("=" * 60)
    print("EXPERIMENT 1: Gradient Boosting (default hyperparams)")
    print("=" * 60)
    train(
        filepath="data/loan_data.csv",
        model_name="gradient_boosting",
        experiment_name="loan_default_experiment",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )

    # Experiment 2: Gradient Boosting (tuned params)
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Gradient Boosting (tuned hyperparams)")
    print("=" * 60)
    train(
        filepath="data/loan_data.csv",
        model_name="gradient_boosting",
        experiment_name="loan_default_experiment",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_split=10,
        random_state=42,
    )
