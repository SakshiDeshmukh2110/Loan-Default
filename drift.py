"""
drift.py - Data Drift Detection
Computes PSI (Population Stability Index), CSI (Characteristic Stability Index),
and KS (Kolmogorov-Smirnov) tests, then logs results to MLflow.
"""

import numpy as np
import pandas as pd
from scipy import stats
import mlflow
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. PSI – Population Stability Index
# ─────────────────────────────────────────────
def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    PSI < 0.1  → No significant change
    PSI 0.1–0.2 → Slight shift
    PSI > 0.2  → Significant shift (investigate)
    """
    # Build quantile bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates at extremes

    def _bucket_counts(arr, bp):
        counts = np.histogram(arr, bins=bp)[0]
        counts = np.where(counts == 0, 0.0001, counts)  # avoid log(0)
        return counts / counts.sum()

    expected_pct = _bucket_counts(expected, breakpoints)
    actual_pct = _bucket_counts(actual, breakpoints)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


# ─────────────────────────────────────────────
# 2. CSI – Characteristic Stability Index
#    (PSI applied per feature)
# ─────────────────────────────────────────────
def compute_csi(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: list,
    buckets: int = 10,
) -> pd.DataFrame:
    """
    Returns a DataFrame with CSI per feature.
    """
    results = []
    for col in numeric_cols:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue
        csi = compute_psi(
            baseline_df[col].dropna().values,
            current_df[col].dropna().values,
            buckets=buckets,
        )
        flag = "OK" if csi < 0.1 else ("WARNING" if csi < 0.2 else "DRIFT")
        results.append({"feature": col, "CSI": round(csi, 6), "status": flag})

    return pd.DataFrame(results).sort_values("CSI", ascending=False)


# ─────────────────────────────────────────────
# 3. KS – Kolmogorov-Smirnov test
# ─────────────────────────────────────────────
def compute_ks(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: list,
) -> pd.DataFrame:
    """
    KS statistic + p-value per feature.
    p-value < 0.05 → distributions differ significantly
    """
    results = []
    for col in numeric_cols:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue
        ks_stat, p_value = stats.ks_2samp(
            baseline_df[col].dropna().values,
            current_df[col].dropna().values,
        )
        flag = "DRIFT" if p_value < 0.05 else "OK"
        results.append({
            "feature": col,
            "KS_statistic": round(ks_stat, 6),
            "p_value": round(p_value, 6),
            "status": flag,
        })

    return pd.DataFrame(results).sort_values("KS_statistic", ascending=False)


# ─────────────────────────────────────────────
# 4. Drift plot helpers
# ─────────────────────────────────────────────
def plot_csi_bar(csi_df: pd.DataFrame, save_path: str = "csi_drift.png"):
    colors = csi_df["status"].map({"OK": "green", "WARNING": "orange", "DRIFT": "red"})
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(csi_df["feature"], csi_df["CSI"], color=colors)
    ax.axhline(0.1, color="orange", linestyle="--", label="Warning (0.1)")
    ax.axhline(0.2, color="red", linestyle="--", label="Drift (0.2)")
    ax.set_title("CSI (Characteristic Stability Index) per Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("CSI Value")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_ks_bar(ks_df: pd.DataFrame, save_path: str = "ks_drift.png"):
    colors = ks_df["status"].map({"OK": "green", "DRIFT": "red"})
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(ks_df["feature"], ks_df["KS_statistic"], color=colors)
    ax.set_title("KS Statistic per Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("KS Statistic")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


# ─────────────────────────────────────────────
# 5. MAIN DRIFT RUN
# ─────────────────────────────────────────────
def run_drift_detection(
    data_path: str = "data/loan_data.csv",
    experiment_name: str = "loan_default_experiment",
    drift_fraction: float = 0.3,
    buckets: int = 10,
):
    """
    Simulates drift by splitting data into baseline (train) and current (test/recent).
    In production, 'current_df' would be your live scoring data.
    """
    from model import load_data, preprocess

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="drift_detection"):
        # ── Load data
        df = load_data(data_path)
        X_train, X_test, y_train, y_test, feature_names = preprocess(df)

        baseline_df = pd.DataFrame(X_train, columns=feature_names)
        current_df = pd.DataFrame(X_test, columns=feature_names)

        # ── Simulate additional drift by adding Gaussian noise to current
        np.random.seed(99)
        noise_cols = feature_names[:3]  # perturb first 3 features
        for col in noise_cols:
            std = current_df[col].std()
            current_df[col] += np.random.normal(0, 0.5 * std, size=len(current_df))

        numeric_cols = feature_names  # all features are numeric after preprocessing

        # ─── PSI on prediction score (if model is available)
        try:
            import joblib
            pipeline = joblib.load("model.pkl")
            baseline_scores = pipeline.predict_proba(baseline_df)[:, 1]
            current_scores = pipeline.predict_proba(current_df)[:, 1]
            psi_score = compute_psi(baseline_scores, current_scores)
            mlflow.log_metric("PSI_prediction_score", round(psi_score, 6))
            psi_flag = "OK" if psi_score < 0.1 else ("WARNING" if psi_score < 0.2 else "DRIFT")
            mlflow.log_param("PSI_flag", psi_flag)
            print(f"[drift] PSI (prediction score): {psi_score:.4f} → {psi_flag}")
        except Exception as e:
            print(f"[drift] Could not compute PSI on scores: {e}")

        # ─── CSI per feature
        csi_df = compute_csi(baseline_df, current_df, numeric_cols, buckets)
        csi_df.to_csv("csi_results.csv", index=False)
        mlflow.log_artifact("csi_results.csv")

        drift_features = csi_df[csi_df["status"] != "OK"]["feature"].tolist()
        mlflow.log_param("features_with_drift_CSI", str(drift_features))
        mlflow.log_metric("num_drifted_features_CSI", len(drift_features))

        # Log top-5 CSI values as metrics
        for _, row in csi_df.head(5).iterrows():
            safe = row["feature"].replace(" ", "_").replace("-", "_")
            mlflow.log_metric(f"CSI_{safe}", row["CSI"])

        csi_plot = plot_csi_bar(csi_df)
        mlflow.log_artifact(csi_plot)

        # ─── KS test per feature
        ks_df = compute_ks(baseline_df, current_df, numeric_cols)
        ks_df.to_csv("ks_results.csv", index=False)
        mlflow.log_artifact("ks_results.csv")

        ks_drift_features = ks_df[ks_df["status"] == "DRIFT"]["feature"].tolist()
        mlflow.log_param("features_with_drift_KS", str(ks_drift_features))
        mlflow.log_metric("num_drifted_features_KS", len(ks_drift_features))

        # Log top-5 KS statistics as metrics
        for _, row in ks_df.head(5).iterrows():
            safe = row["feature"].replace(" ", "_").replace("-", "_")
            mlflow.log_metric(f"KS_{safe}", row["KS_statistic"])

        ks_plot = plot_ks_bar(ks_df)
        mlflow.log_artifact(ks_plot)

        # ─── Summary
        print("\n[drift] CSI Results (top 10):")
        print(csi_df.head(10).to_string(index=False))
        print("\n[drift] KS Results (top 10):")
        print(ks_df.head(10).to_string(index=False))
        print("\n[drift] Drift detection complete. Artifacts logged to MLflow.")

        return csi_df, ks_df


if __name__ == "__main__":
    run_drift_detection()
