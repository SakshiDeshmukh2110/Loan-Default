"""
explain.py - Explainability using SHAP
Generates SHAP summary plots and logs them to MLflow
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import mlflow
import joblib
import warnings
warnings.filterwarnings("ignore")


def load_artifacts(model_path: str = "model.pkl"):
    """Load the trained pipeline."""
    pipeline = joblib.load(model_path)
    return pipeline


def get_shap_explainer(pipeline, X_sample: pd.DataFrame):
    """
    Extract the underlying model from the pipeline and build a SHAP explainer.
    Works with tree-based models (TreeExplainer) and falls back to KernelExplainer.
    """
    # Extract the model step from the pipeline
    model = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]

    # Transform the sample through the scaler (skip SMOTE — inference only)
    X_scaled = scaler.transform(X_sample)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_sample.columns)

    model_type = type(model).__name__
    print(f"[explain] Model type: {model_type}")

    if hasattr(model, "estimators_") or hasattr(model, "feature_importances_"):
        # Tree-based: GradientBoosting, RandomForest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        # For binary classifiers returning a list, take class-1 values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        # Fallback: KernelExplainer (slower but universal)
        background = shap.kmeans(X_scaled, 10)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_scaled[:50])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    return explainer, shap_values, X_scaled_df


def plot_shap_summary(shap_values, X_scaled_df: pd.DataFrame,
                      save_path: str = "shap_summary.png"):
    """Generate and save SHAP summary (beeswarm) plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_scaled_df,
        plot_type="dot",
        max_display=20,
        show=False,
    )
    plt.title("SHAP Feature Importance (Loan Default Prediction)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] SHAP summary plot saved to {save_path}")
    return save_path


def plot_shap_bar(shap_values, X_scaled_df: pd.DataFrame,
                  save_path: str = "shap_bar.png"):
    """Generate and save SHAP mean absolute value bar chart."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_scaled_df,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("SHAP Mean |Value| – Feature Impact", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] SHAP bar chart saved to {save_path}")
    return save_path


def run_explanation(
    model_path: str = "model.pkl",
    data_path: str = "data/loan_data.csv",
    experiment_name: str = "loan_default_experiment",
    n_samples: int = 200,
):
    """
    Full XAI pipeline:
    1. Load model + data
    2. Preprocess a sample
    3. Compute SHAP values
    4. Plot & log to MLflow
    """
    from model import load_data, preprocess  # local import to avoid circular dep

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="shap_explanation"):
        # ── Load & preprocess data
        df = load_data(data_path)
        X_train, X_test, y_train, y_test, feature_names = preprocess(df)

        # Use a representative sample for SHAP (large samples are slow)
        sample_size = min(n_samples, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=42)

        # ── Load pipeline
        pipeline = load_artifacts(model_path)

        # ── SHAP
        explainer, shap_values, X_scaled_df = get_shap_explainer(pipeline, X_sample)

        # ── Plots
        summary_path = plot_shap_summary(shap_values, X_scaled_df)
        bar_path = plot_shap_bar(shap_values, X_scaled_df)

        # ── Log to MLflow
        mlflow.log_artifact(summary_path)
        mlflow.log_artifact(bar_path)

        # ── Feature importance table
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": X_sample.columns,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)

        importance_df.to_csv("shap_importance.csv", index=False)
        mlflow.log_artifact("shap_importance.csv")

        print("\n[explain] Top 10 Features by SHAP Importance:")
        print(importance_df.head(10).to_string(index=False))

        # ── Log top feature importance as MLflow metrics
        for i, row in importance_df.head(10).iterrows():
            safe_name = row["feature"].replace(" ", "_").replace("-", "_")
            mlflow.log_metric(f"shap_{safe_name}", round(row["mean_abs_shap"], 6))

        mlflow.log_param("n_samples_explained", sample_size)
        print("[explain] SHAP artifacts logged to MLflow.")


if __name__ == "__main__":
    run_explanation()
