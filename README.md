# 🚀 Loan Default Prediction using Machine Learning (MLOps Pipeline)

# 📌 Problem Statement

Loan defaults pose a major risk to financial institutions, impacting profitability and stability.
This project builds an end-to-end machine learning system to predict whether a borrower will default on a loan using structured financial data.

# 🎯 Objectives

Build a robust loan default prediction model
Handle missing values, outliers, and class imbalance
Perform feature engineering for better predictions
Track experiments using MLflow
Implement model explainability using SHAP
Monitor data drift using PSI, CSI, and KS tests
Deploy model using FastAPI + Docker

# ⚙️ Tech Stack

Language: Python
Libraries: Pandas, NumPy, Scikit-learn, Imbalanced-learn
Model: Gradient Boosting, Logistic Regression
Tracking: MLflow
Explainability: SHAP
API: FastAPI + Uvicorn
Deployment: Docker

# 🔄 End-to-End Pipeline

Data → Preprocessing → Feature Engineering → SMOTE → Model Training  
→ MLflow Tracking → SHAP Explainability → Drift Detection → API → Docker

# 📊 Exploratory Data Analysis (EDA)

Checked missing values and feature distributions
Detected outliers using IQR method
Analyzed relationship between features and target
Observed class imbalance in default vs non-default

📌 Notebook: EDA.ipynb

# 🧹 Data Preprocessing

Missing values handled using median (numeric) and mode (categorical)
Outliers capped using IQR method
Categorical variables encoded using Label Encoding
ID-like columns removed

# ⚡ Feature Engineering

Loan to Income Ratio
Monthly Payment Estimate
Credit Score Flag (good_credit ≥ 700)
Income per Dependent

These features significantly improved model performance.

# ⚖️ Handling Imbalanced Data

Applied SMOTE (Synthetic Minority Oversampling Technique)
Balanced class distribution before training
Improved recall for default class

# 🤖 Models Used

| Model                    | Purpose               |
| ------------------------ | --------------------- |
| Logistic Regression      | Baseline              |
| Gradient Boosting        | Final optimized model |
| Random Forest (optional) | Comparison            |


# 📌 Key Insights:

Gradient Boosting achieved strong ROC-AUC performance
SMOTE improved recall significantly
Logistic Regression helped in baseline comparison
🔬 MLflow Experiment Tracking
Logged parameters, metrics, and trained models
Compared multiple experiments (default vs tuned models)
Stored SHAP plots and drift reports

# 📌 Experiments include:

Default Gradient Boosting
Tuned Gradient Boosting
🧠 Explainability (SHAP)
Used SHAP to interpret feature importance
Generated:
SHAP Summary Plot
SHAP Feature Importance (Bar Plot)
Saved results as shap_importance.csv

# 📌 Helps answer: “Why did the model predict default?”

📉 Data Drift Monitoring
PSI (Population Stability Index) → prediction drift
CSI (Characteristic Stability Index) → feature drift
KS Test → statistical distribution difference

# 📌 Output:

csi_results.csv
ks_results.csv
Drift plots logged in MLflow


# 🌐 API Deployment (FastAPI)


Endpoints:

GET /health → check model status
POST /predict → single prediction
POST /predict_batch → batch predictions
GET /feature_importance → SHAP insights

Example Request:
```
{
  "income_annum": 900000,
  "loan_amount": 2000000,
  "loan_term": 18,
  "cibil_score": 720
}
```

Example Response:
```
{
  "default_probability": 0.14,
  "prediction": "No Default",
  "risk_level": "LOW",
  "confidence": 0.85
}
```

# 📌 Includes:

Risk classification (LOW / MEDIUM / HIGH)
Confidence score

👉 Defined in your API logic

# 🐳 Docker Deployment
```
docker build -t loan-default-api .
docker run -p 8000:8000 loan-default-api
```

# ▶️ How to Run Project

pip install -r requirements.txt
python model.py        # Train model
python explain.py      # Generate SHAP
python drift.py        # Run drift detection
uvicorn app:app --reload
mlflow ui

# 📁 Project Structure
```
project/
│
├── app.py
├── model.py
├── drift.py
├── explain.py
├── requirements.txt
├── Dockerfile
├── EDA.ipynb
├── data/
├── mlruns/
└── model.pkl
```

# 🏆 Conclusion

This project demonstrates a complete production-ready ML pipeline, including:

✔ Data preprocessing
✔ Feature engineering
✔ Handling class imbalance (SMOTE)
✔ Model training & evaluation
✔ MLflow experiment tracking
✔ SHAP explainability
✔ Data drift monitoring
✔ API deployment with FastAPI
✔ Docker containerization

👨‍💻 Submitted by

Sakshi Deshmukh
