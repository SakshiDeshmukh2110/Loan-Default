# 🏦 Loan Default Prediction

> Predicting the likelihood of a borrower defaulting on a loan using machine learning, with full MLflow tracking, SHAP explainability, and data drift detection.

---

## 📋 Problem Statement

Defaulted loans significantly disrupt the financial health of institutions, leading to substantial losses and reputational damage. Traditional risk assessment methods (credit score, income, collateral) fail to capture complex patterns. This project builds a robust ML pipeline to predict loan defaults.

---

## 🏗️ Project Structure

```
loan-default-prediction/
│
├── app.py              # FastAPI REST API (inference endpoint)
├── model.py            # Training pipeline (preprocessing + SMOTE + model + MLflow)
├── drift.py            # Data drift detection (PSI, CSI, KS) logged to MLflow
├── explain.py          # SHAP explainability logged to MLflow
├── EDA.ipynb           # Exploratory Data Analysis notebook
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker container definition
├── model.pkl           # Trained model artifact (generated after training)
├── mlruns/             # MLflow experiment tracking
└── data/
    └── loan_data.csv   # Input dataset
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/loan-default-prediction.git
cd loan-default-prediction
pip install -r requirements.txt
```

### 2. Add Dataset

Place your `loan_data.csv` inside the `data/` folder.

### 3. Train the Model (runs 2 MLflow experiments)

```bash
python model.py
```

### 4. Generate SHAP Explanations

```bash
python explain.py
```

### 5. Run Drift Detection

```bash
python drift.py
```

### 6. Start the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

### 7. View MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open: `http://localhost:5000`

---

## 🐳 Docker

### Build & Run

```bash
# Train model first
python model.py

# Build image
docker build -t loan-default-api .

# Run API
docker run -p 8000:8000 loan-default-api

# Run MLflow UI (separate container)
docker run -p 5000:5000 loan-default-api mlflow ui --host 0.0.0.0 --port 5000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Single loan prediction |
| `POST` | `/predict_batch` | Batch predictions |
| `GET` | `/feature_importance` | Top SHAP features |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 900000,
    "loan_amount": 2000000,
    "loan_term": 18,
    "cibil_score": 720,
    "residential_assets_value": 3000000,
    "commercial_assets_value": 500000,
    "luxury_assets_value": 200000,
    "bank_asset_value": 150000
  }'
```

### Example Response

```json
{
  "default_probability": 0.1423,
  "prediction": "No Default",
  "risk_level": "LOW",
  "confidence": 0.8577
}
```

---

## 🔬 ML Pipeline

```
Raw Data
   │
   ▼
Missing Value Imputation (Median / Mode)
   │
   ▼
Outlier Capping (IQR × 1.5)
   │
   ▼
Feature Engineering
  ├── loan_to_income
  ├── monthly_payment_est
  ├── good_credit (CIBIL ≥ 700)
  ├── income_per_dependent
  └── total_assets
   │
   ▼
Label Encoding (categorical features)
   │
   ▼
SMOTE (handle class imbalance)
   │
   ▼
StandardScaler
   │
   ▼
Gradient Boosting Classifier
   │
   ▼
MLflow Logging (metrics + artifacts + model)
```

---

## 📊 MLflow Experiments

Two experiments are run with different hyperparameters:

| Parameter | Experiment 1 | Experiment 2 |
|-----------|-------------|-------------|
| `n_estimators` | 100 | 200 |
| `learning_rate` | 0.1 | 0.05 |
| `max_depth` | 3 | 5 |
| `subsample` | 1.0 | 0.8 |

### Tracked Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Cross-validation AUC (mean ± std)
- PSI (Population Stability Index)
- CSI per feature (Characteristic Stability Index)
- KS statistic per feature
- SHAP feature importances

---

## 🧠 Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to explain model predictions:

- **Summary Plot**: Shows feature impact distribution across all samples
- **Bar Chart**: Mean absolute SHAP value per feature
- All artifacts logged to MLflow

---

## 📉 Data Drift Detection

| Method | What it measures | Threshold |
|--------|-----------------|-----------|
| **PSI** | Overall prediction score shift | < 0.1 OK, 0.1–0.2 Warning, > 0.2 Drift |
| **CSI** | Per-feature distribution shift | Same as PSI |
| **KS Test** | Statistical distribution difference | p-value < 0.05 → Drift |

---

## 🛠️ Tech Stack

- **ML**: scikit-learn, imbalanced-learn, XGBoost
- **Tracking**: MLflow
- **Explainability**: SHAP
- **API**: FastAPI + Uvicorn
- **Deployment**: Docker
- **Data**: pandas, numpy, scipy

---

## 📸 Screenshots

### MLflow Experiment Tracking
*(Add screenshots of your MLflow UI here)*

### SHAP Summary Plot
*(Add shap_summary.png here)*

### API Swagger UI
*(Add screenshot of /docs here)*
