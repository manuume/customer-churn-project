# Telco Customer Churn Prediction: Production MLOps Pipeline

> **Business Context**: Acquiring a new customer costs 5x more than retaining an existing one. This project demonstrates a complete, production-ready ML system that predicts customer churn, enabling proactive retention strategies.

---

## 📊 Model Performance

The production XGBoost model was selected after systematic experimentation with Logistic Regression, Random Forest, and XGBoost variants. All experiments were tracked via MLflow for reproducibility.

### Test Set Metrics

| Metric | Value | Business Interpretation |
|--------|-------|------------------------|
| **Accuracy** | 81.4% | Correctly classified 4 out of 5 customers |
| **Precision** | 68.2% | 68% of predicted churners actually churned |
| **Recall** | 74.5% | Identified 74.5% of all actual churners |
| **F1-Score** | 0.71 | Balanced performance metric |
| **AUC-ROC** | 0.85 | Strong discriminatory power |

### 💼 Model Selection Rationale

**Why we prioritized Recall over Precision:**
- **False Negative Cost**: Missing a churning customer means losing their lifetime value (~$1,000-$5,000)
- **False Positive Cost**: Offering unnecessary retention incentive (~$50-$100)
- **Business Strategy**: Better to proactively engage 100 at-risk customers (even if 32 weren't actually leaving) than to miss 25 customers who will churn

---

## 🏗️ Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Raw Data  │────▶│  ML Pipeline │────▶│ Trained Model│
│   (CSV)     │     │  Processing  │     │   (MLflow)   │
└─────────────┘     └──────────────┘     └─────────────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   End User  │◀────│  FastAPI +   │◀────│   Model     │
│  (Browser)  │     │   Gradio     │     │  Inference  │
└─────────────┘     └──────────────┘     └─────────────┘
```

---

## 📂 Project Structure

```
CHURN_PROJECT/
│
├── .github/workflows/
│   └── ci.yml                    # CI/CD: Automated Docker build & push on git push
│
├── app/
│   └── main.py                   # Application entrypoint (FastAPI + Gradio launcher)
│
├── artifacts/
│   ├── feature_columns.json      # Schema validation: 18 expected features
│   └── preprocessing.pkl         # (Optional) Fitted encoders for categorical variables
│
├── data/
│   └── telco_churn.csv          # Source data (7,043 customers × 21 features)
│
├── mlruns/                       # MLflow experiment tracking database (local filesystem)
│   └── 0/                       # Experiment ID folders with metrics & artifacts
│
├── model_build/                  # MLflow packaged model (production-ready)
│   ├── MLmodel                  # Model metadata (flavor, signature, dependencies)
│   ├── model.pkl                # Serialized XGBoost/RandomForest object
│   ├── conda.yaml               # Conda environment specification
│   └── requirements.txt         # Pip dependencies for model runtime
│
├── notebooks/
│   └── EDA.ipynb                # Exploratory Data Analysis (hypothesis generation)
│
├── scripts/
│   ├── run_pipeline.py          # End-to-end training orchestrator
│   └── test_pipeline.py         # Unit tests for data loading & preprocessing
│
├── src/                          # Core ML application logic
│   ├── data/
│   │   ├── load_data.py         # Data ingestion from CSV/database
│   │   └── preprocess.py        # Feature cleaning & encoding pipeline
│   ├── features/
│   │   └── build_features.py    # Feature engineering (derived metrics)
│   ├── models/
│   │   ├── train.py             # Model training with cross-validation
│   │   ├── evaluate.py          # Performance evaluation & metric calculation
│   │   └── tune.py              # Hyperparameter optimization (Optuna/GridSearch)
│   └── serving/
│       └── inference.py         # Prediction API with input validation
│
├── .dockerignore                 # Excludes mlruns/, data/, __pycache__ from image
├── .gitignore                    # Excludes secrets, caches, virtual environments
├── Dockerfile                    # Multi-stage build for optimized production image
├── requirements.txt              # Python dependencies (pinned versions)
└── README.md                     # This file
```

### 🔍 Key Components Explained

**Data Pipeline:**
- **`notebooks/EDA.ipynb`**: Initial exploration, distribution analysis, correlation studies
- **`src/features/`**: Feature engineering logic (e.g., `TotalCharges = MonthlyCharges × Tenure`)
- **`src/data/preprocess.py`**: Data cleaning, missing value imputation, categorical encoding

**Model Development:**
- **`src/models/train.py`**: Model training with stratified k-fold cross-validation
- **`src/models/evaluate.py`**: Performance metrics calculation and visualization
- **`mlruns/`**: All experiment metadata (parameters, metrics, artifacts) stored locally

**Serving Layer:**
- **`app/main.py`**: ASGI application with health checks and graceful shutdown
- **`src/serving/inference.py`**: Input schema validation, model loading, and prediction logic
- **`Dockerfile`**: Multi-stage build separating dependencies from application code

---

## 📈 Dataset Details

**Source**: Telco Customer Churn Dataset  
**Size**: 7,043 customers × 21 features  
**Target Variable**: `Churn` (Binary: Yes/No)  
**Class Distribution**: ~73% Non-Churners, ~27% Churners (imbalanced)

### Feature Categories

| Category | Features | Examples |
|----------|----------|----------|
| **Demographics** | 4 | Gender, SeniorCitizen, Partner, Dependents |
| **Services** | 9 | PhoneService, InternetService, TechSupport, StreamingTV |
| **Account** | 7 | Tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges |

### Key Insights from EDA
1. **Contract Type**: Month-to-month contracts have 3x higher churn rate
2. **Tenure**: Customers with <6 months tenure churn at 50%+ rate
3. **Payment Method**: Electronic check users churn 45% vs. 15% for auto-pay users
4. **Total Charges**: Strong inverse correlation with churn probability

---

## 🚀 Getting Started

### Prerequisites
- **For Docker**: Docker Engine 20.10+
- **For Local Development**: Python 3.9+, pip

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/manuume/customer-churn-project.git
cd customer-churn-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (optional - pre-trained model included)
python scripts/run_pipeline.py

# Start the application
python app/main.py

# Access: http://localhost:8000
```

---

## 🔬 Reproducing the ML Pipeline

### Step 1: Data Preprocessing
```bash
python src/data/preprocess.py \
  --input data/telco_churn.csv \
  --output data/processed.csv
```
**What happens:**
- Missing values imputation (TotalCharges median fill)
- Categorical encoding (Label/One-Hot based on cardinality)
- Feature scaling (StandardScaler for numeric features)

### Step 2: Model Training
```bash
python src/models/train.py \
  --data data/processed.csv \
  --model xgboost \
  --cv-folds 5 \
  --log-mlflow
```
**What happens:**
- Stratified 5-fold cross-validation
- Hyperparameter tuning with Optuna (100 trials)
- Metrics logged to MLflow (`mlruns/` directory)
- Best model serialized to `model_build/`

### Step 3: Model Evaluation
```bash
python src/models/evaluate.py \
  --model model_build/model.pkl \
  --test-data data/test.csv
```
**Outputs:**
- Classification report (precision, recall, F1)
- Confusion matrix visualization
- ROC curve with AUC score
- Feature importance plot

---

## 🧪 API Usage Examples

### REST API (FastAPI)

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    ...
  }'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@data/new_customers.csv"
```

### Python Client

```python
import requests

payload = {
    "tenure": 12,
    "MonthlyCharges": 70.35,
    "Contract": "Month-to-month",
    # ... include all 18 features
}

response = requests.post(
    "http://localhost:8000/predict",
    json=payload
)

result = response.json()
print(f"Churn Probability: {result['churn_probability']:.2%}")
print(f"Prediction: {result['prediction']}")
```

---

## 🛠️ Technology Stack

### ML/Data Science
- **Training**: scikit-learn 1.3.0, XGBoost 1.7.6
- **Experiment Tracking**: MLflow 2.8.0
- **Data Processing**: pandas 2.0.3, numpy 1.24.3

### Backend/Serving
- **API Framework**: FastAPI 0.104.1 (async ASGI)
- **UI**: Gradio 4.7.1 (interactive web interface)
- **Model Format**: MLflow Models (pkl + metadata)

### DevOps/Infrastructure
- **Containerization**: Docker 20.10+ (multi-stage builds)
- **CI/CD**: GitHub Actions (automated testing & deployment)

---

## 🔄 CI/CD Pipeline

The `.github/workflows/ci.yml` automates:

1. **On Push to `main`:**
   - Runs unit tests (`pytest src/`)
   - Lints code (`flake8`, `black`)
   - Builds Docker image
   - Pushes to Docker Hub

2. **On Pull Request:**
   - Runs all tests
   - Generates coverage report
   - Blocks merge if tests fail

---

## 📊 MLflow Tracking

### View Experiments Locally
```bash
mlflow ui --port 5000
# Open: http://localhost:5000
```

### Compare Experiments
- Navigate to `Experiments` tab
- Select multiple runs
- Click `Compare` to see side-by-side metrics

### Key Tracked Metrics
- Train/Val/Test accuracy, precision, recall, F1
- Confusion matrix as artifact
- Model hyperparameters (learning_rate, max_depth, etc.)
- Training duration and environment

---

## 🧩 Extending the Project

### Adding a New Model

1. **Create model class** in `src/models/train.py`:
```python
def train_catboost(X_train, y_train, params):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    return model
```

2. **Register in pipeline**:
```python
# scripts/run_pipeline.py
models = {
    'xgboost': train_xgboost,
    'catboost': train_catboost,  # Add here
}
```

3. **Run experiment**:
```bash
python scripts/run_pipeline.py --model catboost
```

### Integrating a Database

Replace `src/data/load_data.py`:
```python
import psycopg2
import pandas as pd

def load_from_postgres(query):
    conn = psycopg2.connect(
        host="localhost",
        database="telecom",
        user="readonly",
        password="secret"
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df
```

---

## 🚧 Known Limitations

1. **Data Drift**: No automated monitoring for distribution shifts
2. **Model Versioning**: Manual rollback required if model degrades
3. **Scalability**: Single-instance deployment (no horizontal scaling)
4. **Feature Store**: Features are recomputed at inference time

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

**⭐ If this project helped you, please consider giving it a star!**
