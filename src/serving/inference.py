import os
import pandas as pd
import mlflow
import sys
import json

# FIX 1: Correctly find project root (Go up 2 levels, not 3)
# src/serving/inference.py -> src/serving -> src -> PROJECT_ROOT
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# === 1. CONFIGURATION & GLOBALS ===
MODEL_DIR = "/app/model"  # Default for Docker
model = None

# === 2. DEFINE FUNCTIONS FIRST ===

def load_model():
    """
    Loads the model into the global variable if not already loaded.
    Robustly searches for 'MLmodel' file if standard paths fail.
    """
    global model, MODEL_DIR
    if model is None:
        try:
            # Try loading from Docker path first
            model = mlflow.pyfunc.load_model(MODEL_DIR)
            print(f"✅ Model loaded successfully from {MODEL_DIR}")
        except Exception:
            # Fallback: Recursive search in local mlruns
            print(f"⚠️  Docker model not found at {MODEL_DIR}. Searching local mlruns...")
            try:
                # FIX 2: Use the correctly calculated project_root
                mlruns_dir = os.path.join(project_root, "mlruns")
                
                print(f"🔍 Searching for model in: {mlruns_dir}") # Debug print
                
                latest_model_path = None
                latest_timestamp = 0
                
                # Walk through all folders in mlruns to find 'MLmodel'
                found_models = 0
                for root, dirs, files in os.walk(mlruns_dir):
                    if "MLmodel" in files:
                        found_models += 1
                        # We found a valid model directory!
                        try:
                            mod_time = os.path.getmtime(root)
                            if mod_time > latest_timestamp:
                                latest_timestamp = mod_time
                                latest_model_path = root
                        except OSError:
                            continue
                
                print(f"   Found {found_models} model candidate(s).")

                if latest_model_path:
                    model = mlflow.pyfunc.load_model(latest_model_path)
                    MODEL_DIR = latest_model_path
                    print(f"✅ Fallback: Loaded model from {latest_model_path}")
                else:
                    raise Exception(f"No 'MLmodel' file found anywhere in {mlruns_dir}")
                    
            except Exception as e:
                raise Exception(f"CRITICAL: Failed to load model. Error: {e}")
    return model

def get_feature_columns():
    """
    Locate and load feature_columns.json.
    Checks inside the model folder first, then falls back to the project root artifacts.
    """
    filename = "feature_columns.json"
    
    # Path A: Check inside the MLflow model folder
    feature_file = os.path.join(MODEL_DIR, filename)
    
    if not os.path.exists(feature_file):
        # Path B: Check project root 'artifacts' folder using correct project_root
        feature_file = os.path.join(project_root, "artifacts", filename)

    if not os.path.exists(feature_file):
        raise FileNotFoundError(
            f"Could not find {filename} in {MODEL_DIR} or in {os.path.join(project_root, 'artifacts')}"
        )

    print(f"📂 Loading features from: {feature_file}")
    with open(feature_file, 'r') as f:
        return json.load(f)

# === 3. EXECUTE INITIALIZATION (AFTER DEFINITIONS) ===
load_model()  # Load model first to set MODEL_DIR
FEATURE_COLS = get_feature_columns()
print(f"✅ Loaded {len(FEATURE_COLS)} feature columns")

# === 4. TRANSFORMATION LOGIC ===
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # 1. Numeric Coercion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    
    # 2. Binary Encoding
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = df[c].map(mapping)
            df[c] = df[c].fillna(0).astype(int)

    # 3. One-Hot Encoding
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # 4. Boolean to Int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # 5. Alignment
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:
    """Main prediction function called by API."""
    loaded_model = load_model()
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)
    
    try:
        preds = loaded_model.predict(df_enc)
        
        if hasattr(preds, "tolist"):
            result = preds.tolist()[0]
        else:
            result = preds[0]
            
        if result == 1:
            return "Likely to churn"
        else:
            return "Not likely to churn"
            
    except Exception as e:
        return f"Prediction Error: {str(e)}"
