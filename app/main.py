"""
FASTAPI + GRADIO SERVING APPLICATION
====================================
This script serves the Telco Customer Churn model via:
1. REST API (FastAPI) at /predict
2. Web UI (Gradio) at /ui
"""

import sys
import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

# === FIX: ADD PROJECT ROOT TO PATH (CORRECTED) ===
# Current file: G:\churn_project\src\app\main.py
# We need path: G:\churn_project\
# So we go up TWO levels: app -> src -> project_root

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

# Now safe to import 'src' modules
try:
    from src.serving.inference import predict
except ImportError as e:
    print(f"❌ Error importing inference module: {e}")
    print(f"   Debug: Project root added to path: {project_root}")
    raise e

# === 2. FASTAPI SETUP ===
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Production-ready API for churn prediction",
    version="1.0.0"
)

# Health Check
@app.get("/")
def root():
    return {"status": "ok", "service": "churn-prediction"}

# === 3. DATA SCHEMA (Pydantic) ===
class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

# === 4. API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        input_data = data.dict()
        result = predict(input_data)
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}

# === 5. GRADIO UI SETUP ===
def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    return predict(data)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="No"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method", value="Electronic check"),
        gr.Number(label="Tenure (months)", value=1),
        gr.Number(label="Monthly Charges ($)", value=70.0),
        gr.Number(label="Total Charges ($)", value=70.0),
    ],
    outputs="text",
    title="🔮 Telco Customer Churn Predictor",
    description="Enter customer details to predict churn risk."
)

app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
