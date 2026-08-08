from typing import List

import mlflow
import mlflow.sklearn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

MODEL_NAME = "BreastCancerClassifier"
MODEL_VERSION = "1"


# ============================================================
# INITIALIZE MLFLOW
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ============================================================
# LOAD REGISTERED MODEL
# ============================================================

MODEL_PATH = "models/breast_cancer"

model = mlflow.sklearn.load_model(MODEL_PATH)


# ============================================================
# INITIALIZE FASTAPI
# ============================================================

app = FastAPI(
    title="Breast Cancer Classification API",
    description="MLOps prediction API using a registered MLflow model",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 breast cancer feature values"
    )


# ============================================================
# RESPONSE MODEL
# ============================================================

class PredictionResponse(BaseModel):

    prediction: int
    model_name: str
    model_version: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Breast Cancer Classification API is running",
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    try:

        prediction = model.predict(
            [request.features]
        )

        return PredictionResponse(
            prediction=int(prediction[0]),
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )