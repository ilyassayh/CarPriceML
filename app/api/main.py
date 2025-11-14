"""
FastAPI Backend for CarPriceML

This API serves as the backend for the CarPriceML application.
It provides endpoints for car price prediction using a trained machine learning model.

The API is consumed by the Streamlit frontend (app/frontend/streamlit_app.py),
which sends HTTP requests to the /predict endpoint to get price predictions.

Endpoints:
    - GET /health: Health check endpoint
    - POST /predict: Predict car price based on provided features
"""

# Standard library imports
import json
from pathlib import Path
from typing import Any, Dict

# Third-party imports
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Define paths relative to the project root
# APP_ROOT is two levels up from this file (app/api/ -> project root)
APP_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = APP_ROOT / 'models'
MODEL_PATH = MODELS_DIR / 'rf_model.joblib'
META_PATH = MODELS_DIR / 'metadata.json'

# Initialize FastAPI application
# This API is consumed by the Streamlit frontend (app/frontend/streamlit_app.py)
app = FastAPI(title='CarPriceML API', version='1.0.0')

# Configure CORS middleware to allow cross-origin requests
# This enables the Streamlit frontend (app/frontend/streamlit_app.py) to communicate with this API
# The frontend sends POST requests to /predict endpoint from a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allow all origins (consider restricting in production)
    allow_credentials=True,
    allow_methods=['*'],  # Allow all HTTP methods
    allow_headers=['*'],  # Allow all headers
)


def load_artifacts():
    """
    Load the trained model and metadata from disk.
    
    Returns:
        tuple: (model, metadata) - The loaded model and metadata dictionary
        
    Raises:
        FileNotFoundError: If model or metadata files don't exist
    """
    # Check if required files exist before attempting to load
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError('Model or metadata not found. Please train the model first.')
    
    # Load the trained Random Forest model
    model = joblib.load(MODEL_PATH)
    
    # Load metadata containing feature information
    with META_PATH.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    
    return model, meta


@app.get('/health')
def health() -> Dict[str, Any]:
    """
    Health check endpoint to verify API and model availability.
    
    Returns:
        Dict containing status, model type, and expected features
    """
    try:
        # Attempt to load artifacts to verify they're available
        _model, meta = load_artifacts()
        # Return success status with model info and feature list
        return {'status': 'ok', 'model': 'rf', 'features': meta.get('numeric_features', []) + meta.get('categorical_features', [])}
    except Exception as e:
        # Return error status if artifacts can't be loaded
        return {'status': 'error', 'detail': str(e)}


@app.post('/predict')
def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict car price based on provided features.
    
    This endpoint is called by the Streamlit frontend (app/frontend/streamlit_app.py)
    when a user submits the car details form.
    
    Args:
        features: Dictionary containing car features (numeric and categorical)
                  Sent from the Streamlit frontend form
        
    Returns:
        Dict containing the predicted price in format: {'price': float}
        
    Raises:
        HTTPException: If model loading fails (500) or prediction fails (400)
    """
    # Load model and metadata
    try:
        model, meta = load_artifacts()
    except Exception as e:
        # Return 500 error if model can't be loaded
        raise HTTPException(status_code=500, detail=str(e))

    # Extract expected feature lists from metadata
    numeric_cols = meta.get('numeric_features', [])
    categorical_cols = meta.get('categorical_features', [])
    expected_cols = numeric_cols + categorical_cols

    # Build single-row DataFrame aligned to training columns
    # This ensures the input data matches the format expected by the model
    row = {col: features.get(col, None) for col in expected_cols}
    X = pd.DataFrame([row])

    # Make prediction using the loaded model
    try:
        pred = float(model.predict(X)[0])
    except Exception as e:
        # Return 400 error if prediction fails (e.g., invalid input data)
        raise HTTPException(status_code=400, detail=f'Prediction failed: {e}')

    # Return the predicted price
    return {'price': pred}


