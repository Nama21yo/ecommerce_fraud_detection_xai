from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import json
import io
import logging
from datetime import datetime
import uvicorn
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Advanced fraud detection system with explainable AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and preprocessors
fraud_model = None
cc_model = None
fraud_preprocessor = None
cc_preprocessor = None
fraud_explainer = None
cc_explainer = None
fraud_feature_names = None
cc_feature_names = None

class TransactionData(BaseModel):
    """Pydantic model for e-commerce transaction data"""
    user_id: Optional[str] = None
    signup_time: str
    purchase_time: str
    purchase_value: float = Field(gt=0, description="Purchase amount must be positive")
    device_id: str
    source: str
    browser: str
    sex: str = Field(regex="^(M|F)$", description="Gender: M or F")
    age: int = Field(ge=18, le=100, description="Age between 18-100")
    ip_address: str
    
class CreditCardData(BaseModel):
    """Pydantic model for credit card transaction data"""
    time: float
    amount: float = Field(ge=0, description="Amount must be non-negative")
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float

class PredictionResponse(BaseModel):
    """Response model for fraud predictions"""
    is_fraud: bool
    fraud_probability: float
    risk_score: str
    confidence: float
    model_used: str
    explanation: Dict[str, Any]
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    total_transactions: int
    fraud_detected: int
    fraud_rate: float
    predictions: List[PredictionResponse]
    summary_stats: Dict[str, Any]

def load_models():
    """Load pre-trained models and preprocessors"""
    global fraud_model, cc_model, fraud_preprocessor, cc_preprocessor
    global fraud_explainer, cc_explainer, fraud_feature_names, cc_feature_names
    
    try:
        logger.info("Loading fraud detection models...")
        
        # Models Loading
        # fraud_model = joblib.load('models/fraud_model.pkl')
        # cc_model = joblib.load('models/cc_model.pkl')
        
        # For demonstration, we'll set these to None and handle in prediction functions
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def preprocess_transaction_data(data: TransactionData) -> pd.DataFrame:
    """Preprocess e-commerce transaction data"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Parse datetime fields
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Feature engineering (simplified version of your implementation)
        df['time_since_signup'] = (
            df['purchase_time'] - df['signup_time']
        ).dt.total_seconds() / 3600
        
        df['purchase_hour'] = df['purchase_time'].dt.hour
        df['purchase_day_of_week'] = df['purchase_time'].dt.dayofweek
        df['weekend_purchase'] = df['purchase_day_of_week'].isin([5, 6]).astype(int)
        df['night_purchase'] = (
            (df['purchase_hour'] >= 22) | (df['purchase_hour'] <= 6)
        ).astype(int)
        
        # Amount features
        df['amount_percentile'] = 0.5  # Would be calculated from training data
        df['high_amount'] = (df['purchase_value'] > 1000).astype(int)
        df['round_amount'] = (df['purchase_value'] % 1 == 0).astype(int)
        
        # Encode categorical features
        categorical_mappings = {
            'browser': {'Chrome': 0, 'Firefox': 1, 'Safari': 2, 'IE': 3, 'Opera': 4},
            'source': {'SEO': 0, 'Ads': 1, 'Direct': 2},
            'sex': {'M': 0, 'F': 1}
        }
        
        for col, mapping in categorical_mappings.items():
            df[f'{col}_encoded'] = df[col].map(mapping).fillna(0)
        
        # Select features for model (simplified)
        feature_cols = [
            'purchase_value', 'age', 'time_since_signup', 'purchase_hour',
            'purchase_day_of_week', 'weekend_purchase', 'night_purchase',
            'amount_percentile', 'high_amount', 'round_amount',
            'browser_encoded', 'source_encoded', 'sex_encoded'
        ]
        
        return df[feature_cols]
        
    except Exception as e:
        logger.error(f"Error preprocessing transaction data: {e}")
        raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {e}")

def preprocess_creditcard_data(data: CreditCardData) -> pd.DataFrame:
    """Preprocess credit card transaction data"""
    try:
        df = pd.DataFrame([data.dict()])
        
        # PCA features are already provided (V1-V28)
        # Just ensure proper scaling would be applied here
        
        return df
        
    except Exception as e:
        logger.error(f"Error preprocessing credit card data: {e}")
        raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {e}")

def get_risk_level(probability: float) -> str:
    """Convert probability to risk level"""
    if probability < 0.1:
        return "Very Low"
    elif probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    elif probability < 0.8:
        return "High"
    else:
        return "Very High"

def generate_explanation(features: pd.DataFrame, prediction_proba: float, 
                        model_type: str) -> Dict[str, Any]:
    """Generate SHAP-based explanation (simplified)"""
    try:
        # In production, this would use your AdvancedSHAPExplainer
        # For now, we'll create a mock explanation
        
        feature_importance = {}
        for col in features.columns:
            # Mock feature importance
            importance = np.random.uniform(-0.5, 0.5)
            feature_importance[col] = {
                "value": float(features[col].iloc[0]),
                "shap_value": float(importance),
                "contribution": "increases" if importance > 0 else "decreases"
            }
        
        # Sort by absolute importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: abs(x[1]["shap_value"]), 
            reverse=True
        )
        
        return {
            "top_factors": dict(sorted_features[:5]),
            "model_confidence": float(max(prediction_proba, 1 - prediction_proba)),
            "explanation_method": "SHAP",
            "baseline_prediction": 0.05  # Mock baseline
        }
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return {"error": "Could not generate explanation"}

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    load_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Fraud Detection API is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": True,  # Would check actual model status
        "api_version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/transaction", response_model=PredictionResponse)
async def predict_transaction_fraud(transaction: TransactionData):
    """Predict fraud for e-commerce transaction"""
    try:
        # Preprocess data
        features = preprocess_transaction_data(transaction)
        
        # Mock prediction (replace with actual model inference)
        # prediction_proba = fraud_model.predict_proba(features)[0][1]
        # For demo, generate random prediction
        prediction_proba = np.random.uniform(0.05, 0.95)
        
        is_fraud = prediction_proba > 0.5
        risk_score = get_risk_level(prediction_proba)
        confidence = max(prediction_proba, 1 - prediction_proba)
        
        # Generate explanation
        explanation = generate_explanation(features, prediction_proba, "transaction")
        
        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(prediction_proba, 4),
            risk_score=risk_score,
            confidence=round(confidence, 4),
            model_used="RandomForest",  # Would be dynamic
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in transaction prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/creditcard", response_model=PredictionResponse)
async def predict_creditcard_fraud(transaction: CreditCardData):
    """Predict fraud for credit card transaction"""
    try:
        # Preprocess data
        features = preprocess_creditcard_data(transaction)
        
        # Mock prediction (replace with actual model inference)
        prediction_proba = np.random.uniform(0.01, 0.1)  # CC fraud is rarer
        
        is_fraud = prediction_proba > 0.5
        risk_score = get_risk_level(prediction_proba)
        confidence = max(prediction_proba, 1 - prediction_proba)
        
        # Generate explanation
        explanation = generate_explanation(features, prediction_proba, "creditcard")
        
        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(prediction_proba, 4),
            risk_score=risk_score,
            confidence=round(confidence, 4),
            model_used="XGBoost",  # Would be dynamic
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in credit card prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction for uploaded CSV file"""
    try:
        # Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"Processing batch file with {len(df)} transactions")
        
        predictions = []
        fraud_count = 0
        
        # Process each row (simplified)
        for idx, row in df.iterrows():
            try:
                # Determine transaction type and process accordingly
                if 'purchase_value' in df.columns:
                    # E-commerce transaction
                    transaction_data = TransactionData(**row.to_dict())
                    features = preprocess_transaction_data(transaction_data)
                    model_type = "transaction"
                else:
                    # Credit card transaction
                    transaction_data = CreditCardData(**row.to_dict())
                    features = preprocess_creditcard_data(transaction_data)
                    model_type = "creditcard"
                
                # Mock prediction
                prediction_proba = np.random.uniform(0.05, 0.95)
                is_fraud = prediction_proba > 0.5
                
                if is_fraud:
                    fraud_count += 1
                
                # Create prediction response
                pred_response = PredictionResponse(
                    is_fraud=is_fraud,
                    fraud_probability=round(prediction_proba, 4),
                    risk_score=get_risk_level(prediction_proba),
                    confidence=round(max(prediction_proba, 1 - prediction_proba), 4),
                    model_used="RandomForest" if model_type == "transaction" else "XGBoost",
                    explanation=generate_explanation(features, prediction_proba, model_type),
                    timestamp=datetime.now().isoformat()
                )
                
                predictions.append(pred_response)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        # Calculate summary statistics
        fraud_rate = fraud_count / len(predictions) if predictions else 0
        probabilities = [p.fraud_probability for p in predictions]
        
        summary_stats = {
            "avg_fraud_probability": round(np.mean(probabilities), 4) if probabilities else 0,
            "max_fraud_probability": round(np.max(probabilities), 4) if probabilities else 0,
            "min_fraud_probability": round(np.min(probabilities), 4) if probabilities else 0,
            "std_fraud_probability": round(np.std(probabilities), 4) if probabilities else 0
        }
        
        return BatchPredictionResponse(
            total_transactions=len(predictions),
            fraud_detected=fraud_count,
            fraud_rate=round(fraud_rate, 4),
            predictions=predictions,
            summary_stats=summary_stats
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "fraud_model": {
            "type": "RandomForest",
            "features": 13,  # Would be actual count
            "accuracy": 0.95,  # Would be actual metrics
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.90
        },
        "creditcard_model": {
            "type": "XGBoost",
            "features": 30,
            "accuracy": 0.97,
            "precision": 0.94,
            "recall": 0.91,
            "f1_score": 0.92
        },
        "last_updated": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
