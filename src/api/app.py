import sys
from pathlib import Path

# Add src directory to PYTHONPATH
sys.path.append("/Users/omvibhandik/Desktop/branch_loan_analysis/branch_loan_analysis/src")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import logging
from models.model import LoanPredictionModel
from data.database import DatabaseConnector
from config import Config
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Initialize FastAPI app
app = FastAPI(
  title="Loan Prediction API",
  description="API for predicting loan outcomes",
  version="1.0.0"
)

# Initialize global variables
model = None
db = None

class PredictionRequest(BaseModel):
  user_id: str
  age: int
  cash_incoming_30days: float
  application_timestamp: str

  class Config:
      schema_extra = {
          "example": {
              "user_id": "USER123",
              "age": 35,
              "cash_incoming_30days": 50000,
              "application_timestamp": "2024-02-20T10:30:15.123Z"
          }
      }

@app.on_event("startup")
async def startup_event():
  """Initialize model and database connection on startup"""
  global model, db
  try:
      # Load model
      logger.info("Loading model...")
      model = LoanPredictionModel.load(Config.MODEL_PATH)
      logger.info("Model loaded successfully")
      
      # Initialize database connection
      logger.info("Connecting to database...")
      db = DatabaseConnector(Config.DB_CONFIG)
      logger.info("Database connected successfully")
  except Exception as e:
      logger.error(f"Startup error: {str(e)}")
      raise
     

@app.on_event("shutdown")
async def shutdown_event():
  """Clean up resources on shutdown"""
  global db
  if db:
      db.close()
      logger.info("Database connection closed")

@app.get("/health")
async def health_check():
  """Check API health status"""
  return {
      "status": "healthy",
      "timestamp": datetime.now().isoformat(),
      "model_metadata": model.get_model_metadata() if model else None
  }



class PredictionResponse(BaseModel):
  user_id: str
  prediction_probability: float
  risk_level: str
  timestamp: str
  feature_importance: Dict[str, float]

# Data models
class GPSData(BaseModel):
    accuracy: float
    altitude: float
    bearing: float
    latitude: float
    longitude: float
    location_provider: str
    gps_fix_at: str

class PredictionRequest(BaseModel):
    user_id: str
    age: int
    cash_incoming_30days: float
    application_timestamp: str
    gps_data: List[GPSData]  # Make it required since model needs GPS features

def _get_age_group_numeric(age: int) -> int:
    """Convert age to numeric group"""
    if age < 25:
        return 0  # young
    elif age < 35:
        return 1  # adult
    elif age < 50:
        return 2  # middle_aged
    else:
        return 3  # senior

def _get_income_quartile_numeric(income: float) -> int:
    """Determine income quartile numerically"""
    if income < 25000:
        return 0
    elif income < 50000:
        return 1
    elif income < 75000:
        return 2
    else:
        return 3

@app.post("/predict", response_model=PredictionResponse)
async def predict_loan(request: PredictionRequest):
    """Make loan prediction for a user"""
    try:
        # 1. Process GPS data
        gps_df = pd.DataFrame([{
            'user_id': request.user_id,
            'accuracy': gps.accuracy,
            'altitude': gps.altitude,
            'bearing': gps.bearing,
            'latitude': gps.latitude,
            'longitude': gps.longitude,
            'location_provider': gps.location_provider,
            'gps_fix_at': pd.to_datetime(gps.gps_fix_at),
            'server_upload_at': datetime.now()
        } for gps in request.gps_data])

        # Calculate GPS features
        gps_features = pd.DataFrame({
            'user_id': [request.user_id],
            'gps_accuracy_mean': [gps_df['accuracy'].mean()],
            'gps_accuracy_min': [gps_df['accuracy'].min()],
            'gps_accuracy_max': [gps_df['accuracy'].max()],
            'gps_accuracy_std': [gps_df['accuracy'].std()],
            'gps_altitude_mean': [gps_df['altitude'].mean()],
            'gps_altitude_min': [gps_df['altitude'].min()],
            'gps_altitude_max': [gps_df['altitude'].max()],
            'gps_altitude_std': [gps_df['altitude'].std()],
            'gps_bearing_mean': [gps_df['bearing'].mean()],
            'gps_bearing_std': [gps_df['bearing'].std()],
            'gps_bearing_min': [gps_df['bearing'].min()],
            'gps_bearing_max': [gps_df['bearing'].max()],
            'gps_latitude_mean': [gps_df['latitude'].mean()],
            'gps_latitude_std': [gps_df['latitude'].std()],
            'gps_latitude_min': [gps_df['latitude'].min()],
            'gps_latitude_max': [gps_df['latitude'].max()],
            'gps_longitude_mean': [gps_df['longitude'].mean()],
            'gps_longitude_std': [gps_df['longitude'].std()],
            'gps_longitude_min': [gps_df['longitude'].min()],
            'gps_longitude_max': [gps_df['longitude'].max()],
            'gps_upload_delay_mean': [(gps_df['server_upload_at'] - gps_df['gps_fix_at']).dt.total_seconds().mean()],
            'gps_upload_delay_max': [(gps_df['server_upload_at'] - gps_df['gps_fix_at']).dt.total_seconds().max()],
            'gps_upload_delay_min': [(gps_df['server_upload_at'] - gps_df['gps_fix_at']).dt.total_seconds().min()],
            'gps_location_provider_<lambda>': [gps_df['location_provider'].value_counts().index[0]],
            'gps_gps_fix_at_count': [len(gps_df)],
            'gps_location_stability': [1 / (1 + gps_df['accuracy'].std())],
            'gps_provider_reliability': [1.0 if gps_df['location_provider'].iloc[0].lower() == 'gps_data' else 0.7],
            'gps_movement_pattern': [np.abs(np.cos(np.radians(gps_df['bearing'])).mean())]
        })

        # 2. Create base features
        application_dt = pd.to_datetime(request.application_timestamp)
        base_features = pd.DataFrame([{
            'user_id': request.user_id,
            'age': request.age,
            'cash_incoming_30days': request.cash_incoming_30days,
            'application_at_hour': application_dt.hour,
            'application_at_day': application_dt.day,
            'application_at_month': application_dt.month,
            'application_at_year': application_dt.year,
            'application_at_dayofweek': application_dt.dayofweek,
            'hour_of_day': application_dt.hour,
            'day_of_week': application_dt.dayofweek,
            'age_group': _get_age_group_numeric(request.age),
            'income_quartile': _get_income_quartile_numeric(request.cash_incoming_30days)
        }])

         # Ensure all features are numeric to avoid conversion issues
        features = pd.merge(base_features, gps_features, on='user_id', how='left')
        features = features.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert all fields to numeric, filling NaNs

        # # 3. Merge all features
        # features = pd.merge(base_features, gps_features, on='user_id', how='left')

        # 4. Make prediction
        prediction_prob = float(model.predict(features)[0])

        # 5. Get feature importance
        feature_importance = model.get_feature_importance()

        # 6. Determine risk level
        risk_level = _get_risk_level(prediction_prob)

        response = {
            "user_id": request.user_id,
            "prediction_probability": prediction_prob,
            "risk_level": risk_level,
            "timestamp": datetime.now().isoformat(),
            "feature_importance": feature_importance
        }

        logger.info(f"Prediction made for user {request.user_id}: {prediction_prob:.3f}")
        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_risk_level(probability: float) -> str:
  """Convert probability to risk level"""
  if probability < 0.2:
      return "very_low"
  elif probability < 0.4:
      return "low"
  elif probability < 0.6:
      return "medium"
  elif probability < 0.8:
      return "high"
  else:
      return "very_high"

@app.get("/model/info")
async def get_model_info():
  """Get model information and metadata"""
  try:
      if model is None:
          raise HTTPException(
              status_code=500, 
              detail="Model not initialized"
          )
          
      metadata = {
          "model_version": "1.0.0",
          "model_type": "RandomForestClassifier",
          "features": [
              "age",
              "cash_incoming_30days",
              "hour_of_day",
              "day_of_week",
              "application_at_year",
              "application_at_month",
              "application_at_day",
              "application_at_hour",
              "application_at_dayofweek",
              "age_group",
              "income_quartile"
          ],
          "last_updated": model.model_metadata.get('training_date', datetime.now().isoformat()),
          "performance_metrics": model.model_metadata.get('performance_metrics', {}),
          "feature_importance": model.get_feature_importance(),
          "description": "Loan prediction model using Random Forest algorithm",
          "training_data_summary": {
              "total_samples": model.model_metadata.get('total_samples', 0),
              "target_distribution": model.model_metadata.get('target_distribution', {})
          },
          "model_parameters": model.model_metadata.get('model_parameters', {})
      }
      
      return metadata
      
  except Exception as e:
      logger.error(f"Error getting model info: {str(e)}")
      raise HTTPException(
          status_code=500,
          detail=f"Error retrieving model information: {str(e)}"
      )

def _get_risk_level(probability: float) -> str:
  """Convert probability to risk level"""
  if probability < 0.2:
      return "very_low"
  elif probability < 0.4:
      return "low"
  elif probability < 0.6:
      return "medium"
  elif probability < 0.8:
      return "high"
  else:
      return "very_high"

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(
      "app:app",
      host=Config.API_CONFIG.get('host', '0.0.0.0'),
      port=Config.API_CONFIG.get('port', 8000),
      reload=Config.API_CONFIG.get('reload', True)
  )