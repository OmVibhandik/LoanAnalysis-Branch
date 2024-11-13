import sys
from pathlib import Path

# Add src directory to PYTHONPATH
sys.path.append("/Users/omvibhandik/Desktop/branch_loan_analysis/branch_loan_analysis/src")

from ..models.model import LoanPredictionModel
from data.database import DatabaseConnector
from config import Config
import pandas as pd

def train_loan_model():
  try:
      # Initialize database connection
      db = DatabaseConnector(Config.DB_CONFIG)
      
      # Fetch data
      loan_data = db.get_loan_outcomes()
      user_data = db.get_user_attributes()
      gps_data = db.get_gps_fixes()
      
      # Merge data
      features_df = pd.merge(loan_data, user_data, on='user_id', how='left')
      
      # Initialize model
      model = LoanPredictionModel(Config.MODEL_CONFIG)
      
      # Prepare data
      X, y = model.prepare_data(features_df, gps_data,  target_column='loan_outcome')
      
      # Train model
      metrics = model.train(X, y, perform_grid_search=True)
      
      # Print metrics
      print("\nModel Performance Metrics:")
      print("==========================")
      print(f"ROC AUC Score: {metrics['roc_auc_score']:.4f}")
      print(f"Cross-validation Score: {metrics['cross_val_scores']['mean']:.4f}")
      
      # Save model
      model.save(Config.MODEL_PATH)
      
      return model, metrics
      
  except Exception as e:
      raise
  finally:
      db.close()

if __name__ == "__main__":
  model, metrics = train_loan_model()