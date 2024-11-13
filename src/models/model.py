import sys
from pathlib import Path

# Add src directory to PYTHONPATH
sys.path.append("/Users/omvibhandik/Desktop/branch_loan_analysis/branch_loan_analysis/src")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from typing import Dict, Tuple, Any, List
import json
from datetime import datetime
from pathlib import Path
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanPredictionModel:
  """Class for loan prediction model operations"""
  
  def __init__(self, config: Dict = None):
      """
      Initialize the model with configuration
      
      Args:
          config (Dict): Model configuration dictionary
      """
      self.config = config or Config.MODEL_CONFIG
      self.model = RandomForestClassifier(
          n_estimators=self.config['n_estimators'],
          max_depth=self.config['max_depth'],
          min_samples_split=self.config['min_samples_split'],
          min_samples_leaf=self.config['min_samples_leaf'],
          random_state=self.config['random_state']
      )
      self.label_encoder = LabelEncoder()
      self.feature_columns = None
      self.model_metadata = {}
      self.best_params = None
  def process_gps_features(self, gps_df: pd.DataFrame) -> pd.DataFrame:
        """Process GPS fixes data to create meaningful features"""
        # Convert timestamps to datetime
        gps_df['gps_fix_at'] = pd.to_datetime(gps_df['gps_fix_at'])
        gps_df['server_upload_at'] = pd.to_datetime(gps_df['server_upload_at'])
        
        # Calculate upload delay in seconds
        gps_df['upload_delay'] = (gps_df['server_upload_at'] - gps_df['gps_fix_at']).dt.total_seconds()
        
        # Group by user_id and create aggregate features
        gps_features = gps_df.groupby('user_id').agg({
            # Location accuracy metrics
            'accuracy': ['mean', 'min', 'max', 'std'],
            
            # Altitude variations
            'altitude': ['mean', 'min', 'max', 'std'],
            
            # Movement patterns
            'bearing': ['mean', 'std', 'min', 'max'],
            
            # Geographic spread
            'latitude': ['mean', 'std', 'min', 'max'],
            'longitude': ['mean', 'std', 'min', 'max'],
            
            # Upload behavior
            'upload_delay': ['mean', 'max', 'min'],
            
            # Location provider reliability
            'location_provider': lambda x: x.value_counts().index[0],
            
            # Number of GPS fixes
            'gps_fix_at': 'count'
        }).reset_index()
        
        # Flatten column names
        gps_features.columns = ['user_id'] + [
            f'gps_{col[0]}_{col[1]}' if col[1] != '' else f'gps_{col[0]}'
            for col in gps_features.columns[1:]
        ]
        
        # Add derived features
        gps_features['gps_location_stability'] = self._calculate_location_stability(gps_df)
        gps_features['gps_provider_reliability'] = self._calculate_provider_reliability(gps_df)
        gps_features['gps_movement_pattern'] = self._calculate_movement_pattern(gps_df)
        
        return gps_features

  def _calculate_location_stability(self, gps_df: pd.DataFrame) -> pd.Series:
        """Calculate location stability score based on accuracy and consistency"""
        stability_scores = gps_df.groupby('user_id').apply(
            lambda x: (
                (1 / (1 + x['accuracy'].std())) * 
                (1 - x['accuracy'].mean() / 100) *
                (len(x) / x['gps_fix_at'].nunique())
            )
        )
        return stability_scores

  def _calculate_provider_reliability(self, gps_df: pd.DataFrame) -> pd.Series:
        """Calculate reliability score based on location provider"""
        provider_scores = {
            'gps': 1.0,
            'network': 0.7,
            'passive': 0.3
        }
        
        return gps_df.groupby('user_id')['location_provider'].agg(
            lambda x: sum(provider_scores.get(p.lower(), 0.1) for p in x) / len(x)
        )

  def _calculate_movement_pattern(self, gps_df: pd.DataFrame) -> pd.Series:
        """Calculate movement pattern score based on bearing changes"""
        return gps_df.groupby('user_id').apply(
            lambda x: np.abs(np.cos(np.radians(x['bearing'])).mean())
        )

  def prepare_data(self, 
              features_df: pd.DataFrame, 
              gps_df: pd.DataFrame = None,
              target_column: str = 'loan_outcome') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for training
    
    Args:
    features_df (pd.DataFrame): Feature dataframe
    gps_df (pd.DataFrame): GPS fixes dataframe
    target_column (str): Name of target column
    
    Returns:
    Tuple[pd.DataFrame, pd.Series]: Prepared features and target
    """
    try:
        # Process GPS features if provided
        if gps_df is not None:
          gps_features = self.process_gps_features(gps_df)
          features_df = features_df.merge(gps_features, on='user_id', how='left')
        
        # Store feature columns
        self.feature_columns = [col for col in features_df.columns 
                                if col not in [target_column, 'user_id']]
        
        # Select features
        X = features_df[self.feature_columns].copy()
        
        # Handle datetime columns
        datetime_columns = X.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            # Extract useful datetime features
            X[f'{col}_hour'] = X[col].dt.hour
            X[f'{col}_day'] = X[col].dt.day
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_dayofweek'] = X[col].dt.dayofweek
            # Drop original datetime column
            X = X.drop(columns=[col])
        
        # Update feature columns after datetime processing
        self.feature_columns = X.columns.tolist()
        
        # Handle categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = pd.Categorical(X[col]).codes
        
        # Fill missing values
        X = X.fillna(X.mean())
        
        # Encode target variable
        y = self.label_encoder.fit_transform(features_df[target_column])
        
        # Store metadata
        self.model_metadata.update({
            'feature_columns': self.feature_columns,
            'target_column': target_column,
            'target_classes': self.label_encoder.classes_.tolist(),
            'datetime_columns': datetime_columns.tolist(),
            'categorical_columns': categorical_columns.tolist()
        })
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        raise

  def train(self, X: pd.DataFrame, y: pd.Series, perform_grid_search: bool = True) -> Dict[str, Any]:
      """
      Train the model and return performance metrics
      
      Args:
          X (pd.DataFrame): Feature matrix
          y (pd.Series): Target variable
          perform_grid_search (bool): Whether to perform grid search
          
      Returns:
          Dict[str, Any]: Performance metrics
      """
      try:
          # Split data
          X_train, X_test, y_train, y_test = train_test_split(
              X, y,
              test_size=self.config['test_size'],
              random_state=self.config['random_state'],
              stratify=y
          )
          
          if perform_grid_search:
              self._perform_grid_search(X_train, y_train)
          
          # Train model
          self.model.fit(X_train, y_train)
          
          # Calculate metrics
          metrics = self._calculate_metrics(X_train, X_test, y_train, y_test)
          
          # Store training metadata
          self.model_metadata.update({
              'training_date': datetime.now().isoformat(),
              'model_parameters': self.model.get_params(),
              'performance_metrics': metrics,
              'feature_importance': dict(zip(self.feature_columns, 
                                          self.model.feature_importances_))
          })
          
          return metrics
          
      except Exception as e:
          logger.error(f"Error in train: {str(e)}")
          raise

  def _perform_grid_search(self, X_train: pd.DataFrame, y_train: pd.Series):
      """
      Perform grid search for hyperparameter tuning
      
      Args:
          X_train (pd.DataFrame): Training features
          y_train (pd.Series): Training target
      """
      try:
          param_grid = {
              'n_estimators': [50, 100, 200],
              'max_depth': [5, 10, 15, None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]
          }
          
          grid_search = GridSearchCV(
              estimator=RandomForestClassifier(random_state=self.config['random_state']),
              param_grid=param_grid,
              cv=5,
              n_jobs=-1,
              scoring='roc_auc'
          )
          
          grid_search.fit(X_train, y_train)
          
          self.best_params = grid_search.best_params_
          self.model = RandomForestClassifier(**self.best_params, 
                                            random_state=self.config['random_state'])
          
          logger.info(f"Best parameters found: {self.best_params}")
          
      except Exception as e:
          logger.error(f"Error in grid search: {str(e)}")
          raise

  def _calculate_metrics(self, 
                       X_train: pd.DataFrame, 
                       X_test: pd.DataFrame, 
                       y_train: pd.Series, 
                       y_test: pd.Series) -> Dict[str, Any]:
      """
      Calculate model performance metrics
      
      Args:
          X_train, X_test (pd.DataFrame): Training and test features
          y_train, y_test (pd.Series): Training and test targets
          
      Returns:
          Dict[str, Any]: Dictionary of metrics
      """
      try:
          # Make predictions
          y_pred = self.model.predict(X_test)
          y_pred_proba = self.model.predict_proba(X_test)[:, 1]
          
          # Calculate cross-validation score
          cv_scores = cross_val_score(
              self.model, X_train, y_train, 
              cv=self.config['cross_validation_folds'],
              scoring='roc_auc'
          )
          
          metrics = {
              'classification_report': classification_report(
                  y_test, y_pred, output_dict=True
              ),
              'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
              'roc_auc_score': roc_auc_score(y_test, y_pred_proba),
              'cross_val_scores': {
                  'mean': cv_scores.mean(),
                  'std': cv_scores.std(),
                  'scores': cv_scores.tolist()
              }
          }
          
          return metrics
          
      except Exception as e:
          logger.error(f"Error in calculate_metrics: {str(e)}")
          raise

  def predict(self, X: pd.DataFrame) -> np.ndarray:
      """
      Make predictions for new data
      
      Args:
          X (pd.DataFrame): Feature matrix
          
      Returns:
          np.ndarray: Predicted probabilities
      """
      try:
          # Ensure all required features are present
          missing_features = set(self.feature_columns) - set(X.columns)
          if missing_features:
              raise ValueError(f"Missing features: {missing_features}")
          
          # Make predictions
          predictions = self.model.predict_proba(X[self.feature_columns])[:, 1]
          
          return predictions
          
      except Exception as e:
          logger.error(f"Error in predict: {str(e)}")
          raise

  def save(self, model_path: str = None):
    """Save the model"""
    try:
        # Save model
        model_path = model_path or Config.MODEL_PATH
        
        # Ensure the directory exists
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save using joblib instead of pickle
        joblib.dump(self, model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

  @classmethod
  def load(cls, model_path: str) -> 'LoanPredictionModel':
    """Load a saved model"""
    try:
        # Add the project root to Python path
        import sys
        project_root = Path(model_path).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            
        # Load the model
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

  def get_feature_importance(self) -> Dict[str, float]:
      """
      Get feature importance scores
      
      Returns:
          Dict[str, float]: Feature importance scores
      """
      try:
          importance_scores = self.model.feature_importances_
          feature_importance = dict(zip(self.feature_columns, importance_scores))
          return dict(sorted(feature_importance.items(), 
                           key=lambda x: x[1], 
                           reverse=True))
          
      except Exception as e:
          logger.error(f"Error getting feature importance: {str(e)}")
          raise

  def get_model_metadata(self) -> Dict[str, Any]:
      """
      Get model metadata
      
      Returns:
          Dict[str, Any]: Model metadata
      """
      return self.model_metadata

  def validate_input(self, X: pd.DataFrame) -> bool:
      """
      Validate input data
      
      Args:
          X (pd.DataFrame): Input features
          
      Returns:
          bool: Whether input is valid
      """
      try:
          # Check if all required features are present
          missing_features = set(self.feature_columns) - set(X.columns)
          if missing_features:
              raise ValueError(f"Missing features: {missing_features}")
          
          # Check data types
          for column in self.feature_columns:
              if X[column].dtype != X[column].dtype:
                  raise ValueError(f"Invalid data type for feature {column}")
          
          return True
          
      except Exception as e:
          logger.error(f"Input validation error: {str(e)}")
          return False