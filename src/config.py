import sys
from pathlib import Path

# # Add src directory to PYTHONPATH
sys.path.append("/Users/omvibhandik/Desktop/branch_loan_analysis/branch_loan_analysis/src")

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
  # Project structure
  BASE_DIR = Path(__file__).parent.parent
  DATA_DIR = BASE_DIR / 'data'
  MODEL_PATH = Path(__file__).parent / "models" / "loan_prediction_model.pkl"
  
  # Create directories if they don't exist
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  MODELS_DIR = Path(__file__).parent / "models"
  LOG_DIR = os.path.join(BASE_DIR, 'logs')


  # Database configuration
  DB_CONFIG = {
      'host': os.getenv('DB_HOST', 'branchhomeworkdb.cv8nj4hg6yra.ap-south-1.rds.amazonaws.com'),
      'port': int(os.getenv('DB_PORT', 5432)),
      'user': os.getenv('DB_USER', 'datascientist'),
      'password': os.getenv('DB_PASSWORD', '47eyYBLT0laW5j9U24Uuy8gLcrN'),
      'database': os.getenv('DB_NAME', 'branchdsprojectgps')
  }
    # Logging configuration
  LOGGING_CONFIG = {
      'level': 'DEBUG',
      'console_level': 'INFO',
      'file_level': 'DEBUG',
      'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
      'date_format': '%Y-%m-%d %H:%M:%S',
      'log_dir': LOG_DIR,
      'max_bytes': 10485760,  # 10MB
      'backup_count': 5,
      'error_backup_count': 5
  }

  # Model configuration
  MODEL_CONFIG = {
      'n_estimators': 100,
      'max_depth': 10,
      'min_samples_split': 2,
      'min_samples_leaf': 1,
      'random_state': 42,
      'test_size': 0.2,
      'cross_validation_folds': 5
  }

  # Feature engineering configuration
  FEATURE_CONFIG = {
      'min_gps_points': 3,
      'max_speed_kmh': 150,
      'min_accuracy': 0,
      'distance_threshold_km': 100,
      'time_window_hours': 24,
      'categorical_features': ['location_provider', 'age_group', 'income_category'],
      'numeric_features': ['accuracy', 'altitude', 'bearing', 'latitude', 'longitude']
  }

  # API configuration
  API_CONFIG = {
      'allowed_origins': '*',
      'host': '0.0.0.0',
      'port': 8000,
      'debug': True,
      'reload': True,
      'workers': 4
  }

  # Logging configuration
  LOG_CONFIG = {
      'version': 1,
      'disable_existing_loggers': False,
      'formatters': {
          'standard': {
              'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
          },
      },
      'handlers': {
          'default': {
              'level': 'INFO',
              'formatter': 'standard',
              'class': 'logging.StreamHandler',
          },
          'file': {
              'level': 'INFO',
              'formatter': 'standard',
              'class': 'logging.FileHandler',
              'filename': str(BASE_DIR / 'logs' / 'app.log'),
              'mode': 'a',
          },
      },
      'loggers': {
          '': {
              'handlers': ['default', 'file'],
              'level': 'INFO',
              'propagate': True
          }
      }
  }
  METRICS_CONFIG = {
      'classification_metrics': [
          'accuracy',
          'precision',
          'recall',
          'f1',
          'roc_auc'
      ],
      'threshold': 0.5,
      'cv_folds': 5,
      'scoring': {
          'accuracy': 'accuracy',
          'precision': 'precision',
          'recall': 'recall',
          'f1': 'f1',
          'roc_auc': 'roc_auc'
      },
      'report_format': {
          'decimals': 3,
          'include_confusion_matrix': True,
          'include_classification_report': True
      },
      'visualization': {
          'plot_roc_curve': True,
          'plot_precision_recall_curve': True,
          'plot_feature_importance': True
      }
  }

  # Model file paths
  MODEL_PATH = "/Users/omvibhandik/Desktop/branch_loan_analysis/branch_loan_analysis/src/models/loan_prediction_model.pkl"
  LABEL_ENCODER_PATH = str(MODELS_DIR / 'label_encoder.pkl')
  FEATURE_ENCODER_PATH = str(MODELS_DIR / 'feature_encoder.pkl')
  
  # Threshold for loan approval
  LOAN_APPROVAL_THRESHOLD = 0.7

  @staticmethod
  def get_db_uri():
      """Get database URI for SQLAlchemy"""
      return (
          f"postgresql://{Config.DB_CONFIG['user']}:{Config.DB_CONFIG['password']}"
          f"@{Config.DB_CONFIG['host']}:{Config.DB_CONFIG['port']}/{Config.DB_CONFIG['database']}"
      )

  @staticmethod
  def validate_config():
      """Validate configuration settings"""
      required_env_vars = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME']
      missing_vars = [var for var in required_env_vars if not os.getenv(var)]
      
      if missing_vars:
          raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
      
      if not os.path.exists(Config.BASE_DIR):
          raise ValueError(f"Base directory does not exist: {Config.BASE_DIR}")