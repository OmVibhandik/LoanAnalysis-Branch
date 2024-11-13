# Branch Loan Analysis Project

A machine learning system for predicting loan outcomes using customer data, GPS location patterns, and financial behaviors. The system implements a FastAPI-based REST API for real-time predictions.

## Overview

This project implements a loan prediction system that uses:
- Customer demographic data
- Financial transaction history
- GPS location patterns and movement analysis
- Machine learning (Random Forest Classifier)
- REST API for real-time predictions

## Tech Stack

- Python 3.8+
- FastAPI for API development
- Pandas for data processing
- Scikit-learn for machine learning
- SQLAlchemy for database operations
- Uvicorn for ASGI server
- Jupyter Notebooks for analysis

## Project Structure

```
branch_loan_analysis/
├── .venv/                  # Virtual environment
├── data/                   # Data management modules
├── logs/                   # Application logs
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── analysis/         # Analysis modules
│   ├── api/              # API implementation
│   ├── data/             # Data processing
│   ├── features/         # Feature engineering
│   ├── models/           # ML model implementation
│   ├── scripts/          # Training and utility scripts
│   ├── test/            # Test modules
│   └── utils/           # Utility functions
├── .env                  # Environment variables
├── README.md            # Project documentation
└── requirements.txt     # Project dependencies
```

## Installation & Setup

1. Clone the repository and create virtual environment:
```bash
git clone https://github.com/yourusername/branch_loan_analysis.git
cd branch_loan_analysis
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Data Analysis

The project includes comprehensive data analysis in two forms:

### 1. Jupyter Notebook Analysis
```bash
cd notebooks
jupyter notebook data_exploration.ipynb
```

### 2. Automated Insights Generation
```bash
cd src/analysis
python insights_generator.py
```
This script generates `loan_analysis_insights.png` with visualizations of:
- Feature importance plots
- Correlation matrices
- GPS pattern analysis
- Risk distribution charts

Key analysis includes:
- Customer demographic patterns
- Transaction behavior analysis
- GPS location pattern analysis
- Feature importance evaluation

## Model Training

The model uses Random Forest Classifier for predictions, chosen for:
- Handling non-linear relationships
- Feature importance insights
- Robust to outliers
- Good performance with mixed data types

To train the model:
```bash
python src/scripts/train_model.py
```

The training process includes:
- Data preprocessing
- Feature engineering
- Grid search for hyperparameter optimization
- Cross-validation
- Model evaluation

## Feature Engineering

The system processes several types of features:

### Basic Features:
- Age and demographic data
- Financial transaction history
- Application timing features

### GPS Features:
- Location stability metrics
- Movement patterns
- Location accuracy analysis
- Time-based location features

## Running the API

Start the FastAPI application:
```bash
cd src/api
uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

The service exposes three main endpoints:

### 1. Health Check Endpoint
```
GET /health
```
Returns:
- API health status
- Current timestamp
- Model metadata
- Used for monitoring and status checks

Example response:
```json
{
    "status": "healthy",
    "timestamp": "2024-11-14T03:14:27.795000",
    "model_metadata": {
        "feature_columns": [
            "day_of_week",
            "hour_of_day",
            "age",
            "cash_incoming_30days",
            "age_group",
            "income_quartile",
            "gps_accuracy_mean",
            "gps_accuracy_min",
            "gps_accuracy_max",
            "gps_accuracy_std",
            "gps_altitude_mean",
            "gps_altitude_min",
            "gps_altitude_max",
            "gps_altitude_std",
            "gps_bearing_mean",
            "gps_bearing_std",
            "gps_bearing_min",
            "gps_bearing_max",
            "gps_latitude_mean",
            "gps_latitude_std",
            "gps_latitude_min",
            "gps_latitude_max",
            "gps_longitude_mean",
            "gps_longitude_std",
            "gps_longitude_min",
            "gps_longitude_max",
            "gps_upload_delay_mean",
            "gps_upload_delay_max",
            "gps_upload_delay_min",
            "gps_location_provider_<lambda>",
            "gps_gps_fix_at_count",
            "gps_location_stability",
            "gps_provider_reliability",
            "gps_movement_pattern",
            "application_at_hour",
            "application_at_day",
            "application_at_month",
            "application_at_year",
            "application_at_dayofweek"
        ],
        "target_column": "loan_outcome",
        "target_classes": [
            "defaulted",
            "repaid"
        ],
        "datetime_columns": [
            "application_at"
        ],
        "categorical_columns": [
            "age_group",
            "gps_location_provider_<lambda>"
        ],
        "training_date": "2024-11-14T02:00:44.218370",
        "model_parameters": {
            "bootstrap": true,
            "ccp_alpha": 0.0,
            "class_weight": null,
            "criterion": "gini",
            "max_depth": 10,
            "max_features": "sqrt",
            "max_leaf_nodes": null,
            "max_samples": null,
            "min_impurity_decrease": 0.0,
            "min_samples_leaf": 4,
            "min_samples_split": 10,
            "min_weight_fraction_leaf": 0.0,
            "monotonic_cst": null,
            "n_estimators": 50,
            "n_jobs": null,
            "oob_score": false,
            "random_state": 42,
            "verbose": 0,
            "warm_start": false
        },
        "performance_metrics": {
            "classification_report": {
                "0": {
                    "precision": 0.6304347826086957,
                    "recall": 0.725,
                    "f1-score": 0.6744186046511628,
                    "support": 40.0
                },
                "1": {
                    "precision": 0.6764705882352942,
                    "recall": 0.575,
                    "f1-score": 0.6216216216216216,
                    "support": 40.0
                },
                "accuracy": 0.65,
                "macro avg": {
                    "precision": 0.6534526854219949,
                    "recall": 0.6499999999999999,
                    "f1-score": 0.6480201131363922,
                    "support": 80.0
                },
                "weighted avg": {
                    "precision": 0.653452685421995,
                    "recall": 0.65,
                    "f1-score": 0.6480201131363922,
                    "support": 80.0
                }
            },
            "confusion_matrix": [
                [
                    29,
                    11
                ],
                [
                    17,
                    23
                ]
            ],
            "roc_auc_score": 0.7287500000000001,
            "cross_val_scores": {
                "mean": 0.7451171875,
                "std": 0.09472152853892297,
                "scores": [
                    0.7890625,
                    0.7119140625,
                    0.595703125,
                    0.744140625,
                    0.884765625
                ]
            }
        },
        "feature_importance": {
            "day_of_week": 0.01186071677083524,
            "hour_of_day": 0.022550254255733572,
            "age": 0.07845927568012889,
            "cash_incoming_30days": 0.05372791037410548,
            "age_group": 0.015370736303683706,
            "income_quartile": 0.012086098190554688,
            "gps_accuracy_mean": 0.03051923915093847,
            "gps_accuracy_min": 0.023097029459706205,
            "gps_accuracy_max": 0.021766553134933093,
            "gps_accuracy_std": 0.02136972544200025,
            "gps_altitude_mean": 0.0302417114297488,
            "gps_altitude_min": 0.007737723430411209,
            "gps_altitude_max": 0.034740745612455944,
            "gps_altitude_std": 0.03655208647321864,
            "gps_bearing_mean": 0.02009782176490872,
            "gps_bearing_std": 0.030027230120541103,
            "gps_bearing_min": 0.002282267278537579,
            "gps_bearing_max": 0.015220475868341441,
            "gps_latitude_mean": 0.01890067855422683,
            "gps_latitude_std": 0.03283744489963296,
            "gps_latitude_min": 0.019516507110117952,
            "gps_latitude_max": 0.02520461800373176,
            "gps_longitude_mean": 0.02459680823442007,
            "gps_longitude_std": 0.026433520926256313,
            "gps_longitude_min": 0.020804280237305316,
            "gps_longitude_max": 0.02238216947709171,
            "gps_upload_delay_mean": 0.017060977317932817,
            "gps_upload_delay_max": 0.03185002039469678,
            "gps_upload_delay_min": 0.046775101934004415,
            "gps_location_provider_<lambda>": 0.002197558741478314,
            "gps_gps_fix_at_count": 0.0936268404297413,
            "gps_location_stability": 0.02281582879308182,
            "gps_provider_reliability": 0.010884180795754957,
            "gps_movement_pattern": 0.021158682351479517,
            "application_at_hour": 0.016722022343739724,
            "application_at_day": 0.021349515444251156,
            "application_at_month": 0.03146584867191233,
            "application_at_year": 0.012546182162605115,
            "application_at_dayofweek": 0.013163612435755787
        }
    }
}
```

### 2. Prediction Endpoint
```
POST /predict
```
Makes loan predictions based on user data and GPS patterns.

Returns:
- Prediction probability
- Risk level classification
- Feature importance for the prediction
- Timestamp of prediction

Example response:
```json
{
    "user_id": "TEST_USER_123",
    "prediction_probability": 0.3358992951492952,
    "risk_level": "low",
    "timestamp": "2024-11-14T03:15:15.436091",
    "feature_importance": {
        "gps_gps_fix_at_count": 0.0936268404297413,
        "age": 0.07845927568012889,
        "cash_incoming_30days": 0.05372791037410548,
        "gps_upload_delay_min": 0.046775101934004415,
        "gps_altitude_std": 0.03655208647321864,
        "gps_altitude_max": 0.034740745612455944,
        "gps_latitude_std": 0.03283744489963296,
        "gps_upload_delay_max": 0.03185002039469678,
        "application_at_month": 0.03146584867191233,
        "gps_accuracy_mean": 0.03051923915093847,
        "gps_altitude_mean": 0.0302417114297488,
        "gps_bearing_std": 0.030027230120541103,
        "gps_longitude_std": 0.026433520926256313,
        "gps_latitude_max": 0.02520461800373176,
        "gps_longitude_mean": 0.02459680823442007,
        "gps_accuracy_min": 0.023097029459706205,
        "gps_location_stability": 0.02281582879308182,
        "hour_of_day": 0.022550254255733572,
        "gps_longitude_max": 0.02238216947709171,
        "gps_accuracy_max": 0.021766553134933093,
        "gps_accuracy_std": 0.02136972544200025,
        "application_at_day": 0.021349515444251156,
        "gps_movement_pattern": 0.021158682351479517,
        "gps_longitude_min": 0.020804280237305316,
        "gps_bearing_mean": 0.02009782176490872,
        "gps_latitude_min": 0.019516507110117952,
        "gps_latitude_mean": 0.01890067855422683,
        "gps_upload_delay_mean": 0.017060977317932817,
        "application_at_hour": 0.016722022343739724,
        "age_group": 0.015370736303683706,
        "gps_bearing_max": 0.015220475868341441,
        "application_at_dayofweek": 0.013163612435755787,
        "application_at_year": 0.012546182162605115,
        "income_quartile": 0.012086098190554688,
        "day_of_week": 0.01186071677083524,
        "gps_provider_reliability": 0.010884180795754957,
        "gps_altitude_min": 0.007737723430411209,
        "gps_bearing_min": 0.002282267278537579,
        "gps_location_provider_<lambda>": 0.002197558741478314
    }
}
```

### 3. Model Information Endpoint
```
GET /model/info
```
Returns comprehensive model information:
- Model version and type
- Feature list
- Training data summary
- Performance metrics
- Feature importance rankings
- Model parameters

Example response:
```json
{
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
    "last_updated": "2024-11-14T02:00:44.218370",
    "performance_metrics": {
        "classification_report": {
            "0": {
                "precision": 0.6304347826086957,
                "recall": 0.725,
                "f1-score": 0.6744186046511628,
                "support": 40.0
            },
            "1": {
                "precision": 0.6764705882352942,
                "recall": 0.575,
                "f1-score": 0.6216216216216216,
                "support": 40.0
            },
            "accuracy": 0.65,
            "macro avg": {
                "precision": 0.6534526854219949,
                "recall": 0.6499999999999999,
                "f1-score": 0.6480201131363922,
                "support": 80.0
            },
            "weighted avg": {
                "precision": 0.653452685421995,
                "recall": 0.65,
                "f1-score": 0.6480201131363922,
                "support": 80.0
            }
        },
        "confusion_matrix": [
            [
                29,
                11
            ],
            [
                17,
                23
            ]
        ],
        "roc_auc_score": 0.7287500000000001,
        "cross_val_scores": {
            "mean": 0.7451171875,
            "std": 0.09472152853892297,
            "scores": [
                0.7890625,
                0.7119140625,
                0.595703125,
                0.744140625,
                0.884765625
            ]
        }
    },
    "feature_importance": {
        "gps_gps_fix_at_count": 0.0936268404297413,
        "age": 0.07845927568012889,
        "cash_incoming_30days": 0.05372791037410548,
        "gps_upload_delay_min": 0.046775101934004415,
        "gps_altitude_std": 0.03655208647321864,
        "gps_altitude_max": 0.034740745612455944,
        "gps_latitude_std": 0.03283744489963296,
        "gps_upload_delay_max": 0.03185002039469678,
        "application_at_month": 0.03146584867191233,
        "gps_accuracy_mean": 0.03051923915093847,
        "gps_altitude_mean": 0.0302417114297488,
        "gps_bearing_std": 0.030027230120541103,
        "gps_longitude_std": 0.026433520926256313,
        "gps_latitude_max": 0.02520461800373176,
        "gps_longitude_mean": 0.02459680823442007,
        "gps_accuracy_min": 0.023097029459706205,
        "gps_location_stability": 0.02281582879308182,
        "hour_of_day": 0.022550254255733572,
        "gps_longitude_max": 0.02238216947709171,
        "gps_accuracy_max": 0.021766553134933093,
        "gps_accuracy_std": 0.02136972544200025,
        "application_at_day": 0.021349515444251156,
        "gps_movement_pattern": 0.021158682351479517,
        "gps_longitude_min": 0.020804280237305316,
        "gps_bearing_mean": 0.02009782176490872,
        "gps_latitude_min": 0.019516507110117952,
        "gps_latitude_mean": 0.01890067855422683,
        "gps_upload_delay_mean": 0.017060977317932817,
        "application_at_hour": 0.016722022343739724,
        "age_group": 0.015370736303683706,
        "gps_bearing_max": 0.015220475868341441,
        "application_at_dayofweek": 0.013163612435755787,
        "application_at_year": 0.012546182162605115,
        "income_quartile": 0.012086098190554688,
        "day_of_week": 0.01186071677083524,
        "gps_provider_reliability": 0.010884180795754957,
        "gps_altitude_min": 0.007737723430411209,
        "gps_bearing_min": 0.002282267278537579,
        "gps_location_provider_<lambda>": 0.002197558741478314
    },
    "description": "Loan prediction model using Random Forest algorithm",
    "training_data_summary": {
        "total_samples": 0,
        "target_distribution": {}
    },
    "model_parameters": {
        "bootstrap": true,
        "ccp_alpha": 0.0,
        "class_weight": null,
        "criterion": "gini",
        "max_depth": 10,
        "max_features": "sqrt",
        "max_leaf_nodes": null,
        "max_samples": null,
        "min_impurity_decrease": 0.0,
        "min_samples_leaf": 4,
        "min_samples_split": 10,
        "min_weight_fraction_leaf": 0.0,
        "monotonic_cst": null,
        "n_estimators": 50,
        "n_jobs": null,
        "oob_score": false,
        "random_state": 42,
        "verbose": 0,
        "warm_start": false
    }
}
```

## API Testing

The project includes comprehensive API tests:
```bash
cd src/test
python test_api.py
```

This test suite covers:
- Endpoint availability
- Request validation
- Response format validation
- Error handling
- Edge cases
- Load testing

### Example API Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
            "user_id": "TEST_USER_123",
            "age": 175,  
            "cash_incoming_30days": 5000.0,  
            "application_timestamp": "2024-11-14T10:30:00",  
            "gps_data": [
                {
                    "accuracy": 15.0,
                    "altitude": 920.0,
                    "bearing": 45.0,
                    "latitude": 12.9716,
                    "longitude": 77.5946,
                    "location_provider": "gps",
                    "gps_fix_at": "2024-11-14T10:25:00"  
                },
                {
                    "accuracy": 12.0,
                    "altitude": 922.0,
                    "bearing": 48.0,
                    "latitude": 12.9718,
                    "longitude": 77.5948,
                    "location_provider": "gps",
                    "gps_fix_at": "2024-11-14T10:27:00"  
                },
                {
                    "accuracy": 10.0,
                    "altitude": 921.0,
                    "bearing": 46.0,
                    "latitude": 12.9720,
                    "longitude": 77.5950,
                    "location_provider": "gps",
                    "gps_fix_at": "2024-11-14T10:30:00" 
                }
            ]
        }'
```

## Future Work

Planned improvements:
1. Enhanced feature engineering using more GPS patterns
2. Integration of additional data sources
3. Model performance optimization
4. A/B testing framework
5. Enhanced API documentation
6. Dashboard for model monitoring
7. Batch prediction capabilities
8. Extended test coverage


