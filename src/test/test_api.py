import requests
import json
from datetime import datetime

# API endpoint
API_URL = "http://localhost:8000/predict"

# Test data
request_data = {
    "user_id": "TEST_USER_123",
    "age": 35,  # Integer age for compatibility with numeric grouping
    "cash_incoming_30days": 50000.0,  # Float for income, as expected by the API function
    "application_timestamp": "2024-11-14T10:30:00",  # ISO 8601 formatted timestamp
    "gps_data": [
        {
            "accuracy": 15.0,
            "altitude": 920.0,
            "bearing": 45.0,
            "latitude": 12.9716,
            "longitude": 77.5946,
            "location_provider": "gps",
            "gps_fix_at": "2024-11-14T10:25:00"  # ISO 8601 formatted timestamp
        },
        {
            "accuracy": 12.0,
            "altitude": 922.0,
            "bearing": 48.0,
            "latitude": 12.9718,
            "longitude": 77.5948,
            "location_provider": "gps",
            "gps_fix_at": "2024-11-14T10:27:00"  # ISO 8601 formatted timestamp
        },
        {
            "accuracy": 10.0,
            "altitude": 921.0,
            "bearing": 46.0,
            "latitude": 12.9720,
            "longitude": 77.5950,
            "location_provider": "gps",
            "gps_fix_at": "2024-11-14T10:30:00"  # ISO 8601 formatted timestamp
        }
    ]
}

# Sending request to the API
response = requests.post(
  API_URL,
  json=request_data,
  headers={"Content-Type": "application/json"}
)

# Print the JSON response in a readable format
print(json.dumps(response.json(), indent=2))
