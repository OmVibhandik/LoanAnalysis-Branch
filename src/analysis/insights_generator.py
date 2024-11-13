import sys
from pathlib import Path

# Add src directory to PYTHONPATH
sys.path.append("/Users/omvibhandik/Desktop/branch_loan_analysis/branch_loan_analysis/src")

import pandas as pd
import numpy as np
from data.database import DatabaseConnector
from config import Config
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def generate_insights(combined_data=None):
  """
  Generate insights from loan, user, and GPS data
  """
  try:
      # Initialize database connection
      db = DatabaseConnector(Config.DB_CONFIG)
      
      # Fetch data from all tables
      loan_data = db.get_loan_outcomes()
      gps_data = db.get_gps_fixes()
      user_data = db.get_user_attributes()
      
      # Process loan data
      loan_data['application_at'] = pd.to_datetime(loan_data['application_at'])
      loan_data['day_of_week'] = loan_data['application_at'].dt.day_name()
      loan_data['hour_of_day'] = loan_data['application_at'].dt.hour
      
      # Calculate key metrics
      insights = {
          'Loan Statistics': {
              'Total Applications': len(loan_data),
              'Unique Borrowers': loan_data['user_id'].nunique(),
              'Repayment Rate': f"{(loan_data['loan_outcome'] == 'yes').mean():.2%}",
              'Most Common Application Day': loan_data['day_of_week'].mode().iloc[0],
              'Peak Application Hour': f"{loan_data['hour_of_day'].mode().iloc[0]:02d}:00"
          },
          
          'User Demographics': {
              'Total Users': len(user_data),
              'Age Range': f"{user_data['age'].min():.0f} - {user_data['age'].max():.0f}",
              'Average Age': f"{user_data['age'].mean():.1f}",
              'Median Monthly Income (KES)': f"{user_data['cash_incoming_30days'].median():,.0f}",
              'Average Monthly Income (KES)': f"{user_data['cash_incoming_30days'].mean():,.0f}"
          },
          
          'GPS Activity': {
              'Total GPS Fixes': len(gps_data),
              'Unique Users with GPS': gps_data['user_id'].nunique(),
              'Average Accuracy (meters)': f"{gps_data['accuracy'].mean():.2f}",
              'Most Common Location Provider': gps_data['location_provider'].mode().iloc[0],
              'Geographic Coverage': {
                  'Latitude Range': f"{gps_data['latitude'].min():.4f}° to {gps_data['latitude'].max():.4f}°",
                  'Longitude Range': f"{gps_data['longitude'].min():.4f}° to {gps_data['longitude'].max():.4f}°"
              }
          }
      }
      
      # Generate visualizations
      # Set the style before creating any plots
      sns.set_theme(style="whitegrid")
      
      fig, axes = plt.subplots(2, 2, figsize=(15, 12))
      
      # Plot 1: Age Distribution
      sns.histplot(data=user_data, x='age', bins=30, ax=axes[0,0])
      axes[0,0].set_title('Age Distribution of Borrowers')
      axes[0,0].set_xlabel('Age (years)')
      
      # Plot 2: Loan Outcomes by Day
      loan_day_counts = loan_data.groupby('day_of_week')['loan_outcome'].value_counts(normalize=True).unstack()
      loan_day_counts.plot(kind='bar', stacked=True, ax=axes[0,1])
      axes[0,1].set_title('Loan Outcomes by Day of Week')
      axes[0,1].set_xlabel('Day of Week')
      axes[0,1].set_ylabel('Proportion')
      
      # Plot 3: Income Distribution
      sns.histplot(data=user_data, x='cash_incoming_30days', bins=30, ax=axes[1,0])
      axes[1,0].set_title('Monthly Income Distribution')
      axes[1,0].set_xlabel('Monthly Income (KES)')
      
      # Plot 4: GPS Fixes by Hour
      gps_data['hour'] = pd.to_datetime(gps_data['gps_fix_at']).dt.hour
      sns.countplot(data=gps_data, x='hour', ax=axes[1,1])
      axes[1,1].set_title('GPS Fixes by Hour of Day')
      axes[1,1].set_xlabel('Hour of Day')
      
      plt.tight_layout()
      plt.savefig('src/analysis/loan_analysis_insights.png')
      plt.close()
      
      # Print insights
      print("\nLoan Analysis Insights")
      print("=" * 50)
      
      for category, metrics in insights.items():
          print(f"\n{category}:")
          print("-" * 30)
          for key, value in metrics.items():
              if isinstance(value, dict):
                  print(f"\n{key}:")
                  for subkey, subvalue in value.items():
                      print(f"  - {subkey}: {subvalue}")
              else:
                  print(f"- {key}: {value}")
      
      return insights
      
  except Exception as e:
      print(f"Error generating insights: {str(e)}")
      raise
      
  finally:
      db.close()

if __name__ == "__main__":
  insights = generate_insights()