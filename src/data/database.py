import sys
from pathlib import Path

# Add src directory to PYTHONPATH
sys.path.append("/Users/omvibhandik/Desktop/branch_loan_analysis/branch_loan_analysis/src")

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import logging
from config import Config
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseConnector:
  """
  Class to handle all database operations
  """
  def __init__(self, config: Dict = None):
      """
      Initialize database connector
      Args:
          config (Dict): Database configuration dictionary
      """
      self.config = config or Config.DB_CONFIG
      self.engine = self._create_engine()
      self._test_connection()

  def _create_engine(self):
      """Create SQLAlchemy engine with retries"""
      try:
          connection_string = (
              f"postgresql://{self.config['user']}:{self.config['password']}"
              f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
          )
          return create_engine(connection_string, pool_pre_ping=True)
      except Exception as e:
          logger.error(f"Failed to create database engine: {str(e)}")
          raise

  def _test_connection(self):
      """Test database connection"""
      try:
          with self.engine.connect() as conn:
              conn.execute(text("SELECT 1"))
          logger.info("Database connection successful")
      except Exception as e:
          logger.error(f"Database connection failed: {str(e)}")
          raise

  def get_loan_outcomes(self, start_date: str = None, end_date: str = None, user_ids: List[str] = None) -> pd.DataFrame:
    """
    Fetch loan outcomes data with optional date and user filtering
    
    Args:
        start_date (str): Start date for filtering (YYYY-MM-DD)
        end_date (str): End date for filtering (YYYY-MM-DD)
        user_ids (List[str]): List of user IDs to filter by
    
    Returns:
        pd.DataFrame: Loan outcomes data
    """
    try:
        query = """
        SELECT 
            user_id,
            application_at,
            loan_outcome,
            EXTRACT(DOW FROM application_at) as day_of_week,
            EXTRACT(HOUR FROM application_at) as hour_of_day
        FROM loan_outcomes
        WHERE 1=1
        """
        
        if start_date:
            query += f" AND application_at >= '{start_date}'"
        if end_date:
            query += f" AND application_at <= '{end_date}'"
        if user_ids:
            user_ids_str = "','".join(str(id_) for id_ in user_ids)
            query += f" AND user_id IN ('{user_ids_str}')"
        
        df = pd.read_sql(query, self.engine)
        logger.info(f"Retrieved {len(df)} loan outcome records")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch loan outcomes: {str(e)}")
        raise

  # Add this method to your DatabaseConnector class
  def get_loan_statistics(self) -> Dict[str, Any]:
    """
    Get comprehensive loan statistics
    
    Returns:
        Dict[str, Any]: Dictionary containing loan statistics
    """
    try:
        # Basic loan statistics
        basic_stats_query = """
        SELECT 
            COUNT(*) as total_loans,
            SUM(CASE WHEN loan_outcome = 'yes' THEN 1 ELSE 0 END) as repaid_loans,
            SUM(CASE WHEN loan_outcome = 'no' THEN 1 ELSE 0 END) as defaulted_loans,
            ROUND(AVG(CASE WHEN loan_outcome = 'yes' THEN 1 ELSE 0 END)::numeric * 100, 2) as repayment_rate,
            COUNT(DISTINCT user_id) as unique_borrowers
        FROM loan_outcomes
        """
        
        basic_stats = pd.read_sql(basic_stats_query, self.engine).to_dict('records')[0]
        
        # Monthly trends
        monthly_trends_query = """
        SELECT 
            DATE_TRUNC('month', application_at) as month,
            COUNT(*) as total_loans,
            SUM(CASE WHEN loan_outcome = 'yes' THEN 1 ELSE 0 END) as repaid_loans,
            ROUND(AVG(CASE WHEN loan_outcome = 'yes' THEN 1 ELSE 0 END)::numeric * 100, 2) as repayment_rate
        FROM loan_outcomes
        GROUP BY DATE_TRUNC('month', application_at)
        ORDER BY month
        """
        
        monthly_trends = pd.read_sql(monthly_trends_query, self.engine)
        
        # Day of week analysis
        dow_query = """
        SELECT 
            EXTRACT(DOW FROM application_at) as day_of_week,
            COUNT(*) as total_loans,
            ROUND(AVG(CASE WHEN loan_outcome = 'yes' THEN 1 ELSE 0 END)::numeric * 100, 2) as repayment_rate
        FROM loan_outcomes
        GROUP BY EXTRACT(DOW FROM application_at)
        ORDER BY day_of_week
        """
        
        dow_stats = pd.read_sql(dow_query, self.engine)
        
        # Combine all statistics
        stats = {
            'basic_stats': basic_stats,
            'monthly_trends': monthly_trends.to_dict('records'),
            'day_of_week_stats': dow_stats.to_dict('records')
        }
        
        logger.info("Successfully retrieved loan statistics")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to fetch loan statistics: {str(e)}")
        raise

  def get_gps_fixes(self, user_ids: List[str] = None) -> pd.DataFrame:
      """
      Fetch GPS fixes data with optional user filtering
      
      Args:
          user_ids (List[str]): List of user IDs to filter by
          
      Returns:
          pd.DataFrame: GPS fixes data
      """
      try:
          query = """
          SELECT 
              user_id,
              accuracy,
              altitude,
              bearing,
              gps_fix_at,
              latitude,
              longitude,
              location_provider,
              server_upload_at,
              EXTRACT(EPOCH FROM (server_upload_at - gps_fix_at)) as upload_delay_seconds
          FROM gps_fixes
          """
          
          if user_ids:
              user_ids_str = "','".join(user_ids)
              query += f" WHERE user_id IN ('{user_ids_str}')"
              
          df = pd.read_sql(query, self.engine)
          
          # Convert timestamps
          df['gps_fix_at'] = pd.to_datetime(df['gps_fix_at'])
          df['server_upload_at'] = pd.to_datetime(df['server_upload_at'])
          
          logger.info(f"Retrieved {len(df)} GPS fix records")
          return df
          
      except Exception as e:
          logger.error(f"Failed to fetch GPS fixes: {str(e)}")
          raise

  def get_user_attributes(self, user_ids: List[str] = None) -> pd.DataFrame:
      """
      Fetch user attributes data with optional user filtering
      
      Args:
          user_ids (List[str]): List of user IDs to filter by
          
      Returns:
          pd.DataFrame: User attributes data
      """
      try:
          query = """
          SELECT 
              user_id,
              age,
              cash_incoming_30days,
              CASE 
                  WHEN age < 25 THEN 'young'
                  WHEN age BETWEEN 25 AND 35 THEN 'adult'
                  WHEN age BETWEEN 36 AND 50 THEN 'middle_aged'
                  ELSE 'senior'
              END as age_group,
              NTILE(4) OVER (ORDER BY cash_incoming_30days) as income_quartile
          FROM user_attributes
          """
          
          if user_ids:
              user_ids_str = "','".join(user_ids)
              query += f" WHERE user_id IN ('{user_ids_str}')"
              
          df = pd.read_sql(query, self.engine)
          logger.info(f"Retrieved {len(df)} user attribute records")
          return df
          
      except Exception as e:
          logger.error(f"Failed to fetch user attributes: {str(e)}")
          raise

  def get_user_data(self, user_id: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch all data for a specific user
    
    Args:
        user_id (str): User ID to fetch data for
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing all user data
    """
    try:
        # Convert user_id to string if it's not already
        user_id = str(user_id)
        
        # Fetch all data for the user
        user_attributes = self.get_user_attributes([user_id])
        loan_outcomes = self.get_loan_outcomes(user_ids=[user_id])
        gps_fixes = self.get_gps_fixes([user_id])
        
        return {
            'user_attributes': user_attributes,
            'loan_outcomes': loan_outcomes,
            'gps_fixes': gps_fixes
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch user data: {str(e)}")
        raise

  def get_aggregated_stats(self) -> Dict[str, Any]:
      """
      Get aggregated statistics from the database
      
      Returns:
          Dict[str, Any]: Dictionary containing various statistics
      """
      try:
          stats = {}
          
          # Loan outcome statistics
          loan_stats_query = """
          SELECT 
              COUNT(*) as total_loans,
              SUM(CASE WHEN loan_outcome = 'yes' THEN 1 ELSE 0 END) as repaid_loans,
              AVG(CASE WHEN loan_outcome = 'yes' THEN 1 ELSE 0 END)::float as repayment_rate
          FROM loan_outcomes
          """
          stats['loan_stats'] = pd.read_sql(loan_stats_query, self.engine).to_dict('records')[0]
          
          # User statistics
          user_stats_query = """
          SELECT 
              COUNT(*) as total_users,
              AVG(age) as avg_age,
              AVG(cash_incoming_30days) as avg_cash_incoming
          FROM user_attributes
          """
          stats['user_stats'] = pd.read_sql(user_stats_query, self.engine).to_dict('records')[0]
          
          # GPS statistics
          gps_stats_query = """
          SELECT 
              COUNT(*) as total_gps_fixes,
              COUNT(DISTINCT user_id) as users_with_gps,
              AVG(accuracy) as avg_accuracy
          FROM gps_fixes
          """
          stats['gps_stats'] = pd.read_sql(gps_stats_query, self.engine).to_dict('records')[0]
          
          return stats
          
      except Exception as e:
          logger.error(f"Failed to fetch aggregated stats: {str(e)}")
          raise

  def close(self):
      """Close database connection"""
      try:
          self.engine.dispose()
          logger.info("Database connection closed")
      except Exception as e:
          logger.error(f"Error closing database connection: {str(e)}")
          raise

  def __enter__(self):
      """Context manager enter"""
      return self

  def __exit__(self, exc_type, exc_val, exc_tb):
      """Context manager exit"""
      self.close()