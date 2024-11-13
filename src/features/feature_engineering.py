# src/features/feature_engineering.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from geopy.distance import geodesic
from timezonefinder import TimezoneFinder
from pytz import timezone
import holidays
from config import Config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class for feature engineering operations"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize feature engineer
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config or Config.FEATURE_CONFIG
        self.scalers = {}
        self.tf = TimezoneFinder()
        self.us_holidays = holidays.US()
        
    def create_features(self,
                       gps_df: pd.DataFrame,
                       user_df: pd.DataFrame,
                       loan_history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from GPS and user data
        
        Args:
            gps_df (pd.DataFrame): GPS data
            user_df (pd.DataFrame): User attributes
            loan_history_df (pd.DataFrame): Historical loan data
            
        Returns:
            pd.DataFrame: Engineered features
        """
        try:
            # Create different feature groups
            location_features = self._create_location_features(gps_df)
            temporal_features = self._create_temporal_features(gps_df)
            movement_features = self._create_movement_features(gps_df)
            user_features = self._create_user_features(user_df)
            history_features = self._create_history_features(loan_history_df)
            
            # Combine all features
            features = pd.concat([
                location_features,
                temporal_features,
                movement_features,
                user_features,
                history_features
            ], axis=1)
            
            # Handle missing values
            features = self._handle_missing_values(features)
            
            # Scale features
            features = self._scale_features(features)
            
            logger.info(f"Created {features.shape[1]} features")
            return features
            
        except Exception as e:
            logger.error(f"Error in feature creation: {str(e)}")
            raise

    def _create_location_features(self, gps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create location-based features
        
        Args:
            gps_df (pd.DataFrame): GPS data
            
        Returns:
            pd.DataFrame: Location features
        """
        try:
            features = pd.DataFrame()
            
            # Calculate basic location statistics
            for coord in ['latitude', 'longitude']:
                features[f'{coord}_mean'] = gps_df[coord].mean()
                features[f'{coord}_std'] = gps_df[coord].std()
                features[f'{coord}_min'] = gps_df[coord].min()
                features[f'{coord}_max'] = gps_df[coord].max()
            
            # Calculate area covered
            features['area_covered'] = self._calculate_area_covered(gps_df)
            
            # Calculate stay points
            stay_points = self._identify_stay_points(gps_df)
            features['num_stay_points'] = len(stay_points)
            features['avg_stay_duration'] = np.mean([sp['duration'] for sp in stay_points])
            
            # Calculate home/work location probability
            features.update(self._identify_significant_locations(gps_df))
            
            return features
            
        except Exception as e:
            logger.error(f"Error in location feature creation: {str(e)}")
            raise

    def _create_temporal_features(self, gps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            gps_df (pd.DataFrame): GPS data
            
        Returns:
            pd.DataFrame: Temporal features
        """
        try:
            features = pd.DataFrame()
            
            # Convert timestamp to datetime if needed
            if 'server_upload_at' in gps_df.columns:
                timestamps = pd.to_datetime(gps_df['server_upload_at'])
                
                # Time of day features
                features['avg_hour'] = timestamps.dt.hour.mean()
                features['std_hour'] = timestamps.dt.hour.std()
                
                # Day of week features
                features['weekday_ratio'] = (timestamps.dt.dayofweek < 5).mean()
                
                # Activity time features
                features['business_hours_ratio'] = (
                    (timestamps.dt.hour >= 9) & 
                    (timestamps.dt.hour <= 17)
                ).mean()
                
                # Holiday features
                features['holiday_ratio'] = timestamps.apply(
                    lambda x: x.date() in self.us_holidays
                ).mean()
                
                # Timezone features
                features.update(self._create_timezone_features(gps_df))
            
            return features
            
        except Exception as e:
            logger.error(f"Error in temporal feature creation: {str(e)}")
            raise

    def _create_movement_features(self, gps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create movement-based features
        
        Args:
            gps_df (pd.DataFrame): GPS data
            
        Returns:
            pd.DataFrame: Movement features
        """
        try:
            features = pd.DataFrame()
            
            # Calculate distances between consecutive points
            distances = self._calculate_distances(gps_df)
            
            # Basic movement statistics
            features['total_distance'] = np.sum(distances)
            features['avg_speed'] = np.mean(distances / self.config['time_interval'])
            features['max_speed'] = np.max(distances / self.config['time_interval'])
            features['movement_std'] = np.std(distances)
            
            # Calculate movement patterns
            features.update(self._analyze_movement_patterns(distances))
            
            # Calculate stop duration statistics
            features.update(self._analyze_stops(distances))
            
            return features
            
        except Exception as e:
            logger.error(f"Error in movement feature creation: {str(e)}")
            raise

    def _create_user_features(self, user_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user attribute features
        
        Args:
            user_df (pd.DataFrame): User attributes
            
        Returns:
            pd.DataFrame: User features
        """
        try:
            features = pd.DataFrame()
            
            # Basic user attributes
            categorical_columns = self.config['categorical_columns']
            numerical_columns = self.config['numerical_columns']
            
            # Encode categorical variables
            for col in categorical_columns:
                if col in user_df.columns:
                    encoded = pd.get_dummies(user_df[col], prefix=col)
                    features = pd.concat([features, encoded], axis=1)
            
            # Process numerical variables
            for col in numerical_columns:
                if col in user_df.columns:
                    features[col] = user_df[col]
            
            # Create derived features
            features.update(self._create_derived_user_features(user_df))
            
            return features
            
        except Exception as e:
            logger.error(f"Error in user feature creation: {str(e)}")
            raise

    def _create_history_features(self, loan_history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from loan history
        
        Args:
            loan_history_df (pd.DataFrame): Historical loan data
            
        Returns:
            pd.DataFrame: History features
        """
        try:
            features = pd.DataFrame()
            
            if not loan_history_df.empty:
                # Calculate loan history statistics
                features['num_previous_loans'] = len(loan_history_df)
                features['avg_loan_amount'] = loan_history_df['loan_amount'].mean()
                features['max_loan_amount'] = loan_history_df['loan_amount'].max()
                
                # Calculate repayment behavior
                features['repayment_ratio'] = (
                    loan_history_df['status'] == 'paid'
                ).mean()
                
                # Calculate time-based features
                features.update(self._analyze_loan_timing(loan_history_df))
            else:
                # Default values for new users
                features['num_previous_loans'] = 0
                features['avg_loan_amount'] = 0
                features['max_loan_amount'] = 0
                features['repayment_ratio'] = 1
            
            return features
            
        except Exception as e:
            logger.error(f"Error in history feature creation: {str(e)}")
            raise

    def _calculate_area_covered(self, gps_df: pd.DataFrame) -> float:
        """Calculate approximate area covered by GPS points"""
        try:
            if len(gps_df) < 3:
                return 0
            
            from scipy.spatial import ConvexHull
            points = gps_df[['latitude', 'longitude']].values
            hull = ConvexHull(points)
            return hull.area
            
        except Exception:
            return 0

    def _identify_stay_points(self, gps_df: pd.DataFrame) -> List[Dict]:
        """Identify locations where user stayed for extended periods"""
        stay_points = []
        
        try:
            if len(gps_df) < 2:
                return stay_points
            
            current_cluster = []
            for idx in range(len(gps_df) - 1):
                point1 = (gps_df.iloc[idx]['latitude'], gps_df.iloc[idx]['longitude'])
                point2 = (gps_df.iloc[idx + 1]['latitude'], gps_df.iloc[idx + 1]['longitude'])
                
                distance = geodesic(point1, point2).meters
                
                if distance < self.config['stay_point_radius']:
                    current_cluster.append(idx)
                else:
                    if len(current_cluster) > self.config['min_stay_points']:
                        stay_points.append({
                            'center': (
                                np.mean(gps_df.iloc[current_cluster]['latitude']),
                                np.mean(gps_df.iloc[current_cluster]['longitude'])
                            ),
                            'duration': len(current_cluster) * self.config['time_interval']
                        })
                    current_cluster = []
            
            return stay_points
            
        except Exception as e:
            logger.error(f"Error identifying stay points: {str(e)}")
            return stay_points

    def _identify_significant_locations(self, gps_df: pd.DataFrame) -> Dict[str, float]:
        """Identify probable home and work locations"""
        features = {}
        
        try:
            if 'server_upload_at' not in gps_df.columns:
                return features
            
            timestamps = pd.to_datetime(gps_df['server_upload_at'])
            
            # Night time points (home)
            night_points = gps_df[
                (timestamps.dt.hour >= 22) | 
                (timestamps.dt.hour <= 5)
            ]
            
            # Work time points
            work_points = gps_df[
                (timestamps.dt.hour >= 9) & 
                (timestamps.dt.hour <= 17) &
                (timestamps.dt.dayofweek < 5)
            ]
            
            # Calculate probabilities
            if len(night_points) > 0:
                features['home_location_probability'] = self._calculate_location_probability(
                    night_points
                )
            
            if len(work_points) > 0:
                features['work_location_probability'] = self._calculate_location_probability(
                    work_points
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Error identifying significant locations: {str(e)}")
            return features

    def _calculate_location_probability(self, points_df: pd.DataFrame) -> float:
        """Calculate probability of a significant location"""
        try:
            from sklearn.cluster import DBSCAN
            
            coords = points_df[['latitude', 'longitude']].values
            clustering = DBSCAN(
                eps=0.001,  # Approximately 111 meters
                min_samples=3
            ).fit(coords)
            
            if len(set(clustering.labels_)) <= 1:
                return 0
            
            # Calculate the ratio of points in the largest cluster
            largest_cluster = max(set(clustering.labels_), key=list(clustering.labels_).count)
            return (clustering.labels_ == largest_cluster).mean()
            
        except Exception:
            return 0

    def _create_timezone_features(self, gps_df: pd.DataFrame) -> Dict[str, float]:
        """Create timezone-based features"""
        features = {}
        
        try:
            # Get timezone for each point
            timezones = []
            for _, row in gps_df.iterrows():
                tz_str = self.tf.timezone_at(lat=row['latitude'], lng=row['longitude'])
                if tz_str:
                    timezones.append(timezone(tz_str))
            
            if timezones:
                # Calculate timezone changes
                features['timezone_changes'] = len(set(timezones))
                
                # Calculate time spent in each timezone
                timezone_counts = pd.Series(timezones).value_counts()
                features['primary_timezone_ratio'] = timezone_counts.iloc[0] / len(timezones)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating timezone features: {str(e)}")
            return features

    def _calculate_distances(self, gps_df: pd.DataFrame) -> np.ndarray:
        """Calculate distances between consecutive points"""
        try:
            distances = []
            for i in range(len(gps_df) - 1):
                point1 = (gps_df.iloc[i]['latitude'], gps_df.iloc[i]['longitude'])
                point2 = (gps_df.iloc[i + 1]['latitude'], gps_df.iloc[i + 1]['longitude'])
                distances.append(geodesic(point1, point2).meters)
            
            return np.array(distances)
            
        except Exception as e:
            logger.error(f"Error calculating distances: {str(e)}")
            return np.array([])

    def _analyze_movement_patterns(self, distances: np.ndarray) -> Dict[str, float]:
        """Analyze movement patterns from distances"""
        features = {}
        
        try:
            if len(distances) > 0:
                # Movement states
                movement_threshold = self.config['movement_threshold']
                moving_periods = distances > movement_threshold
                
                features['movement_ratio'] = np.mean(moving_periods)
                
                # Movement patterns
                features['rapid_movements'] = np.mean(
                    distances > movement_threshold * 2
                )
                
                # Calculate acceleration
                if len(distances) > 1:
                    acceleration = np.diff(distances)
                    features['avg_acceleration'] = np.mean(acceleration)
                    features['max_acceleration'] = np.max(np.abs(acceleration))
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing movement patterns: {str(e)}")
            return features

    def _analyze_stops(self, distances: np.ndarray) -> Dict[str, float]:
        """Analyze stop durations and patterns"""
        features = {}
        
        try:
            if len(distances) > 0:
                # Identify stops
                stops = distances < self.config['movement_threshold']
                
                if np.any(stops):
                    # Calculate stop durations
                    from itertools import groupby
                    stop_durations = [
                        len(list(g)) for k, g in groupby(stops) if k
                    ]
                                        # Stop duration features
                    features['avg_stop_duration'] = np.mean(stop_durations)
                    features['max_stop_duration'] = np.max(stop_durations)
                    features['num_stops'] = len(stop_durations)
                    features['stop_ratio'] = np.mean(stops)
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing stops: {str(e)}")
            return features

    def _create_derived_user_features(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Create derived features from user attributes"""
        features = {}
        
        try:
            if 'age' in user_df.columns:
                features['age_group'] = pd.qcut(
                    user_df['age'], 
                    q=5, 
                    labels=['very_young', 'young', 'middle', 'senior', 'elderly']
                ).astype(str)
            
            if 'income' in user_df.columns and 'expenses' in user_df.columns:
                features['disposable_income'] = user_df['income'] - user_df['expenses']
                features['expense_ratio'] = user_df['expenses'] / user_df['income']
            
            if 'employment_length' in user_df.columns:
                features['employment_stability'] = np.minimum(
                    user_df['employment_length'] / 5, 1
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating derived user features: {str(e)}")
            return features

    def _analyze_loan_timing(self, loan_history_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze timing patterns in loan history"""
        features = {}
        
        try:
            if 'application_date' in loan_history_df.columns:
                dates = pd.to_datetime(loan_history_df['application_date'])
                
                # Time between loans
                if len(dates) > 1:
                    time_between_loans = np.diff(sorted(dates))
                    features['avg_time_between_loans'] = np.mean(
                        time_between_loans
                    ).total_seconds() / (24 * 3600)  # Convert to days
                    
                    features['std_time_between_loans'] = np.std(
                        time_between_loans
                    ).total_seconds() / (24 * 3600)
                
                # Seasonality
                features['weekend_applications'] = (dates.dt.dayofweek >= 5).mean()
                features['holiday_applications'] = dates.apply(
                    lambda x: x.date() in self.us_holidays
                ).mean()
            
            if 'repayment_date' in loan_history_df.columns:
                # Calculate repayment timing
                loan_history_df['days_to_repay'] = (
                    pd.to_datetime(loan_history_df['repayment_date']) - 
                    pd.to_datetime(loan_history_df['due_date'])
                ).dt.days
                
                features['avg_days_to_repay'] = loan_history_df['days_to_repay'].mean()
                features['late_payments_ratio'] = (
                    loan_history_df['days_to_repay'] > 0
                ).mean()
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing loan timing: {str(e)}")
            return features

    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        try:
            # Fill missing values based on strategy
            for col in features.columns:
                if features[col].dtype in ['int64', 'float64']:
                    # Numerical columns
                    if features[col].isnull().sum() > 0:
                        if col in self.config['zero_fill_columns']:
                            features[col] = features[col].fillna(0)
                        else:
                            features[col] = features[col].fillna(features[col].mean())
                else:
                    # Categorical columns
                    features[col] = features[col].fillna('unknown')
            
            return features
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise

    def _scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        try:
            scaled_features = features.copy()
            
            # Scale numerical columns
            numerical_columns = features.select_dtypes(
                include=['int64', 'float64']
            ).columns
            
            for col in numerical_columns:
                if col not in self.scalers:
                    if col in self.config['minmax_scale_columns']:
                        self.scalers[col] = MinMaxScaler()
                    else:
                        self.scalers[col] = StandardScaler()
                
                scaled_features[col] = self.scalers[col].fit_transform(
                    features[col].values.reshape(-1, 1)
                ).ravel()
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise

    def save_scalers(self, path: str) -> None:
        """Save feature scalers"""
        try:
            import joblib
            joblib.dump(self.scalers, path)
            logger.info(f"Scalers saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving scalers: {str(e)}")
            raise

    def load_scalers(self, path: str) -> None:
        """Load feature scalers"""
        try:
            import joblib
            self.scalers = joblib.load(path)
            logger.info(f"Scalers loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading scalers: {str(e)}")
            raise

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return list(self.scalers.keys())

    def get_feature_metadata(self) -> Dict[str, Dict]:
        """Get metadata for all features"""
        metadata = {}
        
        for feature in self.get_feature_names():
            metadata[feature] = {
                'scaler_type': type(self.scalers.get(feature)).__name__,
                'categorical': feature not in self.scalers,
                'description': self.config.get('feature_descriptions', {}).get(
                    feature, 'No description available'
                )
            }
        
        return metadata