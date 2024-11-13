import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
  accuracy_score, precision_score, recall_score, f1_score,
  roc_auc_score, confusion_matrix, precision_recall_curve,
  average_precision_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path
import logging
from config import Config

logger = logging.getLogger(__name__)

class ModelMetrics:
  """Class for calculating and visualizing model metrics"""
  
  def __init__(self, config: Dict = None):
      """
      Initialize metrics calculator
      
      Args:
          config (Dict): Configuration dictionary
      """
      self.config = config or Config.METRICS_CONFIG
      self.metrics_history = []
      self.current_metrics = {}

  def calculate_metrics(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      y_prob: np.ndarray) -> Dict[str, float]:
      """
      Calculate classification metrics
      
      Args:
          y_true (np.ndarray): True labels
          y_pred (np.ndarray): Predicted labels
          y_prob (np.ndarray): Prediction probabilities
          
      Returns:
          Dict[str, float]: Dictionary of metrics
      """
      try:
          metrics = {
              'accuracy': accuracy_score(y_true, y_pred),
              'precision': precision_score(y_true, y_pred),
              'recall': recall_score(y_true, y_pred),
              'f1': f1_score(y_true, y_pred),
              'roc_auc': roc_auc_score(y_true, y_prob),
              'average_precision': average_precision_score(y_true, y_prob)
          }
          
          # Calculate confusion matrix
          cm = confusion_matrix(y_true, y_pred)
          metrics['confusion_matrix'] = cm.tolist()
          
          # Calculate additional metrics
          metrics['specificity'] = cm[0,0] / (cm[0,0] + cm[0,1])
          metrics['npv'] = cm[0,0] / (cm[0,0] + cm[1,0])  # Negative Predictive Value
          
          # Store current metrics
          self.current_metrics = metrics
          self.metrics_history.append({
              'timestamp': datetime.now().isoformat(),
              'metrics': metrics
          })
          
          return metrics
          
      except Exception as e:
          logger.error(f"Error calculating metrics: {str(e)}")
          raise

  def plot_roc_curve(self, 
                    y_true: np.ndarray, 
                    y_prob: np.ndarray,
                    save_path: Optional[str] = None) -> None:
      """
      Plot ROC curve
      
      Args:
          y_true (np.ndarray): True labels
          y_prob (np.ndarray): Prediction probabilities
          save_path (str): Path to save the plot
      """
      try:
          fpr, tpr, _ = roc_curve(y_true, y_prob)
          roc_auc = roc_auc_score(y_true, y_prob)
          
          plt.figure(figsize=(10, 6))
          plt.plot(fpr, tpr, color='darkorange', lw=2, 
                  label=f'ROC curve (AUC = {roc_auc:.2f})')
          plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
          plt.xlim([0.0, 1.0])
          plt.ylim([0.0, 1.05])
          plt.xlabel('False Positive Rate')
          plt.ylabel('True Positive Rate')
          plt.title('Receiver Operating Characteristic (ROC) Curve')
          plt.legend(loc="lower right")
          
          if save_path:
              plt.savefig(save_path)
              logger.info(f"ROC curve saved to {save_path}")
          
          plt.close()
          
      except Exception as e:
          logger.error(f"Error plotting ROC curve: {str(e)}")
          raise

  def plot_precision_recall_curve(self, 
                                y_true: np.ndarray, 
                                y_prob: np.ndarray,
                                save_path: Optional[str] = None) -> None:
      """
      Plot Precision-Recall curve
      
      Args:
          y_true (np.ndarray): True labels
          y_prob (np.ndarray): Prediction probabilities
          save_path (str): Path to save the plot
      """
      try:
          precision, recall, _ = precision_recall_curve(y_true, y_prob)
          avg_precision = average_precision_score(y_true, y_prob)
          
          plt.figure(figsize=(10, 6))
          plt.plot(recall, precision, color='darkorange', lw=2,
                  label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
          plt.xlabel('Recall')
          plt.ylabel('Precision')
          plt.title('Precision-Recall Curve')
          plt.legend(loc="lower left")
          
          if save_path:
              plt.savefig(save_path)
              logger.info(f"Precision-Recall curve saved to {save_path}")
          
          plt.close()
          
      except Exception as e:
          logger.error(f"Error plotting Precision-Recall curve: {str(e)}")
          raise

  def plot_confusion_matrix(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          save_path: Optional[str] = None) -> None:
      """
      Plot confusion matrix
      
      Args:
          y_true (np.ndarray): True labels
          y_pred (np.ndarray): Predicted labels
          save_path (str): Path to save the plot
      """
      try:
          cm = confusion_matrix(y_true, y_pred)
          plt.figure(figsize=(8, 6))
          sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
          plt.xlabel('Predicted')
          plt.ylabel('True')
          plt.title('Confusion Matrix')
          
          if save_path:
              plt.savefig(save_path)
              logger.info(f"Confusion matrix plot saved to {save_path}")
          
          plt.close()
          
      except Exception as e:
          logger.error(f"Error plotting confusion matrix: {str(e)}")
          raise

  def calculate_threshold_metrics(self, 
                               y_true: np.ndarray, 
                               y_prob: np.ndarray,
                               thresholds: Optional[List[float]] = None) -> pd.DataFrame:
      """
      Calculate metrics at different probability thresholds
      
      Args:
          y_true (np.ndarray): True labels
          y_prob (np.ndarray): Prediction probabilities
          thresholds (List[float]): List of thresholds to evaluate
          
      Returns:
          pd.DataFrame: Metrics at different thresholds
      """
      try:
          if thresholds is None:
              thresholds = np.arange(0.1, 1.0, 0.1)
          
          results = []
          for threshold in thresholds:
              y_pred = (y_prob >= threshold).astype(int)
              metrics = {
                  'threshold': threshold,
                  'accuracy': accuracy_score(y_true, y_pred),
                  'precision': precision_score(y_true, y_pred),
                  'recall': recall_score(y_true, y_pred),
                  'f1': f1_score(y_true, y_pred)
              }
              results.append(metrics)
          
          return pd.DataFrame(results)
          
      except Exception as e:
          logger.error(f"Error calculating threshold metrics: {str(e)}")
          raise

  def plot_threshold_metrics(self, 
                           threshold_metrics: pd.DataFrame,
                           save_path: Optional[str] = None) -> None:
      """
      Plot metrics across different thresholds
      
      Args:
          threshold_metrics (pd.DataFrame): Metrics at different thresholds
          save_path (str): Path to save the plot
      """
      try:
          plt.figure(figsize=(12, 6))
          metrics = ['accuracy', 'precision', 'recall', 'f1']
          for metric in metrics:
              plt.plot(threshold_metrics['threshold'], 
                      threshold_metrics[metric], 
                      label=metric)
          
          plt.xlabel('Threshold')
          plt.ylabel('Score')
          plt.title('Metrics vs Threshold')
          plt.legend()
          plt.grid(True)
          
          if save_path:
              plt.savefig(save_path)
              logger.info(f"Threshold metrics plot saved to {save_path}")
          
          plt.close()
          
      except Exception as e:
          logger.error(f"Error plotting threshold metrics: {str(e)}")
          raise

  def save_metrics(self, 
                  file_path: str,
                  include_history: bool = False) -> None:
      """
      Save metrics to file
      
      Args:
          file_path (str): Path to save metrics
          include_history (bool): Whether to include metrics history
      """
      try:
          data = {
              'current_metrics': self.current_metrics,
              'timestamp': datetime.now().isoformat()
          }
          
          if include_history:
              data['metrics_history'] = self.metrics_history
          
          with open(file_path, 'w') as f:
              json.dump(data, f, indent=4)
          
          logger.info(f"Metrics saved to {file_path}")
          
      except Exception as e:
          logger.error(f"Error saving metrics: {str(e)}")
          raise

  def load_metrics(self, file_path: str) -> Dict[str, Any]:
      """
      Load metrics from file
      
      Args:
          file_path (str): Path to load metrics from
          
      Returns:
          Dict[str, Any]: Loaded metrics
      """
      try:
          with open(file_path, 'r') as f:
              data = json.load(f)
          
          self.current_metrics = data['current_metrics']
          if 'metrics_history' in data:
              self.metrics_history = data['metrics_history']
          
          logger.info(f"Metrics loaded from {file_path}")
          return data
          
      except Exception as e:
          logger.error(f"Error loading metrics: {str(e)}")
          raise

  def get_metrics_summary(self) -> Dict[str, Any]:
      """
      Get summary of current metrics
      
      Returns:
          Dict[str, Any]: Metrics summary
      """
      return {
          'current_metrics': self.current_metrics,
          'metrics_history_length': len(self.metrics_history),
          'last_updated': datetime.now().isoformat()
      }