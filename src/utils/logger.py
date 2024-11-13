import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from typing import Dict, Any, Optional
from config import Config

class CustomLogger:
  """Custom logger with both file and console handlers"""
  
  def __init__(self, 
               name: str,
               log_dir: str = None,
               config: Dict[str, Any] = None):
      """
      Initialize logger with custom configuration
      
      Args:
          name (str): Logger name
          log_dir (str): Directory for log files
          config (Dict[str, Any]): Logger configuration
      """
      self.config = config or Config.LOGGING_CONFIG
      self.log_dir = Path(log_dir or self.config['log_dir'])
      self.log_dir.mkdir(parents=True, exist_ok=True)
      
      # Create logger
      self.logger = logging.getLogger(name)
      self.logger.setLevel(self.config['level'])

      # Configure logging handler and formatter if not already set up
      if not self.logger.hasHandlers():
        handler = logging.StreamHandler()  # or use FileHandler for file logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
      
      # Remove existing handlers
      self.logger.handlers = []
      
      # Add handlers
      self._add_console_handler()
      self._add_file_handler()
      self._add_error_file_handler()
      
      self.logger.info(f"Logger initialized: {name}")

  def _add_console_handler(self):
      """Add console handler with custom formatter"""
      console_handler = logging.StreamHandler(sys.stdout)
      console_handler.setLevel(self.config['console_level'])
      console_handler.setFormatter(self._get_formatter())
      self.logger.addHandler(console_handler)

  def _add_file_handler(self):
      """Add rotating file handler for all logs"""
      file_handler = RotatingFileHandler(
          filename=self.log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log",
          maxBytes=self.config['max_bytes'],
          backupCount=self.config['backup_count']
      )
      file_handler.setLevel(self.config['file_level'])
      file_handler.setFormatter(self._get_formatter())
      self.logger.addHandler(file_handler)

  def _add_error_file_handler(self):
      """Add separate handler for error logs"""
      error_handler = TimedRotatingFileHandler(
          filename=self.log_dir / "error.log",
          when='midnight',
          interval=1,
          backupCount=self.config['error_backup_count']
      )
      error_handler.setLevel(logging.ERROR)
      error_handler.setFormatter(self._get_formatter())
      self.logger.addHandler(error_handler)

  def _get_formatter(self) -> logging.Formatter:
      """Get custom formatter for logs"""
      return logging.Formatter(
          fmt=self.config['format'],
          datefmt=self.config['date_format']
      )
  
  def get_logger(self) -> logging.Logger:
    """Get the configured logger."""
    return self.logger

class JsonLogger(CustomLogger):
  """Logger that outputs in JSON format"""
  
  def _get_formatter(self) -> logging.Formatter:
      """Get JSON formatter"""
      return JsonFormatter(
          fmt=self.config['format'],
          datefmt=self.config['date_format']
      )

class JsonFormatter(logging.Formatter):
  """Custom JSON formatter"""
  
  def format(self, record: logging.LogRecord) -> str:
      """Format log record as JSON"""
      log_data = {
          'timestamp': self.formatTime(record, self.datefmt),
          'level': record.levelname,
          'logger': record.name,
          'message': record.getMessage(),
          'module': record.module,
          'function': record.funcName,
          'line': record.lineno
      }
      
      # Add exception info if present
      if record.exc_info:
          log_data['exception'] = self.formatException(record.exc_info)
          
      # Add extra fields if present
      if hasattr(record, 'extra'):
          log_data.update(record.extra)
          
      return json.dumps(log_data)

class ModelLogger(JsonLogger):
  """Specialized logger for model operations"""
  
  def __init__(self, name: str = 'model_logger', log_dir: str = None):
      super().__init__(name, log_dir)
      
  def log_prediction(self, 
                    user_id: str, 
                    prediction: float, 
                    features: Dict[str, float],
                    metadata: Optional[Dict[str, Any]] = None):
      """Log model prediction"""
      extra = {
          'user_id': user_id,
          'prediction': prediction,
          'features': features,
          'metadata': metadata or {}
      }
      self.logger.info("Model prediction", extra=extra)

  def log_training(self, 
                  metrics: Dict[str, float], 
                  parameters: Dict[str, Any],
                  metadata: Optional[Dict[str, Any]] = None):
      """Log model training"""
      extra = {
          'metrics': metrics,
          'parameters': parameters,
          'metadata': metadata or {}
      }
      self.logger.info("Model training completed", extra=extra)

  def log_feature_importance(self, feature_importance: Dict[str, float]):
      """Log feature importance scores"""
      self.logger.info("Feature importance scores", 
                      extra={'feature_importance': feature_importance})

class APILogger(JsonLogger):
  """Specialized logger for API operations"""
  
  def __init__(self, name: str = 'api_logger', log_dir: str = None):
      super().__init__(name, log_dir)
      
  def log_request(self, 
                  endpoint: str, 
                  method: str, 
                  params: Dict[str, Any],
                  user_id: Optional[str] = None):
      """Log API request"""
      extra = {
          'endpoint': endpoint,
          'method': method,
          'params': params,
          'user_id': user_id
      }
      self.logger.info("API request received", extra=extra)

  def log_response(self, 
                  endpoint: str, 
                  status_code: int, 
                  response_time: float,
                  response_data: Optional[Dict[str, Any]] = None):
      """Log API response"""
      extra = {
          'endpoint': endpoint,
          'status_code': status_code,
          'response_time': response_time,
          'response_data': response_data
      }
      self.logger.info("API response sent", extra=extra)

  def log_error(self, 
                endpoint: str, 
                error_message: str,
                stack_trace: Optional[str] = None):
      """Log API error"""
      extra = {
          'endpoint': endpoint,
          'error_message': error_message,
          'stack_trace': stack_trace
      }
      self.logger.error("API error occurred", extra=extra)


def get_logger(name: str, 
             logger_type: str = 'default', 
             log_dir: Optional[str] = None) -> logging.Logger:
  """
  Factory function to get appropriate logger
  
  Args:
      name (str): Logger name
      logger_type (str): Type of logger ('default', 'json', 'model', 'api')
      log_dir (str): Directory for log files
      
  Returns:
      logging.Logger: Configured logger
  """
  logger_classes = {
      'default': CustomLogger,
      'json': JsonLogger,
      'model': ModelLogger,
      'api': APILogger
  }
  
  logger_class = logger_classes.get(logger_type, CustomLogger)
  return logger_class(name, log_dir).get_logger()

# Example usage
if __name__ == "__main__":
  # Get different types of loggers
  default_logger = get_logger("default_logger")
  json_logger = get_logger("json_logger", "json")
  model_logger = get_logger("model_logger", "model")
  api_logger = get_logger("api_logger", "api")
  
  # Test logging
  default_logger.info("This is a default log message")
  json_logger.info("This is a JSON formatted log message")
  
  # Test model logger
  model_logger.log_prediction(
      user_id="user123",
      prediction=0.85,
      features={"feature1": 0.5, "feature2": 0.3}
  )
  
  # Test API logger
  api_logger.log_request(
      endpoint="/predict",
      method="POST",
      params={"user_id": "user123"},
      user_id="user123"
  )




