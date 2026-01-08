"""
Production Logging Infrastructure
Structured logging for predictions, errors, and monitoring
"""

import logging
import json
from datetime import datetime
from pathlib import Path

# Create logs directory
Path('logs').mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/predictions.log'),
        logging.FileHandler('logs/errors.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('eta_api')
error_logger = logging.getLogger('eta_errors')

def log_prediction(bus_position, result, method='ml_model', processing_time_ms=None):
    """
    Log prediction details for monitoring and analysis.
    
    Args:
        bus_position: GPS data dict
        result: Prediction result dict
        method: Prediction method used
        processing_time_ms: Processing time in milliseconds
    """
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'latitude': bus_position.get('latitude'),
        'longitude': bus_position.get('longitude'),
        'speed': bus_position.get('speed'),
        'predicted_eta_seconds': result.get('eta_seconds'),
        'predicted_eta_minutes': result.get('eta_minutes'),
        'confidence': result.get('confidence', 0),
        'method': method,
        'was_capped': result.get('was_capped', False),
        'processing_time_ms': processing_time_ms
    }
    
    logger.info(json.dumps(log_data))
    
    # Alert on anomalies
    if result.get('eta_seconds', 0) >= 1800:
        logger.warning(f"⚠️  High ETA prediction: {result['eta_minutes']:.1f} minutes")
    
    if result.get('confidence', 1.0) < 0.6:
        logger.warning(f"⚠️  Low confidence prediction: {result['confidence']:.2f}")
    
    if result.get('was_capped', False):
        logger.warning(f"⚠️  Prediction was capped: original may have been out of bounds")

def log_error(error, context, bus_position=None):
    """
    Log error with context for debugging.
    
    Args:
        error: Exception object
        context: Description of what was being done
        bus_position: Optional GPS data
    """
    error_data = {
        'timestamp': datetime.now().isoformat(),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'bus_position': bus_position
    }
    
    error_logger.error(json.dumps(error_data))

def log_fallback(reason, bus_position, fallback_method):
    """
    Log when fallback method is used.
    
    Args:
        reason: Why fallback was triggered
        bus_position: GPS data
        fallback_method: Which fallback was used
    """
    fallback_data = {
        'timestamp': datetime.now().isoformat(),
        'reason': reason,
        'fallback_method': fallback_method,
        'latitude': bus_position.get('latitude'),
        'longitude': bus_position.get('longitude')
    }
    
    logger.warning(f"🔄 Fallback triggered: {json.dumps(fallback_data)}")

def log_validation_failure(field, value, reason):
    """
    Log input validation failures.
    
    Args:
        field: Field that failed validation
        value: Invalid value
        reason: Why it failed
    """
    validation_data = {
        'timestamp': datetime.now().isoformat(),
        'field': field,
        'value': value,
        'reason': reason
    }
    
    logger.warning(f"❌ Validation failed: {json.dumps(validation_data)}")
