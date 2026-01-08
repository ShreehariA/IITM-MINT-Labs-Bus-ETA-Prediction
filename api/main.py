"""
Main API Orchestrator
Coordinates data fetching, preprocessing, prediction, and caching
Production-ready with comprehensive error handling and logging
"""

import asyncio
import aiohttp
import json
import logging
import time
from datetime import datetime
from typing import Dict, List

# Import API modules (now local)
from feature_extractor import LiveDataPreprocessor, validate_gps_data
from predictor import EnsembleModel
from stops import STOPS, ROUTES, get_upcoming_stops
from logger import log_prediction, log_error, log_fallback
from prediction_utils import simple_eta_fallback, default_eta_fallback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api_predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ETAPredictionAPI:
    """
    Main API orchestrator for real-time ETA predictions.
    Fetches GPS data, preprocesses, predicts, and caches results.
    """
    
    def __init__(self):
        """Initialize API components"""
        logger.info("🚀 Initializing ETA Prediction API...")
        
        # API endpoints
        self.vehicle_data_url = "https://traveliitmapi.mydigipin.in/api/vehicle-data"
        self.active_routes_url = "https://traveliitmapi.mydigipin.in/api/vehicle-route/active"
        
        # Initialize preprocessor and model
        self.preprocessor = LiveDataPreprocessor()
        self.model = EnsembleModel()
        
        # Cache for latest predictions
        self.latest_predictions = {}
        self.last_update = None
        
        logger.info("✅ API initialized successfully!")
    
    async def fetch_vehicle_data(self) -> List[Dict]:
        """Fetch current vehicle positions from API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.vehicle_data_url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"📡 Fetched data for {len(data)} vehicles")
                        return data
                    else:
                        logger.error(f"❌ API returned status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"❌ Failed to fetch vehicle data: {e}")
            return []
    
    async def fetch_active_routes(self) -> Dict[str, str]:
        """Fetch active route mappings"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.active_routes_url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {item['vehicleNo']: item['routeCode'] for item in data}
                    else:
                        return {}
        except Exception as e:
            logger.error(f"❌ Failed to fetch routes: {e}")
            return {}
    
    def clean_vehicle_data(self, vehicles: List[Dict]) -> List[Dict]:
        """Clean and validate vehicle data"""
        cleaned = []
        current_time = datetime.utcnow()
        
        for vehicle in vehicles:
            try:
                # Parse timestamp
                timestamp = datetime.fromisoformat(
                    vehicle['timestamp'].replace('Z', '+00:00')
                )
                age_seconds = (current_time - timestamp.replace(tzinfo=None)).total_seconds()
                
                # Filter stale data (>5 minutes old)
                if age_seconds > 300:
                    continue
                
                # Validate coordinates (IIT Madras bounds)
                if not (12.98 <= vehicle['latitude'] <= 13.01 and
                       80.22 <= vehicle['longitude'] <= 80.25):
                    continue
                
                cleaned.append(vehicle)
                
            except Exception as e:
                logger.debug(f"Error processing vehicle: {e}")
                continue
        
        logger.info(f"✅ Cleaned: {len(cleaned)}/{len(vehicles)} vehicles valid")
        return cleaned
    
    async def generate_predictions(self) -> Dict[int, Dict]:
        """
        Generate predictions for all stops.
        
        Returns:
            Dict mapping stop_id to prediction info
        """
        logger.info("\n" + "="*60)
        logger.info(f"Generating predictions - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        # Fetch data
        vehicles = await self.fetch_vehicle_data()
        routes = await self.fetch_active_routes()
        
        # Clean data
        vehicles = self.clean_vehicle_data(vehicles)
        
        if not vehicles:
            logger.warning("⚠️  No valid vehicle data available")
            return self.latest_predictions  # Return cached predictions
        
        # Generate predictions
        predictions = {}
        
        for vehicle in vehicles:
            vehicle_no = vehicle['vehicleno']
            route_code = routes.get(vehicle_no)
            
            if not route_code or route_code not in ROUTES:
                logger.debug(f"No valid route for vehicle {vehicle_no}")
                continue
            
            # Get upcoming stops based on position and route
            upcoming_stops = get_upcoming_stops(
                (vehicle['latitude'], vehicle['longitude']),
                route_code
            )
            
            for stop_id in upcoming_stops:
                try:
                    start_time = time.time()
                    
                    # Validate GPS data first
                    try:
                        validate_gps_data(vehicle)
                    except ValueError as ve:
                        logger.debug(f"GPS validation failed for {vehicle_no}: {ve}")
                        # Use fallback prediction
                        result = simple_eta_fallback(
                            vehicle.get('latitude', 0),
                            vehicle.get('longitude', 0),
                            stop_id
                        )
                        log_fallback(str(ve), vehicle, 'simple_eta')
                        
                        if stop_id not in predictions or result['eta_seconds'] < predictions[stop_id]['eta_seconds']:
                            predictions[stop_id] = {
                                'stop_id': stop_id,
                                'stop_name': STOPS[stop_id]['name'],
                                'eta_seconds': round(result['eta_seconds'], 1),
                                'eta_minutes': round(result['eta_minutes'], 1),
                                'vehicle_no': vehicle_no,
                                'route': route_code,
                                'route_label': ROUTES[route_code]['label'],
                                'confidence': result['confidence'],
                                'method': result['method'],
                                'last_updated': datetime.utcnow().isoformat() + 'Z'
                            }
                        continue
                    
                    # Convert GPS to graph
                    graph = self.preprocessor.gps_to_graph(vehicle, stop_id, route_code)
                    
                    # Predict ETA with bounds and confidence
                    result = self.model.predict_with_bounds(graph)
                    
                    processing_time_ms = (time.time() - start_time) * 1000
                    
                    # Log prediction
                    log_prediction(vehicle, result, result['method'], processing_time_ms)
                    
                    # Keep minimum ETA for each stop
                    if stop_id not in predictions or result['eta_seconds'] < predictions[stop_id]['eta_seconds']:
                        predictions[stop_id] = {
                            'stop_id': stop_id,
                            'stop_name': STOPS[stop_id]['name'],
                            'eta_seconds': round(result['eta_seconds'], 1),
                            'eta_minutes': round(result['eta_minutes'], 1),
                            'vehicle_no': vehicle_no,
                            'route': route_code,
                            'route_label': ROUTES[route_code]['label'],
                            'confidence': result['confidence'],
                            'method': result.get('method', 'ml_model'),
                            'was_capped': result.get('was_capped', False),
                            'last_updated': datetime.utcnow().isoformat() + 'Z'
                        }
                
                except Exception as e:
                    logger.error(f"❌ Prediction error for stop {stop_id}: {e}")
                    log_error(e, f"Prediction for stop {stop_id}", vehicle)
                    
                    # Try fallback
                    try:
                        result = simple_eta_fallback(
                            vehicle.get('latitude', 0),
                            vehicle.get('longitude', 0),
                            stop_id
                        )
                        log_fallback(str(e), vehicle, 'error_fallback')
                        
                        if stop_id not in predictions:
                            predictions[stop_id] = {
                                'stop_id': stop_id,
                                'stop_name': STOPS[stop_id]['name'],
                                'eta_seconds': round(result['eta_seconds'], 1),
                                'eta_minutes': round(result['eta_minutes'], 1),
                                'vehicle_no': vehicle_no,
                                'route': route_code,
                                'route_label': ROUTES[route_code]['label'],
                                'confidence': result['confidence'],
                                'method': 'fallback_after_error',
                                'last_updated': datetime.utcnow().isoformat() + 'Z'
                            }
                    except:
                        # Even fallback failed - skip this prediction
                        continue
        
        logger.info(f"✅ Generated predictions for {len(predictions)} stops")
        
        # Log summary
        if predictions:
            etas = [p['eta_minutes'] for p in predictions.values()]
            logger.info(f"📊 ETA range: {min(etas):.1f}-{max(etas):.1f} min")
            
            # Log top 5 closest buses
            sorted_preds = sorted(
                predictions.values(),
                key=lambda x: x['eta_seconds']
            )[:5]
            
            logger.info("🚌 Closest buses:")
            for pred in sorted_preds:
                logger.info(
                    f"   {pred['stop_name']}: {pred['eta_minutes']:.1f} min "
                    f"({pred['vehicle_no']} on {pred['route_label']})"
                )
        
        # Update cache
        self.latest_predictions = predictions
        self.last_update = datetime.utcnow()
        
        # Save to file for API consumption
        self.save_predictions(predictions)
        
        return predictions
    
    def save_predictions(self, predictions: Dict):
        """Save predictions to JSON file for API consumption"""
        try:
            with open('latest_predictions.json', 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info("💾 Saved predictions to latest_predictions.json")
        except Exception as e:
            logger.error(f"❌ Failed to save predictions: {e}")
    
    def get_prediction(self, stop_id: int) -> Dict:
        """Get prediction for a specific stop"""
        return self.latest_predictions.get(stop_id, {
            'stop_id': stop_id,
            'stop_name': STOPS.get(stop_id, {}).get('name', 'Unknown'),
            'eta_seconds': None,
            'eta_minutes': None,
            'message': 'No buses currently heading to this stop',
            'last_updated': datetime.utcnow().isoformat() + 'Z'
        })
    
    async def run_once(self):
        """Run prediction pipeline once"""
        try:
            predictions = await self.generate_predictions()
            return predictions
        except Exception as e:
            logger.error(f"❌ Error in prediction pipeline: {e}")
            return {}
    
    async def run_continuous(self, interval_seconds: int = 60):
        """
        Run prediction loop continuously.
        
        Args:
            interval_seconds: Update interval (default 60s)
        """
        logger.info(f"🔄 Starting continuous prediction (interval: {interval_seconds}s)")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                await self.generate_predictions()
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"❌ Error in prediction loop: {e}")
                await asyncio.sleep(10)  # Wait 10s before retry


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='IIT Madras Bus ETA Prediction API')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update interval in seconds (default: 60)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (for cron)')
    
    args = parser.parse_args()
    
    # Initialize API
    api = ETAPredictionAPI()
    
    if args.once:
        # Run once (for cron)
        logger.info("🧪 Running single prediction...")
        predictions = asyncio.run(api.run_once())
        logger.info(f"✅ Complete! Generated {len(predictions)} predictions")
    else:
        # Run continuously
        asyncio.run(api.run_continuous(interval_seconds=args.interval))
