"""
Real-Time Bus ETA Prediction Service
Fetches live data every minute and generates predictions using existing ensemble model
"""
import asyncio
import aiohttp
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from typing import Dict, List
import logging
from live_feature_converter import LiveFeatureConverter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealTimeETAPredictor:
    """
    Real-time ETA prediction service.
    Fetches live vehicle data and generates predictions every minute.
    """
    
    def __init__(self, stop_metadata: Dict):
        """
        Initialize predictor with models and metadata.
        
        Args:
            stop_metadata: Dict mapping stop_id to {name, lat, lon, routes}
        """
        logger.info("Initializing Real-Time ETA Predictor...")
        
        # Load trained models
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model('production_xgboost.json')
        
        self.lgb_model = lgb.Booster(model_file='production_lightgbm.txt')
        
        with open('production_ensemble.pkl', 'rb') as f:
            ensemble_data = pickle.load(f)
            self.meta_model = ensemble_data['meta_model']
            self.scaler = ensemble_data['scaler']
        
        # Initialize feature converter
        self.converter = LiveFeatureConverter(stop_metadata)
        self.stops = stop_metadata
        
        # API endpoints
        self.vehicle_data_url = "https://traveliitmapi.mydigipin.in/api/vehicle-data"
        self.active_routes_url = "https://traveliitmapi.mydigipin.in/api/vehicle-route/active"
        
        logger.info("✅ Models loaded successfully")
    
    async def fetch_vehicle_data(self) -> List[Dict]:
        """Fetch current vehicle positions from API."""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.vehicle_data_url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Fetched data for {len(data)} vehicles")
                    return data
                else:
                    logger.error(f"Failed to fetch vehicle data: {response.status}")
                    return []
    
    async def fetch_active_routes(self) -> Dict[str, str]:
        """Fetch active route mappings."""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.active_routes_url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Convert to dict: {vehicleNo: routeCode}
                    return {item['vehicleNo']: item['routeCode'] for item in data}
                else:
                    logger.error(f"Failed to fetch routes: {response.status}")
                    return {}
    
    def clean_vehicle_data(self, vehicles: List[Dict]) -> List[Dict]:
        """
        Clean and validate vehicle data.
        Remove stale or invalid data.
        """
        cleaned = []
        current_time = datetime.utcnow()
        
        for vehicle in vehicles:
            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(
                    vehicle['timestamp'].replace('Z', '+00:00')
                )
                age_seconds = (current_time - timestamp.replace(tzinfo=None)).total_seconds()
                
                # Filter stale data (>5 minutes old)
                if age_seconds > 300:
                    logger.debug(f"Skipping stale data for {vehicle['vehicleno']}")
                    continue
                
                # Validate coordinates (IIT Madras campus bounds)
                if not (12.98 <= vehicle['latitude'] <= 13.00 and
                       80.22 <= vehicle['longitude'] <= 80.24):
                    logger.debug(f"Invalid coordinates for {vehicle['vehicleno']}")
                    continue
                
                cleaned.append(vehicle)
                
            except Exception as e:
                logger.warning(f"Error processing vehicle {vehicle.get('vehicleno')}: {e}")
                continue
        
        logger.info(f"Cleaned data: {len(cleaned)}/{len(vehicles)} vehicles valid")
        return cleaned
    
    def predict_eta(self, vehicle: Dict, stop_id: int) -> float:
        """
        Predict ETA for a vehicle to reach a stop.
        
        Args:
            vehicle: Vehicle data dict
            stop_id: Target stop ID
            
        Returns:
            ETA in seconds
        """
        try:
            # Convert to model features
            features = self.converter.convert(vehicle, stop_id)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get base model predictions
            xgb_pred = self.xgb_model.predict(
                xgb.DMatrix(features_scaled)
            )
            lgb_pred = self.lgb_model.predict(features_scaled)
            
            # Meta-learner prediction
            meta_input = np.column_stack([xgb_pred, lgb_pred])
            eta_seconds = self.meta_model.predict(meta_input)[0]
            
            # Ensure non-negative
            return max(0, eta_seconds)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return float('inf')  # Return infinity on error
    
    def get_upcoming_stops(self, vehicle: Dict, route_code: str) -> List[int]:
        """
        Get list of upcoming stops for a vehicle based on its route.
        
        For now, returns all stops on the route.
        TODO: Filter based on current position and direction.
        """
        # This is a simplified version
        # You'll need to implement proper route-based filtering
        
        # For now, return all stops that serve this route
        upcoming = []
        for stop_id, stop_data in self.stops.items():
            if route_code in stop_data.get('routes', []):
                upcoming.append(stop_id)
        
        return upcoming
    
    async def generate_predictions(self) -> Dict[int, Dict]:
        """
        Generate predictions for all stops.
        
        Returns:
            Dict mapping stop_id to {eta_seconds, vehicle_no, confidence}
        """
        # Fetch data
        vehicles = await self.fetch_vehicle_data()
        routes = await self.fetch_active_routes()
        
        # Clean data
        vehicles = self.clean_vehicle_data(vehicles)
        
        if not vehicles:
            logger.warning("No valid vehicle data available")
            return {}
        
        # Generate predictions
        predictions = {}
        
        for vehicle in vehicles:
            vehicle_no = vehicle['vehicleno']
            route_code = routes.get(vehicle_no)
            
            if not route_code:
                logger.debug(f"No route for vehicle {vehicle_no}")
                continue
            
            # Get upcoming stops
            upcoming_stops = self.get_upcoming_stops(vehicle, route_code)
            
            for stop_id in upcoming_stops:
                # Predict ETA
                eta = self.predict_eta(vehicle, stop_id)
                
                # Keep minimum ETA for each stop
                if stop_id not in predictions or eta < predictions[stop_id]['eta_seconds']:
                    predictions[stop_id] = {
                        'stop_id': stop_id,
                        'stop_name': self.stops[stop_id]['name'],
                        'eta_seconds': eta,
                        'eta_minutes': eta / 60,
                        'vehicle_no': vehicle_no,
                        'route': route_code,
                        'confidence': 'high',  # Based on 93.2% accuracy
                        'last_updated': datetime.utcnow().isoformat()
                    }
        
        logger.info(f"Generated predictions for {len(predictions)} stops")
        return predictions
    
    async def run_continuous(self, interval_seconds: int = 60):
        """
        Run prediction loop continuously.
        
        Args:
            interval_seconds: Update interval (default 60s)
        """
        logger.info(f"Starting continuous prediction (interval: {interval_seconds}s)")
        
        while True:
            try:
                # Generate predictions
                predictions = await self.generate_predictions()
                
                # Log summary
                if predictions:
                    min_eta = min(p['eta_minutes'] for p in predictions.values())
                    max_eta = max(p['eta_minutes'] for p in predictions.values())
                    logger.info(
                        f"Predictions: {len(predictions)} stops, "
                        f"ETA range: {min_eta:.1f}-{max_eta:.1f} min"
                    )
                
                # TODO: Store predictions in cache/database
                # await cache.set('latest_predictions', predictions)
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(10)  # Wait 10s before retry


# Example usage
if __name__ == "__main__":
    # Define stop metadata
    # TODO: Load from file or database
    stop_metadata = {
        1: {
            'name': 'Main Gate',
            'latitude': 12.9916,
            'longitude': 80.2336,
            'routes': ['MAIN_HOSTEL', 'ED_RP']
        },
        2: {
            'name': 'Academic Complex',
            'latitude': 12.9900,
            'longitude': 80.2320,
            'routes': ['MAIN_HOSTEL', 'HOSTEL_MAIN']
        },
        # Add all 29 stops...
    }
    
    # Initialize predictor
    predictor = RealTimeETAPredictor(stop_metadata)
    
    # Run continuous prediction
    asyncio.run(predictor.run_continuous(interval_seconds=60))
