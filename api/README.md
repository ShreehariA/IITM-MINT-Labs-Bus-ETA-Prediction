# IIT Madras Bus ETA Prediction API

Production-ready API for real-time bus ETA predictions using graph-based ensemble models.

## 📁 Structure

```
api/
├── main.py                    # Main orchestrator
├── feature_extractor.py       # GPS → Graph conversion
├── predictor.py               # Ensemble model loader
├── stops.py                   # Stop metadata and routes
│
├── ensemble_model.pkl         # ← Meta-learner + scalers (4KB)
├── tgat_best_random.pt        # ← GNN weights (12MB)
├── xgboost_model.json         # ← XGBoost model (9.4MB)
├── lightgbm_model.txt         # ← LightGBM model (1.3MB)
│
├── gnn_model.py               # GNN architecture
├── utils.py                   # Preprocessing utilities
├── graph.py                   # Graph construction
│
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container config
└── README.md                  # This file
```

**Total size: ~23MB** - Perfect for microservices!

## 🚀 Quick Start

### 1. Test Single Prediction

```bash
cd api
python main.py --once
```

This will:
- Fetch live GPS data
- Convert to graphs
- Generate predictions
- Save to `latest_predictions.json`

### 2. Run Continuous Service

```bash
python main.py --interval 60
```

Updates every 60 seconds.

### 3. Setup Cron (Production)

Add to crontab:
```bash
* * * * * cd /path/to/mint/api && python main.py --once >> cron.log 2>&1
```

### 4. Docker Deployment (Recommended for Production)

**Build image:**
```bash
cd api
docker build -t bus-eta-api .
```

**Run once (for cron/K8s CronJob):**
```bash
docker run --rm bus-eta-api
```

**Run continuously:**
```bash
docker run -d --name bus-eta \
  -v $(pwd)/predictions:/app/predictions \
  bus-eta-api python main.py --interval 60
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  bus-eta:
    build: .
    command: python main.py --interval 60
    volumes:
      - ./predictions:/app/predictions
    restart: unless-stopped
```

**Kubernetes CronJob:**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: bus-eta-predictor
spec:
  schedule: "* * * * *"  # Every minute
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: predictor
            image: bus-eta-api:latest
            command: ["python", "main.py", "--once"]
          restartPolicy: OnFailure
```

## 📊 How It Works

### Data Flow

```
Live GPS API
    ↓
feature_extractor.py (GPS → Graph)
    ↓
predictor.py (Ensemble Prediction)
    ↓
latest_predictions.json
```

### Components

#### 1. **feature_extractor.py**
- Uses existing `utils.py` approach
- Converts GPS point to PyTorch Geometric graph
- Single node graph for real-time (no historical window needed)

**Example:**
```python
from feature_extractor import LiveDataPreprocessor

preprocessor = LiveDataPreprocessor()

gps_data = {
    'latitude': 12.9882,
    'longitude': 80.2237,
    'speed': 15.5,
    'timestamp': datetime.now()
}

graph = preprocessor.gps_to_graph(gps_data, target_stop_id=18)
```

#### 2. **predictor.py**
- Loads saved ensemble from `build_ensemble.py`
- GNN + XGBoost + LightGBM with Ridge meta-learner
- **Accuracy**: 69.7% at ±1min, 84.2% at ±2min, 93.2% at ±5min

**Example:**
```python
from predictor import EnsembleModel

model = EnsembleModel()
eta_seconds = model.predict(graph)
```

#### 3. **stops.py**
- 29 bus stops with actual IIT Madras coordinates
- 8 route definitions (HOSTEL_MAIN, MAIN_HOSTEL, etc.)
- Helper functions for route logic

#### 4. **main.py**
- Orchestrates entire pipeline
- Fetches from `https://traveliitmapi.mydigipin.in/api/vehicle-data`
- Cleans data, generates predictions, caches results
- Saves to `latest_predictions.json`

## 📋 Output Format

`latest_predictions.json`:
```json
{
  "18": {
    "stop_id": 18,
    "stop_name": "Main Gate",
    "eta_seconds": 420.5,
    "eta_minutes": 7.0,
    "vehicle_no": "EV08",
    "route": "HOSTEL_MAIN",
    "route_label": "Hostel → Main Gate",
    "confidence": "high",
    "last_updated": "2026-01-02T07:00:00Z"
  }
}
```

## 🔧 Configuration

### Required Files (all included in api/ folder)

- `tgat_best_random.pt` - GNN model
- `xgboost_model.json` - XGBoost model
- `lightgbm_model.txt` - LightGBM model
- `ensemble_model.pkl` - Meta-learner + scalers
- `utils.py` - Preprocessing utilities
- `graph.py` - Graph construction
- `gnn_model.py` - GNN architecture

### Dependencies

```bash
pip install torch torch-geometric xgboost lightgbm scikit-learn aiohttp pandas numpy
```

## 📊 Performance

### Model Accuracy (Test Set)
- **±1 min**: 69.7%
- **±2 min**: 84.2%
- **±5 min**: 93.2%

### API Performance
- **Prediction latency**: ~100-200ms per vehicle
- **Update frequency**: 60 seconds (configurable)
- **Concurrent vehicles**: Handles 10+ buses simultaneously

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
# Make sure you're in the api/ directory
cd api
python main.py --once
```

### "No valid vehicle data"
- Check API connectivity
- Verify GPS data is recent (<5 min old)
- Check IIT Madras campus bounds (12.98-13.01, 80.22-80.25)

### "Model file not found"
```bash
# All models are now in api/ folder
ls *.pt *.json *.txt *.pkl
```

## 📝 Logs

- **Console**: Real-time progress
- **api_predictions.log**: Detailed logs with timestamps
- **latest_predictions.json**: Latest predictions (updated every minute)

## 🎯 Next Steps

1. **Test**: Run `python main.py --once` to verify setup
2. **Monitor**: Check `api_predictions.log` for errors
3. **Deploy**: Set up cron job for production
4. **Integrate**: Use `latest_predictions.json` in frontend

## 🔄 Data Collection for Retraining

While running, the API can collect data for future model improvements:

```python
# Add to main.py
def save_for_training(vehicle, stop_id, predicted_eta):
    with open('training_data.jsonl', 'a') as f:
        f.write(json.dumps({
            'vehicle': vehicle,
            'stop_id': stop_id,
            'predicted_eta': predicted_eta,
            'timestamp': datetime.utcnow().isoformat()
        }) + '\n')
```

After 1-2 weeks, retrain models with collected data for improved accuracy!

## 📞 Support

For issues or questions, check:
- Logs: `api_predictions.log`
- Test output: `python main.py --once`
- Model status: Check if all `.pt`, `.json`, `.txt`, `.pkl` files exist
