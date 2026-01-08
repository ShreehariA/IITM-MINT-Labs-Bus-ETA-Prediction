# Production Model Validation

Final sanity check for the ensemble model before production deployment.

## 🎯 Purpose

Tests the production ensemble model on **completely unseen data** (20-24 Nov) to validate:
- Model generalization to new data
- Real-world prediction accuracy
- Production readiness

## 📋 What It Does

1. **Loads unseen test data** from `20to24nov/` folder
2. **Preprocesses** using production pipeline
3. **Generates predictions** using deployed ensemble
4. **Outputs detailed CSV** with predictions vs actuals
5. **Calculates metrics**: MAE, RMSE, ±1/2/5min accuracy

## 🚀 How to Run

### Prerequisites

Ensure you're in the `api/` directory:
```bash
cd api
```

### Run Validation

```bash
python validate.py
```

## 📊 Output

The script generates two files:

### 1. Detailed Results CSV
`validation_results_YYYYMMDD_HHMMSS.csv`

Contains row-by-row predictions:
```csv
actual_eta_seconds,predicted_eta_seconds,actual_eta_minutes,predicted_eta_minutes,error_seconds,error_minutes,absolute_error_seconds,absolute_error_minutes
420,435,7.0,7.25,15,0.25,15,0.25
180,175,3.0,2.92,-5,-0.08,5,0.08
...
```

**Columns:**
- `actual_eta_seconds/minutes` - Ground truth ETA
- `predicted_eta_seconds/minutes` - Model prediction
- `error_seconds/minutes` - Signed error (predicted - actual)
- `absolute_error_seconds/minutes` - Absolute error

### 2. Metrics Summary
`validation_metrics_YYYYMMDD_HHMMSS.txt`

Contains performance summary:
```
================================================================================
PRODUCTION MODEL VALIDATION METRICS
================================================================================

Test Date: 2026-01-05 01:57:00
Test Data: 20-24 Nov (Unseen)
Samples: 5,234

Performance Metrics:
----------------------------------------
MAE:  2.15 minutes
RMSE: 3.42 minutes
Median Error: 1.85 minutes
Max Error: 12.30 minutes

Accuracy Thresholds:
----------------------------------------
±1 min: 69.7%
±2 min: 84.2%
±5 min: 93.2%
```

## 📈 Console Output

```
================================================================================
PRODUCTION MODEL VALIDATION TEST
================================================================================

📅 Test Data: ../20to24nov
📊 Model: Ensemble (GNN + XGBoost + LightGBM)
🎯 Purpose: Final sanity check on unseen data

🔄 Loading production model...
✅ Model loaded successfully

📂 Loading unseen test data...
  Total GPS points: 125,432
  Date range: 2025-11-20 to 2025-11-24

🔄 Preprocessing data...
  Generated examples: 5,234
  Features: 53

🔮 Generating predictions...
Predicting: 100%|████████████████████| 5234/5234 [02:15<00:00, 38.67it/s]

💾 Saved detailed results: validation_results_20260105_015700.csv
💾 Saved metrics summary: validation_metrics_20260105_015700.txt

================================================================================
VALIDATION RESULTS
================================================================================

📊 Test Samples: 5,234

📈 Performance Metrics:
----------------------------------------
  MAE:  2.15 minutes
  RMSE: 3.42 minutes
  Median Error: 1.85 minutes
  Max Error: 12.30 minutes

🎯 Accuracy Thresholds:
----------------------------------------
  ±1 min: 69.7%
  ±2 min: 84.2%
  ±5 min: 93.2%

✅ Validation Status:
----------------------------------------
  ✅ EXCELLENT - Ready for production

================================================================================

✅ Validation complete!

📝 Output files:
  - validation_results_20260105_015700.csv (detailed predictions)
  - validation_metrics_20260105_015700.txt (summary metrics)
```

## ✅ Validation Criteria

| ±5min Accuracy | Status | Action |
|---------------|--------|--------|
| ≥ 90% | ✅ EXCELLENT | Ready for production |
| 85-90% | ✅ GOOD | Acceptable for production |
| 80-85% | ⚠️ FAIR | Consider retraining |
| < 80% | ❌ POOR | Retraining required |

## 🔧 Troubleshooting

### "Graph file not found"
```bash
# Ensure bus_graphs_full.pt exists in parent directory
ls ../bus_graphs_full.pt
```

### "Module not found"
```bash
# Ensure you're in api/ directory
cd api
python validate.py
```

### "No data in 20to24nov/"
```bash
# Check test data directory
ls ../20to24nov/
```

## 📊 Expected Performance

Based on training results, expect:
- **MAE**: ~2-3 minutes
- **RMSE**: ~3-4 minutes
- **±1 min**: ~65-70%
- **±2 min**: ~80-85%
- **±5 min**: ~90-95%

## 🎯 Best Practices

1. **Run before deployment** - Always validate on unseen data
2. **Check all metrics** - Don't rely on single metric
3. **Review CSV** - Inspect individual predictions for patterns
4. **Compare to baseline** - Ensure improvement over naive methods
5. **Document results** - Keep validation reports for audit trail

## 📝 Notes

- Uses **production pipeline** - Same code as deployment
- Tests on **unseen data** - 20-24 Nov (not in training)
- Generates **reproducible results** - Timestamped outputs
- Follows **ML best practices** - Proper train/test separation

## 🚨 Important

This is a **final sanity check**. If validation fails:
1. Review training data quality
2. Check for data drift
3. Consider retraining with more data
4. Investigate error patterns in CSV

**Do not deploy if ±5min accuracy < 85%**
