"""
Quick Sanity Check - Test All Production Fixes
Verifies that all new code works before running full validation
"""

import sys
import os

print("="*80)
print("PRODUCTION FIXES - SANITY CHECK")
print("="*80)

# Test 1: Imports
print("\n1️⃣ Testing imports...")
try:
    from feature_extractor import validate_gps_data, LiveDataPreprocessor
    from predictor import EnsembleModel
    from prediction_utils import apply_bounds, calculate_confidence, simple_eta_fallback
    from logger import log_prediction, log_error, log_validation_failure
    from config import GPS_VALIDATION, ETA_BOUNDS, CONFIDENCE
    print("   ✅ All imports successful!")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: GPS Validation
print("\n2️⃣ Testing GPS validation...")
try:
    # Should PASS
    validate_gps_data({
        'latitude': 13.0,
        'longitude': 80.23,
        'speed': 20
    })
    print("   ✅ Valid GPS accepted")
    
    # Should FAIL - (0,0)
    try:
        validate_gps_data({
            'latitude': 0,
            'longitude': 0,
            'speed': 0
        })
        print("   ❌ Should have rejected (0,0)!")
        sys.exit(1)
    except ValueError:
        print("   ✅ (0,0) coordinates rejected")
    
    # Should FAIL - Out of Chennai
    try:
        validate_gps_data({
            'latitude': 50.0,
            'longitude': 50.0,
            'speed': 20
        })
        print("   ❌ Should have rejected out-of-bounds!")
        sys.exit(1)
    except ValueError:
        print("   ✅ Out-of-bounds rejected")
        
except Exception as e:
    print(f"   ❌ Validation test failed: {e}")
    sys.exit(1)

# Test 3: Output Bounds
print("\n3️⃣ Testing output bounds...")
try:
    # Test capping high value
    bounded, capped = apply_bounds(3000)  # 50 minutes
    assert bounded == 1800, f"Should cap at 1800s, got {bounded}"
    assert capped == True, "Should be marked as capped"
    print(f"   ✅ High value capped: 3000s → {bounded}s")
    
    # Test capping low value
    bounded, capped = apply_bounds(10)  # 10 seconds
    assert bounded == 30, f"Should cap at 30s, got {bounded}"
    assert capped == True, "Should be marked as capped"
    print(f"   ✅ Low value capped: 10s → {bounded}s")
    
    # Test normal value
    bounded, capped = apply_bounds(300)  # 5 minutes
    assert bounded == 300, f"Should not cap, got {bounded}"
    assert capped == False, "Should not be marked as capped"
    print(f"   ✅ Normal value unchanged: {bounded}s")
    
except Exception as e:
    print(f"   ❌ Bounds test failed: {e}")
    sys.exit(1)

# Test 4: Confidence Scoring
print("\n4️⃣ Testing confidence scoring...")
try:
    # Close distance = high confidence
    conf = calculate_confidence(400, 120)  # 400m, 2min
    assert conf >= 0.90, f"Should be high confidence, got {conf}"
    print(f"   ✅ Close distance (400m): {conf:.2f} confidence")
    
    # Far distance = lower confidence
    conf = calculate_confidence(3000, 600)  # 3km, 10min
    assert conf < 0.80, f"Should be lower confidence, got {conf}"
    print(f"   ✅ Far distance (3km): {conf:.2f} confidence")
    
except Exception as e:
    print(f"   ❌ Confidence test failed: {e}")
    sys.exit(1)

# Test 5: Fallback Functions
print("\n5️⃣ Testing fallback functions...")
try:
    result = simple_eta_fallback(13.0, 80.23, 18)
    assert 'eta_seconds' in result
    assert 'confidence' in result
    assert 'method' in result
    assert result['method'] == 'simple_fallback'
    assert result['eta_seconds'] <= 1800  # Should be bounded
    print(f"   ✅ Simple fallback works: {result['eta_minutes']:.1f} min")
    
except Exception as e:
    print(f"   ❌ Fallback test failed: {e}")
    sys.exit(1)

# Test 6: Model Loading (with GNN disabled)
print("\n6️⃣ Testing model loading...")
try:
    model = EnsembleModel(use_gnn=False)
    assert model.gnn_model is None, "GNN should be None when disabled"
    assert model.xgb_model is not None, "XGBoost should be loaded"
    assert model.lgb_model is not None, "LightGBM should be loaded"
    print("   ✅ Model loads successfully (GNN disabled)")
    
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    sys.exit(1)

# Test 7: Prediction with Bounds
print("\n7️⃣ Testing prediction with bounds...")
try:
    preprocessor = LiveDataPreprocessor()
    
    # Create valid GPS data
    gps_data = {
        'latitude': 13.0,
        'longitude': 80.23,
        'speed': 20,
        'timestamp': '2025-11-24 10:00:00'
    }
    
    # Convert to graph
    graph = preprocessor.gps_to_graph(gps_data, target_stop_id=18)
    
    # Predict with bounds
    result = model.predict_with_bounds(graph)
    
    assert 'eta_seconds' in result
    assert 'eta_minutes' in result
    assert 'confidence' in result
    assert 'method' in result
    assert result['eta_seconds'] <= 1800, "Should be bounded"
    assert result['eta_seconds'] >= 30, "Should be bounded"
    
    print(f"   ✅ Prediction works: {result['eta_minutes']:.1f} min")
    print(f"      Confidence: {result['confidence']:.2f}")
    print(f"      Method: {result['method']}")
    print(f"      Capped: {result.get('was_capped', False)}")
    
except Exception as e:
    print(f"   ❌ Prediction test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Logging
print("\n8️⃣ Testing logging...")
try:
    # Check logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
        print("   ✅ Created logs directory")
    
    # Test logging functions
    log_prediction(gps_data, result, 'test', 1.5)
    print("   ✅ Prediction logging works")
    
    log_validation_failure('test_field', 'test_value', 'test_reason')
    print("   ✅ Validation logging works")
    
    # Check log files exist
    import glob
    log_files = glob.glob('logs/*.log')
    if log_files:
        print(f"   ✅ Log files created: {len(log_files)} files")
    else:
        print("   ⚠️  No log files yet (will be created on first run)")
    
except Exception as e:
    print(f"   ❌ Logging test failed: {e}")
    sys.exit(1)

# All tests passed!
print("\n" + "="*80)
print("✅ ALL SANITY CHECKS PASSED!")
print("="*80)
print("\n🎉 Production fixes are working correctly!")
print("\n📋 Next steps:")
print("   1. Run validate_with_metrics.py for full accuracy validation")
print("   2. Run quick_validate.py for performance validation")
print("   3. Check logs/ directory for output")
print("\n🚀 Ready for shadow deployment!")
