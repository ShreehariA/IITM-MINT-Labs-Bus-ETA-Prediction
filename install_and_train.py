"""
Quick script to install dependencies and run XGBoost training
Run this in your notebook: %run install_and_train.py
"""
import subprocess
import sys

print("="*80)
print("INSTALLING DEPENDENCIES")
print("="*80)

# Install packages
print("\n📦 Installing XGBoost, LightGBM, scikit-learn...")
subprocess.run([sys.executable, "-m", "pip", "install", "xgboost", "lightgbm", "scikit-learn"], check=True)

print("\n✅ Installation complete!")
print("\n" + "="*80)
print("Now run: %run train_xgboost_ensemble.py")
print("="*80)
