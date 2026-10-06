import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, roc_curve, auc

from feature_engineering import build_cgm_features

def generate_evaluation_artifacts():
    print("Generating evaluation figures and reports...")
    os.makedirs("reports/figures", exist_ok=True)
    
    # Load dataset & features
    df_raw = pd.read_csv("data/cgm_lifestyle_dataset.csv")
    df_feat = build_cgm_features(df_raw)
    
    # Test set patient 5
    test_df = df_feat[df_feat['patient_id'] == 'PATIENT_005'].reset_index(drop=True)
    
    feature_cols = joblib.load("models/feature_cols.joblib")
    scaler = joblib.load("models/scaler.joblib")
    
    X_test = test_df[feature_cols]
    X_test_scaled = scaler.transform(X_test)
    y_test_30 = test_df['target_cgm_30m']
    
    model_30 = joblib.load("models/cgm_model_30min.joblib")
    
    # Predict
    if hasattr(model_30, 'coef_'):
        preds_30 = model_30.predict(X_test_scaled)
    else:
        preds_30 = model_30.predict(X_test)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Figure 1: Predicted vs Actual Scatter Plot (Clarke Error Grid concept)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_30, preds_30, alpha=0.3, color='#2563eb', edgecolors='none', s=18)
    plt.plot([40, 350], [40, 350], 'r--', label='Ideal 1:1 Line')
    plt.axhspan(70, 180, alpha=0.1, color='green', label='Target Glucose Range (70-180 mg/dL)')
    plt.title("GlucoTrend: Actual vs Predicted 30-Min Glucose Trajectory", fontsize=12, fontweight='bold')
    plt.xlabel("Actual Glucose (mg/dL)", fontsize=11)
    plt.ylabel("Predicted Glucose (mg/dL)", fontsize=11)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("reports/figures/actual_vs_predicted_30min.png", dpi=300)
    plt.close()
    
    # Figure 2: Time-Series Sample Segment Prediction
    sample_segment = test_df.iloc[100:350]
    sample_y = sample_segment['target_cgm_30m'].values
    if hasattr(model_30, 'coef_'):
        sample_preds = model_30.predict(scaler.transform(sample_segment[feature_cols]))
    else:
        sample_preds = model_30.predict(sample_segment[feature_cols])

    plt.figure(figsize=(12, 5))
    plt.plot(sample_segment['timestamp'], sample_y, label='Actual CGM (t+30m)', color='#1e293b', linewidth=2)
    plt.plot(sample_segment['timestamp'], sample_preds, label='GlucoTrend AI Forecast', color='#2563eb', linestyle='--', linewidth=2)
    plt.axhline(70, color='red', linestyle=':', label='Hypo Boundary (70 mg/dL)')
    plt.axhline(180, color='orange', linestyle=':', label='Hyper Boundary (180 mg/dL)')
    plt.title("Patient 005 Continuous Glucose Forecasting (30-Minute Horizon)", fontsize=12, fontweight='bold')
    plt.xlabel("Timestamp", fontsize=11)
    plt.ylabel("Glucose Level (mg/dL)", fontsize=11)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("reports/figures/cgm_forecasting_timeseries.png", dpi=300)
    plt.close()

    print("Evaluation artifacts successfully saved to reports/figures/")

if __name__ == "__main__":
    generate_evaluation_artifacts()
