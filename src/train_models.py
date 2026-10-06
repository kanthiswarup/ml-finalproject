import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from data_generator import generate_cgm_dataset
from feature_engineering import build_cgm_features

def evaluate_clinical_zones(y_true, y_pred):
    """
    Calculates percentage of predictions within ISO 15197 / Clinical standard zone 
    (within +/- 15 mg/dL for glucose < 100 mg/dL or +/- 15% for glucose >= 100 mg/dL).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    is_low = y_true < 100
    err_low = np.abs(y_true[is_low] - y_pred[is_low]) <= 15
    
    is_high = y_true >= 100
    err_high = np.abs(y_true[is_high] - y_pred[is_high]) / y_true[is_high] <= 0.15
    
    zone_a_pct = (np.sum(err_low) + np.sum(err_high)) / len(y_true) * 100.0
    return np.round(zone_a_pct, 2)

def train_and_evaluate_all():
    print("Step 1: Generating dataset...")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    raw_data_path = os.path.join(data_dir, "cgm_lifestyle_dataset.csv")
    
    if not os.path.exists(raw_data_path):
        df_raw = generate_cgm_dataset(num_patients=5, days_per_patient=14, seed=42)
        df_raw.to_csv(raw_data_path, index=False)
    else:
        df_raw = pd.read_csv(raw_data_path)
        
    print(f"Loaded raw dataset shape: {df_raw.shape}")
    
    print("Step 2: Building features...")
    df_feat = build_cgm_features(df_raw)
    print(f"Engineered dataset shape: {df_feat.shape}")
    
    # Feature columns specification
    ignore_cols = [
        'timestamp', 'patient_id', 'cgm', 'true_glucose',
        'target_cgm_30m', 'target_cgm_60m', 'target_hypo_30m', 'target_hyper_30m'
    ]
    feature_cols = [c for c in df_feat.columns if c not in ignore_cols]
    
    print(f"Feature set count: {len(feature_cols)} features")
    
    # Patient-aware train/test split (Patients 1-4 for train, Patient 5 for holdout evaluation)
    train_df = df_feat[df_feat['patient_id'] != 'PATIENT_005'].reset_index(drop=True)
    test_df = df_feat[df_feat['patient_id'] == 'PATIENT_005'].reset_index(drop=True)
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    y_train_30 = train_df['target_cgm_30m']
    y_test_30 = test_df['target_cgm_30m']
    
    y_train_60 = train_df['target_cgm_60m']
    y_test_60 = test_df['target_cgm_60m']
    
    y_train_hypo = train_df['target_hypo_30m']
    y_test_hypo = test_df['target_hypo_30m']
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_cols.joblib"))
    
    metrics_report = {}
    
    # -------------------------------------------------------------
    # 30-Minute Glucose Forecasting Models
    # -------------------------------------------------------------
    print("\n--- Training 30-Min Glucose Forecasting Models ---")
    reg_models_30 = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42)
    }
    
    best_30_model = None
    best_30_rmse = float('inf')
    best_30_name = ""
    metrics_report["30_min_forecasting"] = {}

    for name, model in reg_models_30.items():
        if "Ridge" in name:
            model.fit(X_train_scaled, y_train_30)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train_30)
            preds = model.predict(X_test)
            
        rmse = np.sqrt(mean_squared_error(y_test_30, preds))
        mae = mean_absolute_error(y_test_30, preds)
        r2 = r2_score(y_test_30, preds)
        zone_a = evaluate_clinical_zones(y_test_30, preds)
        
        metrics_report["30_min_forecasting"][name] = {
            "RMSE": round(float(rmse), 3),
            "MAE": round(float(mae), 3),
            "R2": round(float(r2), 4),
            "ISO_Zone_A_Pct": round(float(zone_a), 2)
        }
        print(f"[{name}] RMSE: {rmse:.2f} mg/dL | MAE: {mae:.2f} mg/dL | R²: {r2:.4f} | Zone A: {zone_a}%")
        
        if rmse < best_30_rmse:
            best_30_rmse = rmse
            best_30_model = model
            best_30_name = name

    print(f"==> Best 30-Min Model: {best_30_name} (RMSE: {best_30_rmse:.2f})")
    joblib.dump(best_30_model, os.path.join(models_dir, "cgm_model_30min.joblib"))

    # -------------------------------------------------------------
    # 60-Minute Glucose Forecasting Models
    # -------------------------------------------------------------
    print("\n--- Training 60-Min Glucose Forecasting Models ---")
    reg_models_60 = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42)
    }
    
    best_60_model = None
    best_60_rmse = float('inf')
    best_60_name = ""
    metrics_report["60_min_forecasting"] = {}

    for name, model in reg_models_60.items():
        if "Ridge" in name:
            model.fit(X_train_scaled, y_train_60)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train_60)
            preds = model.predict(X_test)
            
        rmse = np.sqrt(mean_squared_error(y_test_60, preds))
        mae = mean_absolute_error(y_test_60, preds)
        r2 = r2_score(y_test_60, preds)
        zone_a = evaluate_clinical_zones(y_test_60, preds)
        
        metrics_report["60_min_forecasting"][name] = {
            "RMSE": round(float(rmse), 3),
            "MAE": round(float(mae), 3),
            "R2": round(float(r2), 4),
            "ISO_Zone_A_Pct": round(float(zone_a), 2)
        }
        print(f"[{name}] RMSE: {rmse:.2f} mg/dL | MAE: {mae:.2f} mg/dL | R²: {r2:.4f} | Zone A: {zone_a}%")
        
        if rmse < best_60_rmse:
            best_60_rmse = rmse
            best_60_model = model
            best_60_name = name

    print(f"==> Best 60-Min Model: {best_60_name} (RMSE: {best_60_rmse:.2f})")
    joblib.dump(best_60_model, os.path.join(models_dir, "cgm_model_60min.joblib"))

    # -------------------------------------------------------------
    # Hypoglycemia Risk Binary Classifier (<70 mg/dL)
    # -------------------------------------------------------------
    print("\n--- Training Hypoglycemia Risk Classifier (<70 mg/dL) ---")
    clf_hypo = XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=3, random_state=42)
    clf_hypo.fit(X_train, y_train_hypo)
    
    hypo_preds = clf_hypo.predict(X_test)
    hypo_probs = clf_hypo.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test_hypo, hypo_preds, zero_division=0)
    prec = precision_score(y_test_hypo, hypo_preds, zero_division=0)
    rec = recall_score(y_test_hypo, hypo_preds, zero_division=0)
    auc = roc_auc_score(y_test_hypo, hypo_probs) if len(np.unique(y_test_hypo)) > 1 else 1.0
    
    metrics_report["hypo_classification"] = {
        "F1_Score": round(float(f1), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "ROC_AUC": round(float(auc), 4)
    }
    print(f"[XGBoost Hypo Classifier] F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUC: {auc:.4f}")
    joblib.dump(clf_hypo, os.path.join(models_dir, "hypo_classifier.joblib"))

    # Save all metrics
    with open(os.path.join(models_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics_report, f, indent=4)
        
    print("\nTraining and Evaluation pipeline complete! Saved all models and metrics to models/")

if __name__ == "__main__":
    train_and_evaluate_all()
