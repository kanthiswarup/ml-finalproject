import numpy as np
import pandas as pd

def build_cgm_features(df):
    """
    Constructs feature set from raw CGM and lifestyle time-series data.
    Creates lag features, rolling statistics, physiological decay features,
    cyclical time encodings, and multi-horizon target variables.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)
    
    # Process per patient group to avoid cross-patient leakage
    processed_dfs = []
    
    for pid, group in df.groupby('patient_id'):
        group = group.copy()
        
        # 1. Temporal cyclical encodings
        hours = group['timestamp'].dt.hour + group['timestamp'].dt.minute / 60.0
        group['sin_hour'] = np.sin(2 * np.pi * hours / 24.0)
        group['cos_hour'] = np.cos(2 * np.pi * hours / 24.0)
        group['day_of_week'] = group['timestamp'].dt.dayofweek
        
        # 2. Glucose Lags (sampling interval = 5 mins)
        # lag_1 (5m), lag_2 (10m), lag_3 (15m), lag_6 (30m), lag_12 (60m)
        for lag in [1, 2, 3, 4, 6, 12]:
            group[f'cgm_lag_{lag}'] = group['cgm'].shift(lag)
            
        # 3. Glucose Rates of Change (ROC)
        group['cgm_roc_5m'] = group['cgm'] - group['cgm_lag_1']
        group['cgm_roc_15m'] = group['cgm'] - group['cgm_lag_3']
        group['cgm_roc_30m'] = group['cgm'] - group['cgm_lag_6']
        group['cgm_acceleration'] = group['cgm_roc_5m'] - group['cgm_roc_5m'].shift(1)
        
        # 4. Rolling Window Statistics (15m, 30m, 60m windows)
        for w in [3, 6, 12]:
            w_min = w * 5
            group[f'cgm_roll_mean_{w_min}m'] = group['cgm'].rolling(window=w).mean()
            group[f'cgm_roll_std_{w_min}m'] = group['cgm'].rolling(window=w).std()
            group[f'cgm_roll_max_{w_min}m'] = group['cgm'].rolling(window=w).max()
            group[f'cgm_roll_min_{w_min}m'] = group['cgm'].rolling(window=w).min()
            
        # 5. Exponentially Weighted Moving Averages
        group['cgm_ewma_short'] = group['cgm'].ewm(span=3).mean()
        group['cgm_ewma_long'] = group['cgm'].ewm(span=12).mean()
        
        # 6. Lifestyle aggregation features
        # Sum of carbs, insulin, steps over past 1 hour (12 steps)
        group['carbs_last_1h'] = group['carbs'].rolling(window=12, min_periods=1).sum()
        group['insulin_last_1h'] = group['insulin_bolus'].rolling(window=12, min_periods=1).sum()
        group['steps_last_1h'] = group['steps'].rolling(window=12, min_periods=1).sum()
        
        # 7. Targets for 30m forecasting (shift -6) and 60m forecasting (shift -12)
        group['target_cgm_30m'] = group['cgm'].shift(-6)
        group['target_cgm_60m'] = group['cgm'].shift(-12)
        
        # Classification Targets
        group['target_hypo_30m'] = (group['target_cgm_30m'] < 70).astype(int)
        group['target_hyper_30m'] = (group['target_cgm_30m'] > 180).astype(int)
        
        processed_dfs.append(group)
        
    full_featured_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Drop initial rows with NaNs resulting from lags/targets
    feature_df = full_featured_df.dropna().reset_index(drop=True)
    return feature_df

if __name__ == "__main__":
    print("Testing feature engineering module...")
    raw_df = pd.read_csv("data/cgm_lifestyle_dataset.csv")
    feat_df = build_cgm_features(raw_df)
    print(f"Engineered feature set shape: {feat_df.shape}")
