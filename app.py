import sys
import os

# Auto-launch Streamlit CLI only when NOT already running inside an active Streamlit Runtime server
try:
    from streamlit.runtime import Runtime
    is_running_in_streamlit = Runtime.exists()
except Exception:
    is_running_in_streamlit = False

if __name__ == "__main__" and not is_running_in_streamlit:
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", __file__] + sys.argv[1:]
    sys.exit(stcli.main())

import json
import joblib
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress minor unpickle/version warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="GlucoTrend AI - CGM Forecasting System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .alert-card-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .alert-card-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .alert-card-success {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">🩸 GlucoTrend</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Based Continuous Glucose Monitoring (CGM) Forecasting and Lifestyle Analytics System</div>', unsafe_allow_html=True)

# Load data and trained models
@st.cache_data
def load_data():
    raw_path = "data/cgm_lifestyle_dataset.csv"
    if not os.path.exists(raw_path):
        from src.data_generator import generate_cgm_dataset
        df = generate_cgm_dataset()
        df.to_csv(raw_path, index=False)
    else:
        df = pd.read_csv(raw_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_resource
def load_models():
    model_30 = joblib.load("models/cgm_model_30min.joblib")
    model_60 = joblib.load("models/cgm_model_60min.joblib")
    scaler = joblib.load("models/scaler.joblib")
    feature_cols = joblib.load("models/feature_cols.joblib")
    hypo_clf = joblib.load("models/hypo_classifier.joblib")
    
    with open("models/model_metrics.json", "r") as f:
        metrics = json.load(f)
        
    return model_30, model_60, scaler, feature_cols, hypo_clf, metrics

df_raw = load_data()
model_30, model_60, scaler, feature_cols, hypo_clf, metrics = load_models()

# Sidebar Setup
st.sidebar.title("🎛️ Analytics Controls")
patient_list = df_raw['patient_id'].unique().tolist()
selected_patient = st.sidebar.selectbox("Select Patient Profile", patient_list, index=4)

patient_data = df_raw[df_raw['patient_id'] == selected_patient].sort_values('timestamp').reset_index(drop=True)

min_date = patient_data['timestamp'].min().date()
max_date = patient_data['timestamp'].max().date()
selected_date = st.sidebar.date_input("Select Date", value=min_date + pd.Timedelta(days=2), min_value=min_date, max_value=max_date)

# Filter patient data for 24-hour viewing window
day_start = pd.to_datetime(selected_date)
day_end = day_start + pd.Timedelta(days=1)
view_df = patient_data[(patient_data['timestamp'] >= day_start) & (patient_data['timestamp'] <= day_end)].copy()

# Compute Clinical Summary Statistics
cgm_vals = view_df['cgm'].values
mean_glucose = np.mean(cgm_vals)
sd_glucose = np.std(cgm_vals)
cv_glucose = (sd_glucose / mean_glucose) * 100.0 if mean_glucose > 0 else 0
gmi_est = 12.71 + (0.09148 * mean_glucose) # Glucose Management Indicator (eA1c)

tir_pct = np.mean((cgm_vals >= 70) & (cgm_vals <= 180)) * 100.0
tbr_pct = np.mean(cgm_vals < 70) * 100.0
tar_pct = np.mean(cgm_vals > 180) * 100.0

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Dynamic Forecasting & Analytics", "🧪 Interactive Lifestyle Simulator", "📊 Model Benchmarks & Report"])

with tab1:
    st.subheader("📊 Clinical Glycemic Metrics (24-Hour Profile)")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Mean Glucose", f"{mean_glucose:.1f} mg/dL")
    with col2:
        st.metric("Time in Range (TIR)", f"{tir_pct:.1f}%")
    with col3:
        st.metric("Time Below Range", f"{tbr_pct:.1f}%", delta_color="inverse")
    with col4:
        st.metric("Time Above Range", f"{tar_pct:.1f}%", delta_color="inverse")
    with col5:
        st.metric("Glucose Variability (%CV)", f"{cv_glucose:.1f}%")
    with col6:
        st.metric("Est. HbA1c (GMI)", f"{gmi_est:.2f}%")

    # Safety Alert Banner
    recent_val = cgm_vals[-1] if len(cgm_vals) > 0 else 120
    if recent_val < 70:
        st.markdown(f'<div class="alert-card-danger">🚨 CRITICAL ALERT: Immediate Hypoglycemia Risk Detected ({recent_val:.1f} mg/dL)! Consume 15g fast-acting carbs immediately.</div>', unsafe_allow_html=True)
    elif recent_val > 180:
        st.markdown(f'<div class="alert-card-warning">⚠️ WARNING: Hyperglycemia Trend Detected ({recent_val:.1f} mg/dL). Consider correction insulin dosage.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-card-success">✅ STABLE: Glucose level is within Target Range ({recent_val:.1f} mg/dL).</div>', unsafe_allow_html=True)

    # Interactive Plotly CGM & Lifestyle Trajectory
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Continuous Glucose Trajectory (CGM mg/dL)", "Lifestyle Factors (Carbs & Insulin Intake)")
    )

    # Target Range Shapes
    fig.add_hrect(y0=70, y1=180, fillcolor="rgba(34, 197, 94, 0.12)", line_width=0, row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", row=1, col=1, annotation_text="Hypo Threshold (70 mg/dL)")
    fig.add_hline(y=180, line_dash="dash", line_color="#F59E0B", row=1, col=1, annotation_text="Hyper Threshold (180 mg/dL)")

    # CGM Trace
    fig.add_trace(
        go.Scatter(
            x=view_df['timestamp'], y=view_df['cgm'],
            mode='lines+markers', name='Observed CGM',
            line=dict(color='#2563EB', width=2.5),
            marker=dict(size=4)
        ), row=1, col=1
    )

    # Carbs trace
    fig.add_trace(
        go.Bar(
            x=view_df['timestamp'], y=view_df['carbs'],
            name='Carbs Intake (g)', marker_color='#F59E0B', opacity=0.8
        ), row=2, col=1
    )

    # Insulin Bolus trace
    fig.add_trace(
        go.Scatter(
            x=view_df['timestamp'], y=view_df['insulin_bolus'],
            mode='lines+markers', name='Insulin Bolus (U)',
            line=dict(color='#DC2626', width=2, dash='dot'),
            yaxis='y2'
        ), row=2, col=1
    )

    fig.update_layout(
        height=550,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified
    )
    fig.update_yaxes(title_text="Glucose (mg/dL)", range=[40, 320], row=1, col=1)
    fig.update_yaxes(title_text="Carbs (g)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🧪 Interactive 'What-If' Lifestyle Simulator")
    st.write("Simulate how planned carb intake, insulin bolus dosage, and physical activity impact glucose trajectories over the next 30 & 60 minutes.")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sim_curr_cgm = st.slider("Current CGM Glucose Level (mg/dL)", 50, 300, 130, step=5)
        sim_carbs = st.number_input("Carbohydrate Intake (grams)", min_value=0, max_value=150, value=45, step=5)
        sim_bolus = st.number_input("Rapid-Acting Insulin Bolus (Units)", min_value=0.0, max_value=20.0, value=3.5, step=0.5)

    with col_sim2:
        sim_steps = st.slider("Physical Activity (Steps over next 30m)", 0, 5000, 1200, step=200)
        sim_stress = st.slider("Stress Level (1-10)", 1, 10, 3)
        sim_trend = st.selectbox("Recent Glucose Trend", ["Rising Rapidly (+3 mg/dL/min)", "Stable (0 mg/dL/min)", "Falling Rapidly (-3 mg/dL/min)"], index=1)

    # Calculate simulated trend factor
    trend_val = 3.0 if "Rising" in sim_trend else (-3.0 if "Falling" in sim_trend else 0.0)

    # Simplified physiological prediction response for interactive simulation
    delta_carbs = sim_carbs * 0.7
    delta_insulin = sim_bolus * 12.0
    delta_activity = (sim_steps / 1000.0) * 8.0
    delta_stress = (sim_stress - 3) * 2.0

    pred_30m = sim_curr_cgm + (trend_val * 6) + (delta_carbs * 0.6) - (delta_insulin * 0.5) - (delta_activity * 0.5) + delta_stress
    pred_60m = sim_curr_cgm + (trend_val * 12) + (delta_carbs * 1.0) - (delta_insulin * 0.95) - (delta_activity * 0.8) + (delta_stress * 1.5)

    pred_30m = float(np.clip(pred_30m, 40, 400))
    pred_60m = float(np.clip(pred_60m, 40, 400))

    st.markdown("---")
    st.subheader("🔮 Predicted Trajectory Results")

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Current Glucose", f"{sim_curr_cgm} mg/dL")
    with res_col2:
        delta_30 = pred_30m - sim_curr_cgm
        st.metric("Forecasted 30-Min Glucose", f"{pred_30m:.1f} mg/dL", delta=f"{delta_30:+.1f} mg/dL", delta_color="inverse" if pred_30m < 70 or pred_30m > 180 else "normal")
    with res_col3:
        delta_60 = pred_60m - sim_curr_cgm
        st.metric("Forecasted 60-Min Glucose", f"{pred_60m:.1f} mg/dL", delta=f"{delta_60:+.1f} mg/dL", delta_color="inverse" if pred_60m < 70 or pred_60m > 180 else "normal")

    if pred_30m < 70 or pred_60m < 70:
        st.error("🚨 HYPOGLYCEMIA WARNING: High risk of low blood sugar (< 70 mg/dL). Consider reducing insulin bolus or increasing carb buffer.")
    elif pred_30m > 180 or pred_60m > 180:
        st.warning("⚠️ HYPERGLYCEMIA ALERT: High risk of elevated blood sugar (> 180 mg/dL). Consider adjusting meal bolus.")
    else:
        st.success("✅ OPTIMAL DOSAGE: Predicted trajectory remains safely within target Time-in-Range (70-180 mg/dL).")

with tab3:
    st.subheader("📊 Model Performance Benchmark & Clinical Accuracy")

    st.markdown("#### 1. 30-Minute Horizon Glucose Forecasting Performance")
    df_30 = pd.DataFrame(metrics["30_min_forecasting"]).T
    st.dataframe(df_30.style.highlight_min(subset=["RMSE", "MAE"], color="#D1FAE5").highlight_max(subset=["R2", "ISO_Zone_A_Pct"], color="#D1FAE5"), use_container_width=True)

    st.markdown("#### 2. 60-Minute Horizon Glucose Forecasting Performance")
    df_60 = pd.DataFrame(metrics["60_min_forecasting"]).T
    st.dataframe(df_60.style.highlight_min(subset=["RMSE", "MAE"], color="#D1FAE5").highlight_max(subset=["R2", "ISO_Zone_A_Pct"], color="#D1FAE5"), use_container_width=True)

    st.markdown("#### 3. Hypoglycemia Early Warning Risk Classifier (< 70 mg/dL)")
    hypo_m = metrics["hypo_classification"]
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("F1 Score", f"{hypo_m['F1_Score']:.4f}")
    with c_m2:
        st.metric("Precision", f"{hypo_m['Precision']:.4f}")
    with c_m3:
        st.metric("Recall (Sensitivity)", f"{hypo_m['Recall']:.4f}")
    with c_m4:
        st.metric("ROC-AUC Score", f"{hypo_m['ROC_AUC']:.4f}")
