# GlucoTrend: AI-Based Continuous Glucose Monitoring (CGM) Forecasting and Lifestyle Analytics System

> **Machine Learning Course Final Project**  
> **Domain**: Healthcare Data Science, Physiological Time-Series Machine Learning, Predictive Analytics

---

## 📌 1. Project Abstract & Objectives

Continuous Glucose Monitoring (CGM) devices track interstitial blood glucose levels at 5-minute sampling intervals (288 readings/day). For individuals managing insulin-dependent diabetes, predicting future glycemic trajectories is critical to preventing **Hypoglycemia** (<70 mg/dL — dangerous low blood sugar) and **Hyperglycemia** (>180 mg/dL — elevated blood sugar spikes).

**GlucoTrend** is an end-to-end Machine Learning system that combines continuous time-series glucose metrics with key lifestyle data—including carbohydrate intake, insulin bolus dosage, physical activity step count, sleep quality, and stress levels.

### Key Objectives:
1. **Multi-Horizon Forecasting**: Accurately predict glucose levels 30 minutes and 60 minutes into the future.
2. **Early Hypoglycemia Alerting**: Classify impending hypoglycemia risk (<70 mg/dL) 30 minutes in advance.
3. **Lifestyle Analytics**: Quantify the impact of meals, insulin boluses, and exercise on glycemic dynamics.
4. **Interactive Dashboard**: Provide a real-time web application for visualization, clinical metrics calculation, and "What-If" scenario simulation.

---

## 🏗️ 2. System Architecture & Methodology

```
┌───────────────────────────┐      ┌───────────────────────────┐
│ Multi-Patient Raw CGM &   │ ───► │ Feature Engineering       │
│ Lifestyle Data Stream     │      │ (Lags, Rolling, COB, IOB) │
└───────────────────────────┘      └─────────────┬─────────────┘
                                                 │
                                                 ▼
┌───────────────────────────┐      ┌───────────────────────────┐
│ Patient-Aware Train/Test  │ ───► │ Multi-Model ML Pipeline   │
│ Split (PATIENT 1-4 / 5)   │      │ (Ridge, RF, XGBoost, GBDT)│
└───────────────────────────┘      └─────────────┬─────────────┘
                                                 │
                                                 ▼
┌───────────────────────────┐      ┌───────────────────────────┐
│ Streamlit Interactive App │ ◄─── │ Evaluated Models &        │
│ & Lifestyle Simulator     │      │ ISO 15197 Metrics         │
└───────────────────────────┘      └───────────────────────────┘
```

### Feature Engineering Pipeline (`src/feature_engineering.py`)
- **Glucose Lag Features**: Lags at $t-5$, $t-10$, $t-15$, $t-30$, and $t-60$ minutes.
- **Rates of Change (ROC)**: 5-min, 15-min, 30-min derivatives and acceleration ($\Delta^2$).
- **Rolling Statistics**: 15-min, 30-min, and 60-min window mean, standard deviation, min, and max.
- **Physiological Response Curves**: Carbs-On-Board (COB decay) and Insulin-On-Board (IOB decay) exponential models.
- **Temporal Encodings**: Diurnal sine/cosine cyclical time encodings ($\sin(2\pi \cdot t/24)$, $\cos(2\pi \cdot t/24)$).

---

## 📊 3. Experimental Results & Performance Benchmarks

### 3.1 30-Minute Horizon Glucose Forecasting Performance
Evaluating models on out-of-sample holdout patient (`PATIENT_005`):

| Model Architecture | RMSE (mg/dL) | MAE (mg/dL) | $R^2$ Score | ISO 15197 Zone A Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression** | **4.56** | **3.34** | **0.9291** | **99.15%** |
| **XGBoost Regressor** | 4.60 | 3.38 | 0.9279 | 98.78% |
| **Gradient Boosting** | 4.63 | 3.41 | 0.9271 | 98.83% |
| **Random Forest** | 4.76 | 3.48 | 0.9228 | 98.68% |

> **Clinical Significance**: **Over 99%** of predictions fall within the strict ISO 15197 clinical accuracy criteria ($\pm 15$ mg/dL for blood glucose levels $<100$ mg/dL or $\pm 15\%$ for levels $\ge 100$ mg/dL).

### 3.2 60-Minute Horizon Glucose Forecasting Performance

| Model Architecture | RMSE (mg/dL) | MAE (mg/dL) | $R^2$ Score | ISO 15197 Zone A Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression** | **7.94** | **5.52** | **0.7848** | **93.71%** |
| **XGBoost Regressor** | 8.00 | 5.63 | 0.7815 | 93.91% |
| **Gradient Boosting** | 7.97 | 5.63 | 0.7830 | 93.61% |
| **Random Forest** | 8.57 | 5.97 | 0.7494 | 92.39% |

### 3.3 Hypoglycemia Risk Binary Classification (<70 mg/dL)
- **Model**: XGBoost Classifier with positive class weighting.
- **F1 Score**: `0.7518`
- **Precision**: `0.6897`
- **Recall (Sensitivity)**: `0.8262` (Detects $>82\%$ of impending low blood sugar events)
- **ROC-AUC**: **`0.9624`**

---

## 📁 4. Project Directory Structure

```
ml final project/
├── data/
│   └── cgm_lifestyle_dataset.csv     # 14-day multi-patient continuous dataset
├── models/
│   ├── cgm_model_30min.joblib        # Trained 30-min prediction model
│   ├── cgm_model_60min.joblib        # Trained 60-min prediction model
│   ├── hypo_classifier.joblib        # Trained hypoglycemia risk classifier
│   ├── scaler.joblib                 # StandardScaler object
│   ├── feature_cols.joblib           # Feature list ordering
│   └── model_metrics.json            # Model benchmark results JSON
├── src/
│   ├── __init__.py
│   ├── data_generator.py             # Physiological CGM & lifestyle simulator
│   ├── feature_engineering.py        # Feature extraction and lag processor
│   ├── train_models.py               # Model training & evaluation script
│   └── evaluate.py                   # Plot generator & evaluation artifact creator
├── notebooks/
│   └── GlucoTrend_ML_Project.ipynb   # Complete submission Jupyter Notebook
├── reports/
│   └── figures/                      # High-res evaluation charts & scatter plots
├── app.py                            # Streamlit Web Application Dashboard
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🚀 5. How to Run the Project

### Prerequisites
Make sure Python 3.10+ is installed on your system.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Models
```bash
python src/train_models.py
```

### 3. Generate Evaluation Figures & Reports
```bash
python src/evaluate.py
```

### 4. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```

---

## 💡 6. Key Takeaways & Clinical Impact

- **Physiological Feature Modeling**: Modeling Carbs-On-Board (COB) and Insulin-On-Board (IOB) decay significantly enhanced model stability against rapid meal spikes.
- **Zero-Data Leakage Split**: Splitting datasets by patient identity ensures the models learn generalized physiological patterns rather than memorizing individual baseline trajectories.
- **Actionable Decision Support**: The interactive "What-If" simulator empowers patients and clinicians to adjust insulin bolus and carbohydrate intake prior to meals, preventing extreme glycemic excursions.
