import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_cgm_dataset(
    num_patients=5,
    days_per_patient=14,
    sampling_rate_minutes=5,
    seed=42
):
    """
    Generates synthetic Continuous Glucose Monitoring (CGM) dataset integrated with 
    lifestyle factors (Carbohydrate intake, Insulin bolus/basal, Physical activity, Stress, Sleep).
    
    Physiological dynamics modelled:
    - Meal carbohydrate absorption curves (Carbs-on-Board, COB)
    - Insulin absorption & action curves (Insulin-on-Board, IOB)
    - Exercise-induced glucose utilization
    - Circadian rhythm / Dawn phenomenon
    - Interstitial measurement noise
    """
    np.random.seed(seed)
    records_per_day = (24 * 60) // sampling_rate_minutes
    total_records_per_patient = days_per_patient * records_per_day
    
    all_data = []
    start_date = datetime(2026, 1, 1, 0, 0, 0)

    for patient_id in range(1, num_patients + 1):
        # Patient physiological parameters
        basal_glucose = np.random.uniform(105, 125)        # mg/dL
        carb_sensitivity = np.random.uniform(2.5, 4.0)     # mg/dL per gram carb
        insulin_sensitivity = np.random.uniform(30, 50)    # mg/dL drop per unit insulin
        exercise_sensitivity = np.random.uniform(0.6, 1.2) # mg/dL drop per workout unit
        
        # State arrays
        time_stamps = [start_date + timedelta(minutes=i * sampling_rate_minutes) for i in range(total_records_per_patient)]
        glucose = np.full(total_records_per_patient, basal_glucose)
        carbs = np.zeros(total_records_per_patient)
        insulin_bolus = np.zeros(total_records_per_patient)
        insulin_basal = np.full(total_records_per_patient, 0.8) # units/hr baseline
        steps = np.zeros(total_records_per_patient)
        stress_level = np.zeros(total_records_per_patient) # 0 to 10
        sleep_quality = np.full(total_records_per_patient, 7.5) # 1 to 10 scale
        
        # Simulate day-by-day lifestyle events
        for day in range(days_per_patient):
            day_offset = day * records_per_day
            
            # Daily sleep quality (fixed for the day)
            daily_sleep = np.clip(np.random.normal(7.5, 1.2), 3, 10)
            sleep_quality[day_offset : day_offset + records_per_day] = daily_sleep

            # Meal Schedule per day (Breakfast 7-9am, Lunch 12-2pm, Dinner 6-8pm, Snack 9-10pm)
            meal_times = [
                day_offset + int((7 + np.random.uniform(0, 1.5)) * 60 / sampling_rate_minutes),
                day_offset + int((12 + np.random.uniform(0, 1.5)) * 60 / sampling_rate_minutes),
                day_offset + int((18.5 + np.random.uniform(0, 1.5)) * 60 / sampling_rate_minutes),
            ]
            
            # Optional snack (50% probability)
            if np.random.rand() > 0.5:
                meal_times.append(day_offset + int((21 + np.random.uniform(0, 0.8)) * 60 / sampling_rate_minutes))

            for m_idx, m_step in enumerate(meal_times):
                if m_step < total_records_per_patient:
                    # Carb amount based on meal
                    if m_idx == 0:
                        c_amount = np.random.uniform(35, 65) # Breakfast
                    elif m_idx == 1:
                        c_amount = np.random.uniform(45, 85) # Lunch
                    elif m_idx == 2:
                        c_amount = np.random.uniform(55, 95) # Dinner
                    else:
                        c_amount = np.random.uniform(15, 30) # Snack

                    carbs[m_step] += c_amount

                    # Corresponding bolus insulin given (with realistic slight under/over estimation)
                    est_bolus = (c_amount * carb_sensitivity) / insulin_sensitivity
                    actual_bolus = max(0, est_bolus * np.random.uniform(0.85, 1.15))
                    insulin_bolus[m_step] += actual_bolus

            # Daily exercise session (60% probability per day around 5 PM or 7 AM)
            if np.random.rand() < 0.6:
                ex_hour = 17 if np.random.rand() > 0.3 else 7
                ex_step = day_offset + int((ex_hour + np.random.uniform(-0.5, 0.5)) * 60 / sampling_rate_minutes)
                duration_steps = int(np.random.uniform(30, 60) / sampling_rate_minutes)
                for s in range(ex_step, min(ex_step + duration_steps, total_records_per_patient)):
                    steps[s] = np.random.uniform(800, 1500) # steps per 5-min interval

            # Daily stress peaks
            stress_hours = np.random.choice(range(9, 18), size=np.random.randint(1, 3), replace=False)
            for sh in stress_hours:
                st_step = day_offset + int(sh * 60 / sampling_rate_minutes)
                for s in range(st_step, min(st_step + 6, total_records_per_patient)):
                    stress_level[s] = np.random.uniform(5, 9)

        # Compute physiological dynamics across full timeline
        cob = np.zeros(total_records_per_patient) # Carbs on board
        iob = np.zeros(total_records_per_patient) # Insulin on board

        # Exponential decay parameters (half-lives)
        carb_decay = 0.03   # ~45-60 min peak carb impact
        insulin_decay = 0.02 # ~3-4 hours insulin action duration

        curr_cob = 0.0
        curr_iob = 0.0

        for t in range(total_records_per_patient):
            curr_cob = curr_cob * (1 - carb_decay) + carbs[t]
            curr_iob = curr_iob * (1 - insulin_decay) + insulin_bolus[t]
            cob[t] = curr_cob
            iob[t] = curr_iob

        # Compute cumulative glucose dynamics
        for t in range(1, total_records_per_patient):
            dt = sampling_rate_minutes
            
            # Diurnal circadian variation (Dawn phenomenon early morning 4-8 AM)
            hour_of_day = time_stamps[t].hour + time_stamps[t].minute / 60.0
            dawn_effect = 12.0 * np.exp(-((hour_of_day - 6.0)**2) / 4.0)
            
            # Carb elevation effect
            carb_effect = cob[t-1] * carb_sensitivity * 0.04
            
            # Insulin lowering effect
            insulin_effect = iob[t-1] * insulin_sensitivity * 0.035
            
            # Exercise effect
            activity_effect = steps[t-1] * 0.005 * exercise_sensitivity
            
            # Stress effect (cortisol elevation)
            stress_effect = stress_level[t-1] * 0.8
            
            # Sleep penalty effect (poor sleep increases basal glucose)
            sleep_effect = max(0, (7.0 - sleep_quality[t-1])) * 1.5
            
            # Mean reversion towards individual baseline
            reversion = 0.03 * (basal_glucose + dawn_effect + sleep_effect - glucose[t-1])
            
            # Random physiological noise
            noise = np.random.normal(0, 1.2)
            
            # Update glucose state
            d_glucose = reversion + carb_effect - insulin_effect - activity_effect + stress_effect + noise
            glucose[t] = np.clip(glucose[t-1] + d_glucose, 40, 400) # clinical bounds

        # Add interstitial fluid sensor lag (5 min EMA smoothing) + sensor noise
        cgm_measured = pd.Series(glucose).ewm(span=3).mean().values + np.random.normal(0, 1.5, size=total_records_per_patient)
        cgm_measured = np.clip(cgm_measured, 40, 400)

        patient_df = pd.DataFrame({
            'timestamp': time_stamps,
            'patient_id': f'PATIENT_{patient_id:03d}',
            'cgm': np.round(cgm_measured, 1),
            'true_glucose': np.round(glucose, 1),
            'carbs': np.round(carbs, 1),
            'insulin_bolus': np.round(insulin_bolus, 2),
            'insulin_basal': np.round(insulin_basal, 2),
            'steps': np.round(steps, 0),
            'stress_level': np.round(stress_level, 1),
            'sleep_quality': np.round(sleep_quality, 1),
            'cob': np.round(cob, 1),
            'iob': np.round(iob, 2)
        })
        all_data.append(patient_df)

    full_df = pd.concat(all_data, ignore_index=True)
    return full_df

if __name__ == "__main__":
    import os
    print("Generating synthetic multi-patient CGM dataset...")
    df = generate_cgm_dataset(num_patients=5, days_per_patient=14)
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "cgm_lifestyle_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path} with shape {df.shape}")
    print(df.head())
