import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow imports for the LSTM model
from tensorflow.keras.models import load_model

# Scikit-learn imports for the other models
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="EV Predictive Maintenance Dashboard", layout="wide")

st.title("⚡ AI-Driven EV/HEV Predictive Maintenance")
st.markdown("""
**System Status:**
* **Battery RUL:** Deep Learning (LSTM) Model Loaded
* **Fault Diagnosis:** Random Forest Classifier Active
* **Maintenance:** Predictive Analytics Active
""")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Battery RUL (LSTM)", "NEV Fault Diagnosis", "Maintenance Prediction"])

# ==========================================
# MODEL LOADING FUNCTION
# ==========================================
@st.cache_resource
def load_and_train_models():
    models = {}
    
    # --- 1. Load Pre-Trained LSTM (From Colab) ---
    print("Loading LSTM Battery Model...")
    if os.path.exists('lstm_battery_model.h5') and os.path.exists('scaler_X.pkl'):
        try:
            # Load the Neural Network
            lstm_model = load_model('lstm_battery_model.h5')
            
            # Load the Scalers (Critical for correct predictions)
            with open('scaler_X.pkl', 'rb') as f:
                scaler_X = pickle.load(f)
            with open('scaler_y.pkl', 'rb') as f:
                scaler_y = pickle.load(f)
            
            # Load Test Data for the Graph
            df_batt = pd.read_csv("Battery_dataset.csv")
            test_batt = df_batt[df_batt['battery_id'] == 'B7']
            features_batt = ['cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt', 'SOH']
            
            models['batt'] = (lstm_model, scaler_X, scaler_y, test_batt, features_batt)
            print("LSTM Model Loaded Successfully.")
        except Exception as e:
            st.error(f"Error loading LSTM files: {e}")
            models['batt'] = None
    else:
        models['batt'] = None

    # --- 2. NEV Fault Model (Random Forest) ---
    print("Training NEV Fault Model...")
    try:
        train_nev = pd.read_csv("NEV_fault_training_dataset.csv")
        test_nev = pd.read_csv("NEV_fault_testing_dataset.csv")
        
        X_train = train_nev.drop('Fault Label', axis=1)
        y_train = train_nev['Fault Label']
        X_test = test_nev.drop('Fault Label', axis=1)
        y_test = test_nev['Fault Label']
        
        scaler_nev = StandardScaler()
        X_train_scaled = scaler_nev.fit_transform(X_train)
        
        rf_fault = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_fault.fit(X_train_scaled, y_train)
        
        models['nev'] = (rf_fault, scaler_nev, X_test, y_test)
    except Exception as e:
        print(f"NEV Model Error: {e}")
        models['nev'] = None

    # --- 3. Maintenance Model (Random Forest) ---
    print("Training Maintenance Model...")
    try:
        df_ev = pd.read_csv("EV_Predictive_Maintenance_Dataset_15min.csv")
        features_ev = ['SoC', 'SoH', 'Battery_Voltage', 'Battery_Current', 
                       'Motor_Temperature', 'Motor_Vibration', 'Driving_Speed']
        if 'Distance_Traveled' in df_ev.columns:
            features_ev.append('Distance_Traveled')
            
        df_clean = df_ev[features_ev + ['Maintenance_Type']].dropna().sample(5000)
        X = df_clean[features_ev]
        y = df_clean['Maintenance_Type']
        
        rf_maint = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_maint.fit(X, y)
        models['maint'] = (rf_maint, features_ev)
    except Exception as e:
        print(f"Maintenance Model Error: {e}")
        models['maint'] = None
        
    return models

# Load models once
data = load_and_train_models()

# ==========================================
# PAGE 1: BATTERY RUL (LSTM)
# ==========================================
if page == "Battery RUL (LSTM)":
    st.header("🔋 Battery Remaining Useful Life (RUL)")
    
    if data['batt']:
        model, scaler_X, scaler_y, test_data, features = data['batt']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Live Telemetry Input")
            cycle = st.number_input("Current Cycle Count", value=100)
            soh = st.slider("State of Health (SOH %)", 80.0, 100.0, 95.0)
            voltage = st.number_input("Charging Voltage (V)", value=4.2)
            temp = st.slider("Battery Temp (°C)", 20.0, 50.0, 25.0)
            
            if st.button("Predict RUL with AI"):
                # 1. Prepare Input Dataframe
                # (We fill other sensors with average values for this demo to make input easy)
                input_df = pd.DataFrame([{
                    'cycle': cycle, 'chI': 1.5, 'chV': voltage, 'chT': temp, 
                    'disI': 2.0, 'disV': 3.5, 'disT': 35.0, 'BCt': 2.0, 'SOH': soh
                }])
                
                # 2. Scale Input
                input_scaled = scaler_X.transform(input_df)
                
                # 3. Reshape for LSTM [1 sample, 1 timestep, N features]
                input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
                
                # 4. Predict
                pred_scaled = model.predict(input_reshaped)
                
                # 5. Inverse Scale to get Real Cycle Count
                pred_actual = scaler_y.inverse_transform(pred_scaled)[0][0]
                
                st.success(f"Predicted RUL: **{int(pred_actual)} cycles**")
                
                # Visual Indicator
                st.progress(min(int(soh), 100))
                st.caption(f"Based on Deep Learning analysis of battery degradation patterns.")

        with col2:
            st.subheader("Model Validation (Battery B7)")
            st.write("Comparing LSTM predictions against actual historical data for Battery #7.")
            
            # Prepare Test Data for Plot
            X_test_raw = test_data[features]
            X_test_scaled = scaler_X.transform(X_test_raw)
            # Reshape
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            
            # Batch Prediction
            y_pred_scaled = model.predict(X_test_reshaped)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_true = test_data['RUL'].values
            
            # Plotting
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_true, label='Actual RUL', color='blue', linewidth=2)
            ax.plot(y_pred, label='LSTM Prediction', color='red', linestyle='--', linewidth=2)
            ax.set_title("LSTM Model Accuracy on Unseen Data")
            ax.set_xlabel("Time (Cycles)")
            ax.set_ylabel("Remaining Useful Life")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
    else:
        st.warning("⚠️ LSTM Model files not found.")
        st.info("Please train the model in Colab, download `.h5` & `.pkl` files, and place them in this folder.")

# ==========================================
# PAGE 2: NEV FAULT DIAGNOSIS
# ==========================================
elif page == "NEV Fault Diagnosis":
    st.header("🛠️ Component Fault Diagnosis")
    
    if data['nev']:
        model, scaler, X_test, y_test = data['nev']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Sensor Simulator")
            v_val = st.number_input("Voltage (Normalized)", value=0.5)
            c_val = st.number_input("Current (Normalized)", value=0.5)
            rpm_val = st.number_input("Motor RPM (Normalized)", value=0.5)
            temp_val = st.number_input("Temperature (Normalized)", value=0.5)
            vib_val = st.number_input("Vibration (Normalized)", value=0.5)
            
            # Prepare Input Vector (7 features total in training data)
            # We fill ambient temp/humidity with defaults
            input_vec = np.array([[v_val, c_val, rpm_val, temp_val, vib_val, 0.5, 0.5]])
            input_scaled = scaler.transform(input_vec)
            
            if st.button("Diagnose System"):
                pred = model.predict(input_scaled)[0]
                
                fault_map = {0: "✅ Normal", 1: "⚠️ Voltage Sag", 2: "⚠️ Current Leakage", 3: "🔥 Overheating/Vibration"}
                status = fault_map.get(pred, "Unknown")
                
                if pred == 0:
                    st.success(f"System Status: {status}")
                else:
                    st.error(f"System Status: {status}")

        with col2:
            st.subheader("Confusion Matrix")
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

    else:
        st.error("NEV Data not found.")

# ==========================================
# PAGE 3: MAINTENANCE PREDICTION
# ==========================================
elif page == "Maintenance Prediction":
    st.header("🔧 Maintenance Forecasting")
    
    if data['maint']:
        model, features = data['maint']
        
        st.subheader("Vehicle Conditions")
        c1, c2, c3 = st.columns(3)
        soc = c1.slider("State of Charge", 0.0, 1.0, 0.5)
        soh = c2.slider("State of Health", 0.0, 1.0, 0.9)
        speed = c3.slider("Avg Speed (km/h)", 0.0, 120.0, 60.0)
        
        c4, c5 = st.columns(2)
        mot_temp = c4.number_input("Motor Temp (°C)", value=45.0)
        dist = c5.number_input("Odometer (km)", value=100.0)
        
        # DataFrame for input
        input_data = pd.DataFrame([{
            'SoC': soc, 'SoH': soh, 'Battery_Voltage': 200, 'Battery_Current': 10,
            'Motor_Temperature': mot_temp, 'Motor_Vibration': 0.5, 'Driving_Speed': speed,
            'Distance_Traveled': dist
        }])
        
        # Align columns
        input_data = input_data.reindex(columns=features, fill_value=0)
        
        if st.button("Analyze Maintenance Needs"):
            pred = model.predict(input_data)[0]
            
            maint_map = {0: "None Required", 1: "Routine Checkup", 2: "Urgent Service"}
            res = maint_map.get(pred, "Manual Inspection")
            
            st.info(f"Recommended Action: **{res}**")
            
            # Feature Importance
            st.write("---")
            st.caption("Key Contributing Factors:")
            importances = model.feature_importances_
            # Sort top 5
            indices = np.argsort(importances)[::-1][:5]
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh([features[i] for i in indices], importances[indices], color='#4CAF50')
            ax.invert_yaxis()
            st.pyplot(fig)

    else:
        st.error("Maintenance Data not found.")