import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_project():
    print("==================================================")
    print("   AI-Driven Predictive Maintenance for EVs/HEVs")
    print("==================================================\n")

    # ---------------------------------------------------------
    # MODULE 1: Battery Remaining Useful Life (RUL) Prediction
    # ---------------------------------------------------------
    print(">>> Module 1: Training Battery RUL Model...")
    
    try:
        # Load Data
        df_batt = pd.read_csv("Battery_dataset.csv")
        
        # Preprocessing & Feature Selection
        features_batt = ['cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt', 'SOH']
        target_batt = 'RUL'
        
        # Strategic Split: Train on Batteries B5, B6; Test on B7 (Simulating real-world usage)
        train_batt = df_batt[df_batt['battery_id'].isin(['B5', 'B6'])]
        test_batt = df_batt[df_batt['battery_id'] == 'B7']
        
        X_train = train_batt[features_batt]
        y_train = train_batt[target_batt]
        X_test = test_batt[features_batt]
        y_test = test_batt[target_batt]
        
        # Model Training (Random Forest Regressor)
        rf_rul = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_rul.fit(X_train, y_train)
        
        # Evaluation
        y_pred = rf_rul.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"   [Result] RUL Model R2 Score: {r2:.2f}")
        print(f"   [Result] RMSE (Cycles): {rmse:.2f}")
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label='Actual RUL', color='#1f77b4', linewidth=2)
        plt.plot(y_pred, label='Predicted RUL', color='#ff7f0e', linestyle='--', linewidth=2)
        plt.title('Battery RUL Prediction (Unseen Battery B7)')
        plt.xlabel('Cycle Index')
        plt.ylabel('Remaining Useful Life')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        print(f"   [Error] Could not train Battery Model: {e}")

    # ---------------------------------------------------------
    # MODULE 2: NEV Fault Diagnosis
    # ---------------------------------------------------------
    print("\n>>> Module 2: Training NEV Fault Classification Model...")
    
    try:
        # Load Data
        train_nev = pd.read_csv("NEV_fault_training_dataset.csv")
        test_nev = pd.read_csv("NEV_fault_testing_dataset.csv")
        
        X_train_n = train_nev.drop('Fault Label', axis=1)
        y_train_n = train_nev['Fault Label']
        X_test_n = test_nev.drop('Fault Label', axis=1)
        y_test_n = test_nev['Fault Label']
        
        # Preprocessing: Standard Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_n)
        X_test_scaled = scaler.transform(X_test_n)
        
        # Model Training (Random Forest Classifier)
        rf_fault = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_fault.fit(X_train_scaled, y_train_n)
        
        # Evaluation
        y_pred_n = rf_fault.predict(X_test_scaled)
        acc = accuracy_score(y_test_n, y_pred_n)
        
        print(f"   [Result] Fault Diagnosis Accuracy: {acc*100:.2f}%")
        print("\n   [Report] Classification Report:")
        print(classification_report(y_test_n, y_pred_n))
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test_n, y_pred_n)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('NEV Fault Classification Confusion Matrix')
        plt.ylabel('Actual Fault Class')
        plt.xlabel('Predicted Fault Class')
        plt.show()
        
    except Exception as e:
        print(f"   [Error] Could not train Fault Model: {e}")

    # ---------------------------------------------------------
    # MODULE 3: Holistic EV Maintenance Prediction
    # ---------------------------------------------------------
    print("\n>>> Module 3: Training General Maintenance Predictor...")
    
    try:
        df_ev = pd.read_csv("EV_Predictive_Maintenance_Dataset_15min.csv")
        
        # Feature Engineering: Select relevant telemetry
        features_ev = ['SoC', 'SoH', 'Battery_Voltage', 'Battery_Current', 
                       'Motor_Temperature', 'Motor_Vibration', 'Driving_Speed']
                       
        if 'Distance_Traveled' in df_ev.columns:
            features_ev.append('Distance_Traveled')
            
        target_ev = 'Maintenance_Type'
        
        # Drop missing values
        df_ev_clean = df_ev[features_ev + [target_ev]].dropna()
        
        # Use a subset for faster demonstration (20k samples)
        df_ev_sample = df_ev_clean.sample(n=20000, random_state=42)
        
        X = df_ev_sample[features_ev]
        y = df_ev_sample[target_ev]
        
        X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model Training
        rf_maint = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_maint.fit(X_train_e, y_train_e)
        
        # Evaluation
        acc_e = accuracy_score(y_test_e, rf_maint.predict(X_test_e))
        print(f"   [Result] Maintenance Prediction Accuracy: {acc_e*100:.2f}%")
        
        # Feature Importance Plot
        importances = rf_maint.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Key Factors for EV Maintenance")
        plt.bar(range(X.shape[1]), importances[indices], align="center", color='green')
        plt.xticks(range(X.shape[1]), [features_ev[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"   [Error] Could not train EV Maintenance Model: {e}")

if __name__ == "__main__":
    run_project()