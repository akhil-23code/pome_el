import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load and Preprocess Data
df_batt = pd.read_csv("Battery_dataset.csv")
features = ['cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt', 'SOH']
target = 'RUL'

# Split by Battery ID
train_df = df_batt[df_batt['battery_id'].isin(['B5', 'B6'])]
test_df = df_batt[df_batt['battery_id'] == 'B7']

# Scaling (Neural Networks require scaled data, 0-1 range)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(train_df[features])
y_train = scaler_y.fit_transform(train_df[[target]])
X_test = scaler_X.transform(test_df[features])
y_test = scaler_y.transform(test_df[[target]])

# 2. Reshape for LSTM [Samples, TimeSteps, Features]
# We treat each row as a single timestep for simplicity here, 
# or you can create sliding windows (sequences) for better accuracy.
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 3. Build the LSTM Neural Network
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, len(features))))
model.add(Dropout(0.2)) # Prevents overfitting
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
model.add(Dense(1)) # Output layer (Predicting RUL)

model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Train
print("Training Neural Network...")
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)

# 5. Predict and Inverse Scale
y_pred_scaled = model.predict(X_test_reshaped)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test)

# 6. Visualize
plt.figure(figsize=(10, 5))
plt.plot(y_actual, label='Actual RUL', color='blue')
plt.plot(y_pred, label='LSTM Prediction', color='red', linestyle='--')
plt.title('Battery RUL Prediction using LSTM Neural Network')
plt.xlabel('Samples')
plt.ylabel('RUL (Cycles)')
plt.legend()
plt.show()