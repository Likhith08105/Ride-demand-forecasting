# train.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ===========================================================
# 1️⃣ LOAD DATA
# ===========================================================
data_path = os.path.join('data', 'uber_fares.csv')   # update if your file has a different name
print(f"📂 Loading dataset from: {data_path}")

data = pd.read_csv(data_path)
print(f"✅ Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns")

# ===========================================================
# 2️⃣ FEATURE ENGINEERING
# ===========================================================
# We’ll create synthetic weather/time features to simulate realistic ride demand predictors.
# Make sure your dataset has at least pickup_datetime or similar time columns.

if 'pickup_datetime' in data.columns:
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['year'] = data['pickup_datetime'].dt.year
    data['month'] = data['pickup_datetime'].dt.month
    data['day_of_week'] = data['pickup_datetime'].dt.dayofweek
    data['hour'] = data['pickup_datetime'].dt.hour
else:
    # fallback if dataset doesn’t contain datetime
    print("⚠️ pickup_datetime not found — generating synthetic time features.")
    np.random.seed(42)
    data['year'] = 2025
    data['month'] = np.random.randint(1, 13, len(data))
    data['day_of_week'] = np.random.randint(0, 7, len(data))
    data['hour'] = np.random.randint(0, 24, len(data))

# Add synthetic weather columns if not present
for col in ['temperature', 'humidity', 'wind_speed', 'weather_condition']:
    if col not in data.columns:
        if col == 'temperature':
            data[col] = np.random.uniform(20, 35, len(data))  # °C
        elif col == 'humidity':
            data[col] = np.random.uniform(40, 90, len(data))  # %
        elif col == 'wind_speed':
            data[col] = np.random.uniform(0, 20, len(data))   # km/h
        elif col == 'weather_condition':
            data[col] = np.random.randint(0, 5, len(data))    # 0=Clear, 1=Cloudy, etc.

# ===========================================================
# 3️⃣ TARGET VARIABLE
# ===========================================================
# If fare_amount exists, use it as a proxy for ride demand
if 'fare_amount' in data.columns:
    data = data[data['fare_amount'] > 0]
    y = data['fare_amount']  # predicting fare as ride demand proxy
else:
    # create synthetic target if not available
    y = np.random.uniform(50, 300, len(data))

# ===========================================================
# 4️⃣ SELECT FEATURES
# ===========================================================
features = ['year', 'month', 'day_of_week', 'hour',
            'temperature', 'humidity', 'wind_speed', 'weather_condition']

X = data[features].copy()

# ===========================================================
# 5️⃣ TRAIN TEST SPLIT
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split complete.")

# ===========================================================
# 6️⃣ SCALING
# ===========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================================================
# 7️⃣ MODEL TRAINING
# ===========================================================
print("🚀 Training XGBoost model...")
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train_scaled, y_train)
print("✅ Model training complete.")

# ===========================================================
# 8️⃣ MODEL EVALUATION
# ===========================================================
preds = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"📊 MAE: {mae:.2f}, R²: {r2:.2f}")

# ===========================================================
# 9️⃣ SAVE MODEL + SCALER
# ===========================================================
os.makedirs('model', exist_ok=True)
joblib.dump(model, os.path.join('model', 'ride_demand_model.pkl'))
joblib.dump(scaler, os.path.join('model', 'scaler.pkl'))
print("💾 Model and scaler saved successfully to /model directory.")

# ===========================================================
# ✅ DONE
# ===========================================================
print("🎯 Training process completed successfully! Ready for prediction.")
