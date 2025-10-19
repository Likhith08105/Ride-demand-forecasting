from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder='templates')

# ===========================================================
# Load model and scaler
# ===========================================================
MODEL_PATH = os.path.join('model', 'ride_demand_model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===========================================================
# Routes
# ===========================================================
@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user inputs
        year = float(request.form.get('year', 0))
        month = float(request.form.get('month', 0))
        day_of_week = float(request.form.get('day_of_week', 0))
        hour = float(request.form.get('hour', 0))
        temperature = float(request.form.get('temperature', 0))
        humidity = float(request.form.get('humidity', 0))
        wind_speed = float(request.form.get('wind_speed', 0))
        weather_condition = float(request.form.get('weather_condition', 0))

        # Combine and scale
        features = np.array([[year, month, day_of_week, hour, temperature, humidity, wind_speed, weather_condition]])
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        prediction = round(float(prediction), 2)

        return render_template('index.html', prediction_text=f"🚖 Predicted Ride Demand: {prediction} rides/hour")

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
