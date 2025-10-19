# 🚖 Ride Demand Forecasting using Machine Learning

A **Flask-based Machine Learning web app** that predicts hourly ride demand for services like Rapido, Ola, or Uber using environmental and time-based features such as temperature, humidity, wind speed, and weather conditions.

---

## 📊 Project Overview
The goal of this project is to build a **data-driven model** capable of forecasting real-time ride demand based on historical ride patterns and environmental factors.  
It demonstrates an **end-to-end Data Science workflow**, covering:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Building (using Scikit-Learn)
- Flask Integration for Web Deployment
- Frontend UI built using HTML + Tailwind CSS + Chart.js

---

## ⚙️ Tech Stack
| Layer | Tools & Libraries |
|-------|-------------------|
| **Language** | Python 3 |
| **Framework** | Flask |
| **ML Libraries** | Pandas, NumPy, Scikit-learn, Joblib |
| **Visualization** | Chart.js, Matplotlib |
| **Frontend** | HTML5, TailwindCSS |
| **Version Control** | Git & GitHub |

---

## 🧠 Machine Learning Pipeline

1. **Data Preprocessing**
   - Handling missing values  
   - Encoding categorical features (e.g., weather conditions)  
   - Scaling numerical features using `MinMaxScaler`  

2. **Feature Engineering**
   - Extracted time features: year, month, hour, day_of_week  
   - Integrated weather & environment factors like temperature, humidity, wind speed  

3. **Model Training**
   - Algorithm: `RandomForestRegressor` (tuned via GridSearchCV)  
   - Evaluated on RMSE, MAE, and R² metrics  

4. **Model Deployment**
   - Trained model serialized using `joblib`  
   - Flask app serves real-time predictions through a web interface  

---

## 🌦 Input Features

| Feature | Description |
|----------|-------------|
| **Year** | Current year (e.g., 2025) |
| **Month** | Month number (1–12) |
| **Day of Week** | 0 = Monday … 6 = Sunday |
| **Hour** | Hour of the day (0–23) |
| **Temperature (°C)** | Ambient temperature |
| **Humidity (%)** | Relative humidity |
| **Wind Speed (km/h)** | Speed of wind |
| **Weather Condition** | Clear, Cloudy, Rainy, Snowy, or Fog |

---

## 🧩 Project Structure

