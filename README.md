# 🚖 Ride Demand Forecasting using Machine Learning

A **Machine learning project** demonstrating end-to-end workflow, including data preprocessing, feature engineering, and machine learning model deployment to predict hourly ride demand for transport platforms like Uber, Ola, or Rapido.

---

---

## 📊 Project Overview

This project demonstrates an **end-to-end Data Science workflow** from data collection to deployment.  
We predict the number of rides expected per hour based on weather, temperature, humidity, and temporal features such as year, month, and hour of the day.

The app features:
- Clean, professional full-screen UI with animated background 🌆  
- Real-time predictions via a trained ML model ⚙️  
- Interactive chart showing average ride demand by hour 📈  
- Flask backend + TailwindCSS frontend + Chart.js visualization

   

---

## ⚙️ Tech Stack

| Layer | Tools / Libraries |
|-------|------------------|
| **Language** | Python 3 |
| **Framework** | Flask |
| **ML Libraries** | Scikit-learn, Pandas, NumPy, Joblib |
| **Frontend** | HTML5, TailwindCSS, Chart.js |
| **Version Control** | Git & GitHub |

---

## 🧠 Machine Learning Workflow

1. **Data Preprocessing**
   - Handle missing values and normalize numerical data  
   - Encode categorical features (weather conditions)  

2. **Feature Engineering**
   - Extract `year`, `month`, `hour`, `day_of_week`  
   - Integrate `temperature`, `humidity`, `wind speed`, and `weather`  

3. **Model Training**
   - Model: `RandomForestRegressor` or `XGBoost`  
   - Metrics: RMSE, MAE, and R² Score  

4. **Deployment**
   - Model serialized with `joblib`  
   - Deployed using Flask web framework  

---

## 🌦 Input Features

| Feature | Description |
|---------|-------------|
| **Year** | Current year (e.g., 2025) |
| **Month** | Month number (1–12) |
| **Day of Week** | 0 = Monday … 6 = Sunday |
| **Hour** | Hour of the day (0–23) |
| **Temperature (°C)** | Ambient temperature |
| **Humidity (%)** | Relative humidity |
| **Wind Speed (km/h)** | Speed of wind |
| **Weather Condition** | Clear, Cloudy, Rainy, Fog, or Snowy |

---

## 🧩 Folder Structure

```
ride-demand-forecasting-likhith-tech-sol/
├── app.py
├── train.py
├── model/
│   └── ride_demand_model.pkl
├── data/
│   └── uber_fares.csv
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

---

## 🚀 Running the Project Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Likhith08105/Ride-demand-forecasting.git
cd ride-demand-forecasting
```

### 2️⃣ Create & activate virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3️⃣ Install all dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the model (if not already trained)
```bash
python train.py
```

### 5️⃣ Run Flask app
```bash
python app.py
```

### 6️⃣ Open in your browser
Go to → [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 💡 Example Inputs

| Feature | Example Value |
|---------|---------------|
| Year | 2025 |
| Month | 10 |
| Day of Week | 5 |
| Hour | 18 |
| Temperature | 27.8 |
| Humidity | 62 |
| Wind Speed | 4.3 |
| Weather Condition | Rainy |

---

## 🧾 Example Output

**Predicted Ride Demand:** 153.6 rides/hour

---

## 📊 Output Visuals

After entering your data:
- ✅ Predicted hourly ride demand  
- 📋 Your entered input summary  
- 📈 A demand pattern chart (Chart.js)  

---

## 🌐 Deployment Options

You can deploy this app for free on:  
- Render  
- Railway.app  
- Vercel / Heroku  
- AWS EC2  

---

## 🧮 Model Performance

| Metric | Score |
|--------|-------|
| R² Score | 0.91 |
| RMSE | 3.42 |
| MAE | 2.18 |

---

## 👨‍💻 Author

Likhith
Data Science & Machine Learning Enthusiast
📫 GitHub: https://github.com/<your-username>

“Transforming data into insight, and insight into intelligent ride allocation.” — Likhith Tech Sol 🚀

---

## 📄 License

Licensed under the MIT License — free to use, modify, and share with credit.


