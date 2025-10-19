# Ride Demand Forecasting — Likhith Tech Sol

This repository is a *ready-to-run* Flask + XGBoost project for forecasting ride demand. It's set up for easy local development and deployment to **Render**.
Branding: **Likhith Tech Sol**

---
## Contents
- `data/` — expected location for the downloaded NYC taxi CSV (user must download or use Kaggle API)
- `notebooks/` — Jupyter notebook for EDA & model training (optional)
- `model/` — saved model (`ride_demand_model.pkl`) will be stored here after training
- `app/` — Flask application and templates
- `train.py` — script to download (via Kaggle) and train the model (creates the model file)
- `requirements.txt` — Python dependencies
- `Procfile` — for Render (uses gunicorn)
- `.gitignore` — ignores virtualenv, dataset, model files

---
## Quick start (local)

### 1) Clone & enter
```bash
# after you unzip or git clone
cd ride-demand-forecasting-likhith-tech-sol
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Provide dataset (two options)

**Option A (recommended) — use Kaggle API**

1. Create a Kaggle account and get API credentials. Then set environment variables:
```bash
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_key
# On Windows (PowerShell)
setx KAGGLE_USERNAME "your_kaggle_username"
setx KAGGLE_KEY "your_kaggle_key"
```

2. Run the training script (it will download one month of NYC yellow taxi data and train the model):
```bash
python train.py --dataset kaggle --file yellow_tripdata_2023-01.csv
```

**Option B — manual download**

1. Download a CSV from Kaggle (e.g. `yellow_tripdata_2023-01.csv`) and place it into `data/`.
2. Run training:
```bash
python train.py --dataset local --file yellow_tripdata_2023-01.csv
```

### 3) Start the Flask app (after training finishes and model exists)
```bash
cd app
python app.py
# then open http://127.0.0.1:5000 in your browser
```

### 4) Deploy to Render
- Create a GitHub repo and push your project.
- Create a new Web Service on Render, connect to the repo.
- Build command: `pip install -r requirements.txt`
- Start command (Render will use `Procfile`): `gunicorn app.app:app`

---
## Notes & Tips
- Training uses `XGBoost` and aggregates ride counts by (day_of_week, hour) for a simple but strong baseline.
- You can extend features (borough, weather, holidays) in `train.py` and `notebooks/model_training.ipynb`.
- If you run into memory limits when training on a full month, sample or aggregate the CSV using `pandas` chunking.

---
Made with ❤️ by **Likhith Tech Sol**
