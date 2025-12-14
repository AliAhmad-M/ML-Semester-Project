# Lahore AQI & Smog Prediction

This repository provides a complete workflow for predicting Air Quality Index (AQI) and assessing smog risk in Lahore. It includes data processing, model training, and a Streamlit dashboard for visualization.

## Project Structure
- The root directory contains data preprocessing and model training notebook
- The `input` directory contains our raw data files
- The `output` directory contains our processed data and saved models
- The `dashboard` directory contains our 'proof of concept' dashboard

## Saved Models
- `xgb_model.pkl` - Our best saved XGBoost model with scaler and feature names.
- Data for other models saved in the `output/models` directory

## Running the Dashboard

1. Ensure all dependencies are installed (`pandas`, `streamlit`, `joblib`, `scikit-learn`, etc.).
2. Run the dashboard using:

```bash
cd dashboard
python -m streamlit run dashboard/dashboard.py
```
