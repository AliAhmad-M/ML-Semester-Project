import streamlit as st
import pandas as pd
import joblib

# Load model artifacts
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("../output/models/xgb_model.pkl")
    model = artifacts["model"]
    scaler = artifacts["scaler_X"]
    feature_names = artifacts["feature_names"]
    return model, scaler, feature_names

xgb_final, scaler_final, feature_names = load_artifacts()

# Load training data (for columns)
@st.cache_data
def load_training_data():
    return pd.read_csv("../output/data_merged_cleaned.csv")

df_original = load_training_data()

# Forecast input data
forecast_data = pd.DataFrame(
    [
        ["15/12/2025",None,23,15,7,12,10,8,65,52,40,5,3,1,30.05,30.00,29.95,12,0,16,0,0,0,1,0,0,0,1],
        ["16/12/2025",None,24,16,8,13,11,9,62,50,38,6,4,2,30.06,30.01,29.96,12,0,16,0,0,0,1,0,0,0,1],
        ["17/12/2025",None,24,17,9,13,11,9,60,48,35,6,4,2,30.04,29.99,29.94,12,0,15,0,0,0,1,0,0,0,1],
        ["18/12/2025",None,25,18,10,14,12,10,58,46,33,6,4,2,30.03,29.98,29.93,12,0,15,0,0,0,1,0,0,0,1],
        ["19/12/2025",None,25,18,10,14,12,10,55,44,30,6,4,2,30.02,29.97,29.92,12,0,15,0,0,0,1,0,0,0,1],
        ["20/12/2025",None,26,19,11,15,13,11,53,42,28,7,5,3,30.01,29.96,29.91,12,0,14,0,0,0,1,0,0,0,1],
        ["21/12/2025",None,26,20,11,15,13,11,50,40,25,7,5,3,30.00,29.95,29.90,12,0,14,0,0,0,1,0,0,0,1]
    ],
    columns=df_original.columns
)

forecast_data["date"] = pd.to_datetime(forecast_data["date"], format="%d/%m/%Y")

# Feature engineering
X_forecast = forecast_data.drop(columns=["aqi_pm2.5", "date"])
X_forecast_fe = X_forecast.copy()
X_forecast_fe["temp_avg_lag1"] = X_forecast["temp_avg_c"]
X_forecast_fe["wind_avg_lag1"] = X_forecast["wind_speed_avg_mph"]
X_forecast_fe["humidity_avg_lag1"] = X_forecast["humidity_avg_percent"]
X_forecast_fe["wind_temp_interaction"] = X_forecast_fe["wind_speed_avg_mph"] * X_forecast_fe["temp_min_c"]
X_forecast_fe["wind_humidity_interaction"] = X_forecast_fe["wind_speed_avg_mph"] * X_forecast_fe["humidity_avg_percent"]
X_forecast_fe["high_risk_month"] = X_forecast_fe["month"].isin([1,2,10,11]).astype(int)
X_forecast_fe["early_winter"] = X_forecast_fe["month"].isin([11,12]).astype(int)
X_forecast_fe["wind_3day_avg"] = X_forecast_fe["wind_speed_avg_mph"]
X_forecast_fe["temp_3day_avg"] = X_forecast_fe["temp_avg_c"]
X_forecast_fe = X_forecast_fe.fillna(X_forecast_fe.mean())

# Scale and predict
X_forecast_scaled = scaler_final.transform(X_forecast_fe[feature_names])
forecast_data["Predicted_AQI"] = xgb_final.predict(X_forecast_scaled)

# AQI categories and colors
def aqi_condition(aqi):
    if aqi <= 50:
        return "Good", "#2ECC71"        # green
    elif aqi <= 100:
        return "Moderate", "#F1C40F"    # yellow
    elif aqi <= 150:
        return "Unhealthy S", "#E67E22" # orange
    elif aqi <= 200:
        return "Unhealthy", "#E74C3C"   # red
    elif aqi <= 300:
        return "Very Unhealthy", "#8E44AD" # purple
    else:
        return "Hazardous", "#7F0000"   # maroon

forecast_data["AQI_condition"], forecast_data["color"] = zip(*forecast_data["Predicted_AQI"].map(aqi_condition))

# Smog risk levels
def smog_risk_level(aqi, winter_flag):
    if winter_flag:
        if aqi > 200:
            return "Very High Smog Risk"
        elif aqi > 150:
            return "Moderate-High Smog Risk"
    return "None-Low Smog Risk"

forecast_data["Smog_Risk"] = [
    smog_risk_level(aqi, winter) for aqi, winter in zip(forecast_data["Predicted_AQI"], X_forecast_fe["early_winter"])
]

# Streamlit dashboard
st.set_page_config(page_title="AQI & Smog Predictor", layout="wide")
st.title("AQI & Smog Forecast Dashboard")
st.header("Forecasts:")
cols = st.columns(len(forecast_data))
cards_per_row = 3
n_days = len(forecast_data)

for i in range(0, n_days, cards_per_row):
    cols = st.columns(min(cards_per_row, n_days - i))
    for j, col in enumerate(cols):
        row = forecast_data.iloc[i + j]
        col.metric(
            label=row.date.strftime("%d %b %Y"),
            value=f"{row.Predicted_AQI:.1f} ({row.AQI_condition})",
            delta=row.Smog_Risk
        )
        col.markdown(
            f"<div style='background-color:{row.color};height:5px;width:100%;border-radius:4px'></div>",
            unsafe_allow_html=True
        )