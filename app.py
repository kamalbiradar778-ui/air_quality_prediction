import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import os

# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "data", "air_quality_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "aqi_model.pkl")

# -----------------------------
# Flexible column mapping
# -----------------------------
COLUMN_MAP = {
    'PM25': ['pm25', 'pm2.5', 'pm_25', 'pm_2_5'],
    'PM10': ['pm10', 'pm_10'],
    'NO2': ['no2', 'no_2'],
    'SO2': ['so2', 'so_2'],
    'CO': ['co'],
    'AQI': ['aqi']
}

def normalize_and_map_columns(df):
    # Lowercase and remove spaces/units
    df.columns = [c.strip().lower().replace(" ", "").replace("µg/m³","").replace("ppb","").replace("ppm","") for c in df.columns]
    mapped_cols = {}
    for target, variants in COLUMN_MAP.items():
        for var in variants:
            if var in df.columns:
                mapped_cols[target] = var
                break
    return df, mapped_cols

# -----------------------------
# Function to train and save model
# -----------------------------
def train_and_save_model():
    if not os.path.exists(DATA_FILE):
        st.error(f"Data file '{DATA_FILE}' not found.")
        st.stop()
    
    data = pd.read_csv(DATA_FILE)
    data, mapped_cols = normalize_and_map_columns(data)
    
    if len(mapped_cols) < len(COLUMN_MAP):
        st.error(f"CSV missing required columns. Detected: {list(mapped_cols.keys())}")
        st.stop()
    
    X = data[[mapped_cols['PM25'], mapped_cols['PM10'], mapped_cols['NO2'], mapped_cols['SO2'], mapped_cols['CO']]]
    y = data[mapped_cols['AQI']]
    
    model = LinearRegression()
    model.fit(X, y)
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    
    return model

# -----------------------------
# Load or train model (cached)
# -----------------------------
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        try:
            return pickle.load(open(MODEL_FILE, "rb"))
        except:
            return train_and_save_model()
    else:
        return train_and_save_model()

model = load_or_train_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AQI Prediction App", layout="centered")
st.title("🌫️ Air Quality Index (AQI) Prediction")
st.write("Enter pollutant values to predict AQI:")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=10.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=20.0)
    no2 = st.number_input("NO₂ (ppb)", min_value=0.0, value=15.0)

with col2:
    so2 = st.number_input("SO₂ (ppb)", min_value=0.0, value=5.0)
    co = st.number_input("CO (ppm)", min_value=0.0, value=0.5)

# -----------------------------
# AQI Category Function
# -----------------------------
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#009966"
    elif aqi <= 100:
        return "Moderate", "#ffde33"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff9933"
    elif aqi <= 200:
        return "Unhealthy", "#cc0033"
    elif aqi <= 300:
        return "Very Unhealthy", "#660099"
    else:
        return "Hazardous", "#7e0023"

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict AQI"):
    input_data = np.array([[pm25, pm10, no2, so2, co]])
    prediction = model.predict(input_data)
    aqi_value = round(prediction[0], 2)
    
    category, color = get_aqi_category(aqi_value)
    
    st.markdown(f"### Predicted AQI: {aqi_value}")
    st.markdown(f"<h3 style='color:{color}'>Category: {category}</h3>", unsafe_allow_html=True)
