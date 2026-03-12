import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "data", "air_quality_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "aqi_model.pkl")

# -----------------------------
# Load CSV and detect columns
# -----------------------------
def load_csv():
    if not os.path.exists(DATA_FILE):
        st.error(f"Data file '{DATA_FILE}' not found.")
        st.stop()
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip().replace(" ", "").upper() for c in df.columns]
    return df

# -----------------------------
# Train and save model
# -----------------------------
def train_and_save_model(df):
    # Identify pollutant columns (exclude AQI)
    required_cols = ['PM25','PM10','NO2','SO2','CO']
    X_cols = [c for c in required_cols if c in df.columns]
    
    if 'AQI' not in df.columns:
        st.error("CSV must contain the 'AQI' column.")
        st.stop()
    if not X_cols:
        st.error("CSV must contain at least one pollutant column among PM25, PM10, NO2, SO2, CO.")
        st.stop()
    
    X = df[X_cols]
    y = df['AQI']
    
    model = LinearRegression()
    model.fit(X, y)
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({'model': model, 'features': X_cols}, f)
    
    return model, X_cols

# -----------------------------
# Load or train model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    df = load_csv()
    if os.path.exists(MODEL_FILE):
        try:
            data = pickle.load(open(MODEL_FILE, "rb"))
            return data['model'], data['features'], df
        except:
            model, features = train_and_save_model(df)
            return model, features, df
    else:
        model, features = train_and_save_model(df)
        return model, features, df

model, features, data = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AQI Prediction App", layout="centered")
st.title("🌫️ Air Quality Index (AQI) Prediction")
st.write("Enter pollutant values to predict AQI:")

# Dynamic inputs based on available columns
input_values = {}
cols = st.columns(2)
for i, feature in enumerate(features):
    with cols[i % 2]:
        if feature == 'PM25':
            input_values['PM25'] = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=10.0)
        elif feature == 'PM10':
            input_values['PM10'] = st.number_input("PM10 (µg/m³)", min_value=0.0, value=20.0)
        elif feature == 'NO2':
            input_values['NO2'] = st.number_input("NO₂ (ppb)", min_value=0.0, value=15.0)
        elif feature == 'SO2':
            input_values['SO2'] = st.number_input("SO₂ (ppb)", min_value=0.0, value=5.0)
        elif feature == 'CO':
            input_values['CO'] = st.number_input("CO (ppm)", min_value=0.0, value=0.5)

# -----------------------------
# AQI Category
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
# Predict
# -----------------------------
if st.button("Predict AQI"):
    input_array = np.array([[input_values[f] for f in features]])
    prediction = model.predict(input_array)
    aqi_value = round(prediction[0], 2)
    
    category, color = get_aqi_category(aqi_value)
    
    st.markdown(f"### Predicted AQI: {aqi_value}")
    st.markdown(f"<h3 style='color:{color}'>Category: {category}</h3>", unsafe_allow_html=True)
