import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import os

# ------------------------
# Project paths
# ------------------------
MODEL_FILE = "aqi_model.pkl"
DATA_FILE = "data/air_quality.csv"

# ------------------------
# Function to train and save model
# ------------------------
def train_and_save_model():
    if not os.path.exists(DATA_FILE):
        st.error(f"Data file '{DATA_FILE}' not found. Make sure your CSV exists.")
        st.stop()
    
    data = pd.read_csv(DATA_FILE)
    
    required_columns = ['PM2.5','PM10','NO2','SO2','CO','AQI']
    if not all(col in data.columns for col in required_columns):
        st.error(f"CSV must contain columns: {required_columns}")
        st.stop()
    
    X = data[['PM2.5','PM10','NO2','SO2','CO']]
    y = data['AQI']
    
    model = LinearRegression()
    model.fit(X, y)
    
    with open(MODEL_FILE,"wb") as f:
        pickle.dump(model,f)
    
    return model

# ------------------------
# Load model (train if missing)
# ------------------------
if os.path.exists(MODEL_FILE):
    try:
        model = pickle.load(open(MODEL_FILE,"rb"))
    except:
        st.warning("Existing model file corrupted. Re-training model...")
        model = train_and_save_model()
else:
    st.info("Model file not found. Training model now...")
    model = train_and_save_model()

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="AQI Prediction App", layout="centered")
st.title("🌫️ Air Quality Index (AQI) Prediction")

st.write("Enter pollutant values to predict AQI:")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, value=10.0)
    pm10 = st.number_input("PM10 (µg/m³)", 0.0, value=20.0)
    no2 = st.number_input("NO₂ (ppb)", 0.0, value=15.0)

with col2:
    so2 = st.number_input("SO₂ (ppb)", 0.0, value=5.0)
    co = st.number_input("CO (ppm)", 0.0, value=0.5)

# ------------------------
# AQI Category
# ------------------------
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

# ------------------------
# Predict Button
# ------------------------
if st.button("Predict AQI"):
    input_data = np.array([[pm25, pm10, no2, so2, co]])
    prediction = model.predict(input_data)
    aqi_value = round(prediction[0],2)
    
    category, color = get_aqi_category(aqi_value)
    
    st.markdown(f"### Predicted AQI: {aqi_value}")
    st.markdown(f"<h3 style='color:{color}'>Category: {category}</h3>", unsafe_allow_html=True)
