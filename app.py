import streamlit as st
import pickle
import numpy as np

# ------------------------
# Load the trained model
# ------------------------
try:
    model = pickle.load(open("aqi_model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model file 'aqi_model.pkl' not found. Make sure it is in the same folder as app.py.")
    st.stop()

# ------------------------
# App title
# ------------------------
st.set_page_config(page_title="Air Quality Index (AQI) Predictor", layout="centered")
st.title("🌫️ Air Quality Index (AQI) Prediction App")
st.write("Enter the pollutant concentrations to predict AQI.")

# ------------------------
# User inputs
# ------------------------
col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=10.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=20.0)
    no2 = st.number_input("NO₂ (ppb)", min_value=0.0, value=15.0)

with col2:
    so2 = st.number_input("SO₂ (ppb)", min_value=0.0, value=5.0)
    co = st.number_input("CO (ppm)", min_value=0.0, value=0.5)
    o3 = st.number_input("O₃ (ppb)", min_value=0.0, value=10.0)

# ------------------------
# AQI Categories
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
    # Prepare input
    input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
    
    # Make prediction
    prediction = model.predict(input_data)
    aqi_value = round(prediction[0], 2)
    
    # Get category
    category, color = get_aqi_category(aqi_value)
    
    # Display result
    st.markdown(f"### Predicted AQI: {aqi_value}")
    st.markdown(f"<h3 style='color:{color}'>Category: {category}</h3>", unsafe_allow_html=True)
