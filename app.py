import streamlit as st
import numpy as np
import pickle

st.title("Air Quality Prediction App")

try:
    model = pickle.load(open("aqi_model.pkl","rb"))
except:
    st.error("Model file not found! Make sure aqi_model.pkl is in the same folder.")

pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
no2 = st.number_input("NO2")
so2 = st.number_input("SO2")
co = st.number_input("CO")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")

if st.button("Predict AQI"):

    data = np.array([[pm25,pm10,no2,so2,co,temperature,humidity]])

    try:
        prediction = model.predict(data)
        st.success(f"Predicted AQI: {prediction[0]:.2f}")
    except:
        st.error("Prediction error. Check model file.")