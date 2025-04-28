# app/main.py

import streamlit as st
import pandas as pd
import joblib

st.title('🚗⚡ EV Charging Station Optimization - KNN Model')

@st.cache_data
def load_model():
    """Load the trained KNN model"""
    model = joblib.load('app/knn_model.pkl')
    return model

model = load_model()

st.subheader('📋 Input Data for Prediction')

# ✅ INPUTS matching your KNN model
latitude = st.number_input('Latitude', format="%.6f")
longitude = st.number_input('Longitude', format="%.6f")
vehicle_type = st.selectbox('Vehicle Type (Encoded)', options=[0, 1, 2, 3])
duration = st.number_input('Duration (seconds)', min_value=0)

if st.button('Predict Availability'):
    # ✅ Prepare input data for KNN
    input_data = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'vehicle_type': [vehicle_type],
        'duration': [duration]
    })

    # ✅ Predict using KNN model
    prediction = model.predict(input_data)

    # ✅ Show result
    st.success(f"🔋 Predicted Available Charging Slots: {int(prediction[0])}")
