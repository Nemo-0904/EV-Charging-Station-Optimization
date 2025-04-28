# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETTINGS ---
st.set_page_config(
    page_title="EV Charging Station Optimization",
    page_icon="⚡",
    layout="wide"
)

# --- Load Model and Data ---
@st.cache_resource
def load_model():
    model = joblib.load('app/knn_model.pkl')  # Adjust if your model is somewhere else
    return model

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/final_cleaned.csv')  # Final cleaned data
    return df

model = load_model()
data = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Data Exploration", "Make Prediction", "About"])

# --- HOME PAGE ---
if page == "Home":
    st.title("🚗⚡ EV Charging Station Optimization")
    st.markdown("""
    Welcome to the EV Charging Station Optimization app!
    
    This tool predicts EV station availability based on location and vehicle type information.
    """)
    
    st.subheader("Sample of Dataset")
    st.dataframe(data.head())

# --- DATA EXPLORATION PAGE ---
elif page == "Data Exploration":
    st.title("📊 Data Exploration")
    
    st.subheader("Available Charging Slots Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['available'], bins=30, kde=True, ax=ax)
    plt.title("Distribution of Available Charging Slots")
    st.pyplot(fig)

    st.subheader("Vehicle Types Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='vehicle_type', data=data, ax=ax)
    plt.title("Vehicle Type Distribution")
    st.pyplot(fig)

    st.subheader("Latitude vs Longitude of Stations")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='longitude', y='latitude', data=data, hue='vehicle_type', palette='viridis', ax=ax)
    plt.title("Geographic Distribution of Charging Stations")
    st.pyplot(fig)

# --- MAKE PREDICTION PAGE ---
elif page == "Make Prediction":
    st.title("🔮 Predict Available Slots")
    
    st.subheader("Enter Station Details")
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input('Latitude', format="%.6f")
        longitude = st.number_input('Longitude', format="%.6f")
    with col2:
        vehicle_type = st.selectbox('Vehicle Type (Encoded)', options=[0, 1, 2, 3])
        duration = st.number_input('Duration (in seconds)', min_value=0)

    if st.button("Predict"):
        input_features = pd.DataFrame([{
            'latitude': latitude,
            'longitude': longitude,
            'vehicle_type': vehicle_type,
            'duration': duration
        }])
        prediction = model.predict(input_features)[0]
        st.success(f"🔋 Predicted Available Charging Slots: {int(prediction)}")

# --- ABOUT PAGE ---
elif page == "About":
    st.title("ℹ About This Project")
    
    st.markdown("""
    ### EV Charging Station Optimization
    
    This project helps in identifying and optimizing electric vehicle charging stations based on real-world geo-demographic and vehicle usage data.
    
    **Features used for prediction:**
    - Latitude
    - Longitude
    - Vehicle Type (encoded)
    - Charging Duration (seconds)
    
    **Developed By:**  
    - Atharva Sakpal  
    - Ramyaa Balasubramanian  
    - Shashin Vathode  
    
    **Built with ❤️ using Streamlit and Scikit-Learn.**
    """)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Made with ❤️ by your team.")
