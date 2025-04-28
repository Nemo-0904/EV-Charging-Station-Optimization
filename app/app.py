# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EV Charging Station Optimization",
    page_icon="⚡",
    layout="wide"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #ffffff, #e0f7ff);
}
.main {
    background: linear-gradient(to right, #ffffff, #e0f7ff);
    padding: 2rem;
    border-radius: 8px;
}
/* Headings and Text */
h1, h2, h3, h4, h5, h6, p, li {
    color: #000080;
    font-family: 'Segoe UI', sans-serif;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffffff, #f9f9f9);
    color: #000080;
}
/* Button Styles */
div.stButton > button {
    background-color: white;
    color: #004080;
    border: 1px solid #004080;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 16px;
}
div.stButton > button:hover {
    background-color: #e6f0ff;
}
/* Dataframe Border */
.stDataFrame {
    background-color: white;
    border: 1px solid #cce6ff;
    border-radius: 8px;
}
/* Fix Dropdown (selectbox) Background and Text */
div[data-baseweb="select"] {
    background-color: white !important;
}
div[data-baseweb="select"] span {
    color: #000080 !important; /* Navy Blue Text inside dropdown */
    font-weight: bold;
}
div[data-baseweb="select"] div[role="button"] {
    color: #000080 !important; /* Selected option text color */
}
</style>

""", unsafe_allow_html=True)

# --- LOAD MODEL AND DATA ---
@st.cache_resource
def load_model():
    model = joblib.load('app/knn_model.pkl')
    return model

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/final_cleaned.csv')
    return df

model = load_model()
data = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page:", ["Home", "Find Places", "How We Made It", "About"])

# --- HOME PAGE ---
if page == "Home":
    st.title("Electric Vehicle Charging Station Optimization")
    st.subheader("Project Overview")
    st.write("""
        With the rise of electric vehicles (EVs), it is critical to optimize the placement of EV charging stations.
        Our project uses geospatial and demographic analysis combined with machine learning to identify optimal charging locations and predict station availability.
    """)
    st.subheader("Key Objectives")
    st.write("""
    - Analyze EV station availability based on location and vehicle types.
    - Predict the number of available charging slots using a K-Nearest Neighbors (KNN) model.
    - Provide an interactive map to visualize recommended places.
    """)
    st.subheader("Solution Approach")
    st.write("""
    - Collect and clean real-world data of charging stations.
    - Engineer features such as latitude, longitude, vehicle type, and duration.
    - Build a KNN model to predict slot availability.
    - Deploy an interactive web application using Streamlit.
    """)

# --- FIND PLACES PAGE ---
elif page == "Find Places":
    st.title("Find Places for EV Charging Stations")
    st.subheader("Input Your Details")

    # Dataset Reference
    with st.expander("Click here to view Sample Dataset for Reference"):
        selected_columns = ['latitude', 'longitude', 'vehicle_type']
        st.dataframe(data[selected_columns].head(20), use_container_width=True)

    # User Form
    with st.form(key='find_places_form'):
        latitude = st.number_input('Enter Latitude', format="%.6f")
        longitude = st.number_input('Enter Longitude', format="%.6f")
        vehicle_type = st.selectbox(
            'Select Vehicle Type',
            options=list(range(0, 9)),
            format_func=lambda x: f"Vehicle Type {x}"
        )
        duration = st.number_input('Enter Charging Duration (in seconds)', min_value=0)
        
        submit_button = st.form_submit_button(label="Find Place")

    if submit_button:
        with st.spinner('Processing your request...'):
            input_features = pd.DataFrame([{
                'latitude': latitude,
                'longitude': longitude,
                'vehicle_type': vehicle_type,
                'duration': duration
            }])

            prediction = model.predict(input_features)[0]

            st.subheader("Prediction Result:")
            if prediction > 0:
                st.success(f"Charging Slot is Likely Available ({int(prediction)} slots predicted)")
            else:
                st.error("Charging Slot is Likely Not Available at this Location")

            # Show location
            st.subheader("Location on Map")
            map_data = pd.DataFrame({
                'latitude': [latitude],
                'longitude': [longitude]
            })
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=12,
                    pitch=50,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_data,
                        get_position='[longitude, latitude]',
                        get_color='[0, 128, 255, 160]',
                        get_radius=500,
                    ),
                ],
            ))

# --- HOW WE MADE IT PAGE ---
elif page == "How We Made It":
    st.title("How We Made This Project")
    st.subheader("Problem Statement")
    st.write("""
        The goal is to predict EV charging station availability and optimize station planning 
        based on geographical and vehicle usage data.
    """)
    st.subheader("Steps Followed")
    st.write("""
    - **Data Collection**: Gathered datasets containing EV station details.
    - **Data Cleaning**: Handled missing values and standardized data.
    - **Feature Engineering**: Extracted latitude, longitude, vehicle type, and charging duration as key features.
    - **Model Building**: Used K-Nearest Neighbors (KNN) for predicting station availability.
    - **Deployment**: Developed an interactive Streamlit web app.
    """)
    st.subheader("Dataset Parameters Used")
    st.write("""
    - **Latitude**
    - **Longitude**
    - **Vehicle Type (Encoded)**
    - **Charging Duration (in seconds)**
    """)
    st.subheader("Machine Learning Model")
    st.write("""
    - **Model**: K-Nearest Neighbors (KNN) Regressor
    - **Hyperparameter**: n_neighbors = 5
    - **Evaluation Metrics**: Mean Squared Error (MSE), R-Squared Score
    """)
    st.subheader("Development Environment")
    st.write("""
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - Pydeck
    """)

# --- ABOUT PAGE ---
elif page == "About":
    st.title("About This Project")
    st.write("""
        This project was undertaken as part of the **Data Science Laboratory** coursework. 
        It showcases the use of data preprocessing, machine learning, and interactive dashboards 
        for a real-world electric mobility problem.
    """)
    st.subheader("Team Members")
    st.write("""
    - NITHYA CHERALA
    - SHREYA CHAUDHARI
    """)
    st.subheader("Tools & Technologies Used")
    st.write("""
    - Python
    - Streamlit
    - Scikit-Learn
    - Pandas
    - Pydeck
    """)

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info("Developed as part of Data Science Lab Project, 2025.")
