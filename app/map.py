# app/map.py

import streamlit as st
import pydeck as pdk
import pandas as pd

def app():
    st.title("🗺️ Recommended Locations Map")
    data = pd.read_csv('data/processed/final_cleaned.csv')
    
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=data['latitude'].mean(),
            longitude=data['longitude'].mean(),
            zoom=10,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
               'ScatterplotLayer',
               data=data,
               get_position='[longitude, latitude]',
               get_color='[255, 0, 0, 160]',
               get_radius=300,
            ),
        ],
    ))
