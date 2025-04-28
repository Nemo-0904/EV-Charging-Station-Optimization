# app/signup.py

import streamlit as st
from app.auth import signup

def app():
    st.title("📝 Signup")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    if st.button("Signup"):
        if signup(username, password):
            st.success("Account created successfully!")
        else:
            st.error("Username already exists.")


