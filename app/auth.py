# app/auth.py

import streamlit as st

# Very simple user database
users = {}

def signup(username, password):
    if username in users:
        return False
    users[username] = password
    return True

def login(username, password):
    return users.get(username) == password
