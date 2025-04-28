# scripts/clean_data.py

import pandas as pd

def load_raw_data(filepath):
    """Load the raw Excel data"""
    df = pd.read_excel(filepath)
    return df

def clean_data(df):
    """Apply basic cleaning steps"""
    # Simple example: Drop missing values
    df = df.dropna()
    return df

def save_cleaned_data(df, filepath):
    """Save cleaned data to CSV"""
    df.to_csv(filepath, index=False)
