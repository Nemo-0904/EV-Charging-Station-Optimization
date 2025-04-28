# app/utils.py

import pandas as pd
import joblib

# Load final cleaned data
data = pd.read_csv('data/processed/final_cleaned.csv')

# Load KNN model
knn_model = joblib.load('app/knn_model.pkl')

def recommend_locations(input_features, n_recommendations=3):
    distances, indices = knn_model.kneighbors([input_features])
    recommendations = data.iloc[indices[0]]
    return recommendations
