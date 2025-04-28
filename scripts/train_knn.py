# scripts/train_knn.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import joblib

# Load final cleaned dataset
df = pd.read_csv('data/processed/final_cleaned.csv')

# Select Features
features = ['latitude', 'longitude', 'vehicle_type', 'duration']
X = df[features]
y = df['available']  # your prediction target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN Regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Save model
joblib.dump(knn, 'app/knn_model.pkl')

print("✅ KNN Regressor model trained and saved as 'knn_model.pkl'")