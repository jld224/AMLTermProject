import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

directory = 'bargraphs'
if not os.path.exists(directory):
    os.makedirs(directory)

# Load your dataset
data = pd.read_csv('../../../data/processed.csv')

# Extract features and target variable
X = data.drop(columns=['η / mPa s', 'SMILES', 'MaxAbsEStateIndex'])
y = data['η / mPa s']

# Apply one-hot encoding for categorical variables
categorical_cols = ['Cationic family', 'Anionic family']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_cols)], remainder='passthrough')
X_encoded = ct.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train a RandomForestRegressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# Get feature importances
importances = rf.feature_importances_

# Get feature names
feature_names = np.array(ct.get_feature_names_out())

# Sort feature importances and feature names
indices = np.argsort(importances)[::-1]
sorted_importances = importances[indices]
sorted_feature_names = feature_names[indices]

# Plotting
plt.figure(figsize=(24, 6))
plt.bar(range(X_scaled.shape[1]), sorted_importances, align='center')
plt.xticks(range(X_scaled.shape[1]), sorted_feature_names, rotation=90)
plt.xlabel('Features')
plt.ylabel('Gini Importance')
plt.title('Feature Importance using Gini Importance')
plt.tight_layout()
plt.savefig(os.path.join(directory, 'feature_importance.svg'))
plt.close()

# Plotting Top 10
plt.figure(figsize=(8, 5))
plt.bar(range(10), sorted_importances[:10], align='center')
plt.xticks(range(10), sorted_feature_names[:10], rotation=90)
plt.xlabel('Features')
plt.ylabel('Gini Importance')
plt.title('Top 10 Feature Importance using Gini Importance')
plt.tight_layout()
plt.savefig(os.path.join(directory, 'top_10_feature_importance.svg'))
plt.close()
