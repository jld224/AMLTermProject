import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

class GiniImportanceSelector:
    def __init__(self, n_estimators=100, random_state=42, importance_threshold=0.01):
        self.rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.scaler = StandardScaler()
        self.importance_threshold = importance_threshold
        self.selected_features = None
        self.importances = None

    def fit(self, X_train, y_train):
        X_normalized = self.scaler.fit_transform(X_train)
        self.rf.fit(X_normalized, y_train)
        self.importances = self.rf.feature_importances_
        # Select features based on importance threshold
        self.selected_features = [i for i, imp in enumerate(self.importances) if imp >= self.importance_threshold]

    def transform(self, X):
        X_normalized = self.scaler.transform(X)
        return X_normalized[:, self.selected_features]

    def get_support(self):
        # Check if fit has been called
        if self.selected_features is None:
            raise RuntimeError("GiniImportance selector has not been fitted before calling get_support.")
        
        # Create a boolean mask with the same length as the number of features
        support_mask = np.zeros(len(self.importances), dtype=bool)
        support_mask[self.selected_features] = True
        return support_mask

    def save_importances(self):
        # Ensure the 'results' directory exists
        os.makedirs('results', exist_ok=True)

        # Save the feature importances to a CSV file
        importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.importances
        })
        importances_df.sort_values(by='Importance', ascending=False, inplace=True)
        importances_df.to_csv('results/feature_importances.csv', index=False)