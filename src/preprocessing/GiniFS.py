import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

class GiniImportanceSelector:
    def __init__(self, n_estimators=100, random_state=42):
        self.rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.scaler = StandardScaler()
        self.selected_features = None
        self.importances = None

    def fit(self, X_train, y_train):
        # Normalize features for better convergence
        X_normalized = self.scaler.fit_transform(X_train)

        # Fit the RandomForest model
        self.rf.fit(X_normalized, y_train)

        # Get feature importances
        self.importances = self.rf.feature_importances_

        # Determine the features to keep (those with significant importance)
        self.selected_features = np.where(self.importances > np.percentile(self.importances, 75))[0]  # Threshold at 75th percentile

    def transform(self, X):
        # Check if fit has been called
        if self.selected_features is None:
            raise RuntimeError("GiniImportance selector has not been fitted before transformation performed.")

        # Transform features
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