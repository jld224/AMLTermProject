import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


class GiniImportanceSelector:
    def __init__(self, n_estimators=100, random_state=42):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
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
        self.selected_features = np.where(
            self.importances > np.percentile(self.importances, 75)
        )[
            0
        ]  # Threshold at 75th percentile

    def transform(self, X):
        # Check if fit has been called
        if self.selected_features is None:
            raise RuntimeError(
                "GiniImportance selector has not been fitted before transformation performed."
            )

        # Transform features
        X_normalized = self.scaler.transform(X)
        return X_normalized[:, self.selected_features]

    def get_support(self):
        # Check if fit has been called
        if self.selected_features is None:
            raise RuntimeError(
                "GiniImportance selector has not been fitted before calling get_support."
            )

        # Create a boolean mask with the same length as the number of features
        support_mask = np.zeros(len(self.importances), dtype=bool)
        support_mask[self.selected_features] = True
        return support_mask

    def get_feature_ranking(self):
        if self.importances is None:
            raise RuntimeError("The model has not been fitted yet.")
        rankings = sorted(
            zip(self.feature_names, self.importances), key=lambda x: x[1], reverse=True
        )
        return rankings
