import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class PCAFeatureSelector:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.scaler = StandardScaler()
        self.selected_components = None

    def fit(self, X_train, y_train=None):
        # Normalize features for better convergence
        X_normalized = self.scaler.fit_transform(X_train)

        # Fit the PCA model
        self.pca.fit(X_normalized)

        # Determine the components to keep (those that explain significant variance)
        explained_variance = self.pca.explained_variance_ratio_
        self.selected_components = np.where(explained_variance > 0.01)[
            0
        ]  # Threshold can be adjusted

    def transform(self, X):
        # Check if fit has been called
        if self.selected_components is None:
            raise RuntimeError(
                "PCA selector has not been fitted before transformation performed."
            )

        # Transform features
        X_normalized = self.scaler.transform(X)
        X_pca = self.pca.transform(X_normalized)
        return X_pca[:, self.selected_components]

    def get_support(self):
        # Check if fit has been called
        if self.selected_components is None:
            raise RuntimeError(
                "PCA selector has not been fitted before calling get_support."
            )

        # Create a boolean mask with the same length as the number of components
        support_mask = np.zeros(len(self.pca.components_), dtype=bool)
        support_mask[self.selected_components] = True
        return support_mask

    def get_feature_rankings(self):
        if self.pca.components_ is None:
            raise RuntimeError("The PCA model has not been fitted yet.")
        component_contributions = np.abs(self.pca.components_).sum(axis=0)
        rankings = sorted(
            zip(self.feature_names, component_contributions),
            key=lambda x: x[1],
            reverse=True,
        )
        return rankings
