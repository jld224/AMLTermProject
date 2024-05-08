import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class PCAFeatureSelector:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()

    def fit(self, X_train):
        X_normalized = self.scaler.fit_transform(X_train)
        # Calculate the appropriate number of components if a percentage is provided
        if isinstance(self.n_components, str) and '%' in self.n_components:
            percentage = float(self.n_components.strip('%')) / 100
            n_components_adjusted = int(X_train.shape[1] * percentage)
        else:
            n_components_adjusted = self.n_components
        
        # Ensure n_components does not exceed number of features
        n_components_adjusted = min(X_train.shape[1], n_components_adjusted)
        
        self.pca = PCA(n_components=n_components_adjusted)
        self.pca.fit(X_normalized)

    def transform(self, X):
        X_normalized = self.scaler.transform(X)
        return self.pca.transform(X_normalized)

    def get_support(self):
        # This method should logically define how to determine 'support' for PCA, which isn't typical
        # For PCA, 'support' might mean components selected based on a threshold of explained variance
        # Here, simply assume all components are selected
        support_mask = np.ones(self.pca.n_components_, dtype=bool)
        return support_mask

    def save_components(self):
        # Ensure the 'results' directory exists
        os.makedirs('results', exist_ok=True)

        # Save the PCA components to a CSV file
        components = pd.DataFrame(self.pca.components_, columns=[f'feature_{i}' for i in range(self.pca.components_.shape[1])])
        components.to_csv('results/pca_components.csv', index=False)