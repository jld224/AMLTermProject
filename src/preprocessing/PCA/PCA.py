import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

directory = 'biplots'
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

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Define a function to plot biplot
def myplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c='lightgray')
    colors = ['green', 'cyan', 'blue', 'brown', 'purple']
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color=colors[i % 5], ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color=colors[i % 5], ha='center', va='center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

# Save each biplot as a separate SVG file
num_plots = len(X.columns) // 4 + (len(X.columns) % 4 > 0)
for i in range(num_plots):
    start_idx = i * 4
    end_idx = min((i + 1) * 4, len(X.columns))
    plt.figure(figsize=(8, 6))
    myplot(X_pca[:, 0:2], np.transpose(pca.components_[0:2, start_idx:end_idx]), labels=X.columns[start_idx:end_idx])
    plt.title(f'Biplot - Attributes {start_idx+1} to {end_idx}')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'biplot_attributes_{start_idx+1}_to_{end_idx}.svg'))
    plt.close()

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Sort attributes based on their contribution to variance
sorted_indices = np.argsort(-explained_variance_ratio)

# Select top 5 attributes
top_5_indices = sorted_indices[:5]
top_5_attributes = X.columns[top_5_indices]

# Plot graph for top 5 attributes
plt.figure(figsize=(8, 6))
myplot(X_pca[:, 0:2], np.transpose(pca.components_[0:2, top_5_indices]), labels=top_5_attributes)
plt.title('Biplot - Top 5 Attributes Contributing to Variance')
plt.tight_layout()
plt.savefig(os.path.join(directory, 'top_5_attributes_variance.svg'))
plt.show()

# Print top 5 attribute names
print("Top 5 attributes contributing to variance:")
for attribute in top_5_attributes:
    print(attribute)
