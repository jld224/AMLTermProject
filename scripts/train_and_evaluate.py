import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor

# Import the custom feature selectors
from src.preprocessing.PCAFS import PCAFeatureSelector
from src.preprocessing.GiniFS import GiniImportanceSelector

def apply_feature_selection(fs_strategy, X, y):
    print(f"{fs_strategy['name']} Started")
    if fs_strategy["name"] == "SelectKBest":
        selector = SelectKBest(mutual_info_regression, k=fs_strategy.get("k", 10))
        selector.fit(X, y)
    elif fs_strategy["name"] == "RFE":
        selector = RFE(RandomForestRegressor(n_estimators=5), n_features_to_select=fs_strategy.get("k", 10))
        selector.fit(X, y)
    elif fs_strategy["name"] == "PCA":
        selector = PCAFeatureSelector(n_components=fs_strategy.get("k", 10))
        selector.fit(X)
    elif fs_strategy["name"] == "GiniImportance":
        selector = GiniImportanceSelector(n_estimators=100, random_state=42)
        selector.fit(X, y)
    else:
        raise ValueError("Invalid feature selection strategy")
    print(f"{fs_strategy['name']} Ended")
    return selector.transform(X), selector

def visualize_feature_importances(selector, title, filename):
    try:
        importances = selector.feature_importances_
        features = range(len(importances))
        plt.figure(figsize=(10, 5))
        plt.title(title)
        sns.barplot(x=features, y=importances)
        plt.savefig(filename)
        plt.show()
    except AttributeError:
        print("No feature importances available for this selector.")

def plot_predictions(y_test, predictions, filename):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, predictions, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Prediction vs Actual')
    plt.savefig(filename)
    plt.show()

def plot_residuals(y_test, predictions, filename):
    residuals = y_test - predictions
    plt.figure(figsize=(10, 5))
    plt.scatter(predictions, residuals, alpha=0.3)
    plt.hlines(0, predictions.min(), predictions.max(), colors='red', linestyles='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.savefig(filename)
    plt.show()

def plot_heatmap_of_weights(weights, title, filename):
    plt.figure(figsize=(10, 5))
    sns.heatmap(weights, annot=False, cmap='coolwarm')
    plt.title(title)
    plt.xlabel('Neurons in Next Layer')
    plt.ylabel('Neurons in Current Layer')
    plt.savefig(filename)
    plt.show()

def plot_regression_heatmap(y_test, predictions, filename):
    """
    Creates a heatmap for regression tasks to show how predictions differ from actual values using pandas.
    Both arrays are binned and a matrix-like visualization is created using crosstab.
    """
    # Convert arrays to pandas Series for easier handling
    y_test_series = pd.Series(y_test, name='Actual Values')
    predictions_series = pd.Series(predictions, name='Predicted Values')
    
    # Create bins
    bins = pd.cut(y_test_series, 20)  # Automatically choose bins based on y_test
    pred_bins = pd.cut(predictions_series, 20)  # Similarly for predictions

    # Create a cross-tabulation to build a 2D histogram
    heatmap_data = pd.crosstab(bins, pred_bins, normalize='index')

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".2f")
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Heatmap of Predicted vs Actual Values')
    plt.savefig(filename)
    plt.show()

def run_experiment(X, y, model_params, feature_selection_strategies):
    X = preprocess_data(X.drop("SMILES", axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    results = []
    for fs_strategy in feature_selection_strategies:
        X_train_fs, selector = apply_feature_selection(fs_strategy, X_train, y_train)
        X_test_fs = selector.transform(X_test)

        model_instance = MLPRegressor(**model_params)
        model_instance.fit(X_train_fs, y_train)
        predictions = model_instance.predict(X_test_fs)

        r2, mse = r2_score(y_test, predictions), mean_squared_error(y_test, predictions)
        results.append((fs_strategy["name"], r2, mse))
        print(f"{fs_strategy['name']}: R2={r2}, MSE={mse}")

        visualize_feature_importances(selector, f"{fs_strategy['name']} Feature Importances", f"feature_importances_{fs_strategy['name']}.svg")
        plot_predictions(y_test, predictions, f"predictions_vs_actual_{fs_strategy['name']}.svg")
        plot_residuals(y_test, predictions, f"residuals_{fs_strategy['name']}.svg")
        plot_heatmap_of_weights(model_instance.coefs_[0], f"Weights from Input to First Hidden Layer - {fs_strategy['name']}", f"weights_heatmap_{fs_strategy['name']}.svg")
        plot_regression_heatmap(y_test, predictions, f"regression_heatmap_{fs_strategy['name']}.svg")

    return results


def preprocess_data(X):
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    
    imputer = SimpleImputer(strategy="mean")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))
    
    return X_scaled

if __name__ == "__main__":
    data = pd.read_csv("./data/processed.csv").dropna(subset=["η / mPa s"])
    X, y = data.drop("η / mPa s", axis=1), data["η / mPa s"].values

    feature_selection_strategies = [
        {"name": "SelectKBest", "k": 50},
        {"name": "PCA", "k": '95%'},  # Retain 95% of variance; adjust based on variance analysis
        {"name": "GiniImportance", "k": 80}
    ]

    model_params = {
        "hidden_layer_sizes": (100, 100),
        "max_iter": 3000,
        "verbose": True
    }

    run_experiment(X, y, model_params, feature_selection_strategies)